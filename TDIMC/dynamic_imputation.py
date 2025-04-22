# dynamic_imputation.py
import numpy as np
from typing import List, Dict
from model import SimpleMultiViewClustering  # 导入同级的Cluster模型
from utils import split_dataset  # 导入数据集划分函数
from sklearn.metrics import silhouette_score, davies_bouldin_score

def should_impute(info, threshold=0.3):
    """
    根据信息量判断是否应该插补
    Args:
        info: 信息量字典或信息量值
        threshold: 信息量阈值
    Returns:
        decision: 是否应该插补的布尔值
    """
    if isinstance(info, dict):
        total_info = info.get('total_info', 0)
    else:
        total_info = info
        
    return total_info >= threshold


class DynamicImputation:
    """动态插补类，根据聚类结果判断信息量并插补缺失值"""
    
    def __init__(self, config: Dict, threshold: float = 0.5, n_clusters: int = 5):
        """
        Args:
            config: 配置字典
            threshold: 信息量阈值，决定是否插补
            n_clusters: 聚类数量
        """
        self.config = config
        self.threshold = threshold
        self.n_clusters = n_clusters
        self.clusterer = Cluster({"n_clusters": n_clusters}, n_clusters=n_clusters)  # 初始化Cluster模型

    def cluster_base_impute(self, X_view, mask, cluster_indices, n_samples=5, reg_param=1e-3):
        """
        基于聚类的局部插补，使用多元高斯分布生成缺失值
        
        Args:
            X_view: 单视图数据
            mask: 掩码 (1表示数据存在，0表示缺失)
            cluster_indices: 簇内样本索引
            n_samples: 采样次数，默认5次取平均
            reg_param: 正则化参数，默认1e-3，防止协方差矩阵奇异
            
        Returns:
            imputed_data: 插补后的数据
        """
        # 只对簇内数据进行局部插补
        cluster_data = X_view[cluster_indices].copy()
        cluster_mask = mask[cluster_indices]
        
        # 将掩码转化为缺失值
        cluster_data_masked = cluster_data.copy()
        cluster_data_masked[cluster_mask == 0] = np.nan
        
        # 获取非缺失样本
        valid_indices = np.where(~np.isnan(cluster_data_masked).any(axis=1))[0]
        
        # 如果有足够的非缺失样本，使用多元高斯分布插补
        if len(valid_indices) > 3:  # 确保有足够样本计算协方差
            # 获取非缺失样本
            valid_data = cluster_data_masked[valid_indices]
            
            # 计算均值和协方差
            mean = np.mean(valid_data, axis=0)
            cov = np.cov(valid_data, rowvar=0)
            
            # 添加正则化参数以确保非奇异
            cov = cov + np.eye(len(cov)) * reg_param
            
            try:
                # 使用Cholesky分解生成多元高斯分布样本
                L = np.linalg.cholesky(cov)
                
                # 对缺失样本进行插补
                for i, has_value in enumerate(cluster_mask):
                    if has_value == 0:  # 样本缺失
                        # 生成n_samples个样本
                        samples = np.zeros((n_samples, cluster_data.shape[1]))
                        rng = np.random.default_rng()
                        
                        for j in range(n_samples):
                            noise = rng.normal(size=cluster_data.shape[1])
                            samples[j] = mean + noise @ L.T
                        
                        # 取均值作为最终插补值
                        cluster_data_masked[i] = np.mean(samples, axis=0)
                
                print(f"使用多元高斯分布插补 {np.sum(cluster_mask == 0)} 个样本")
                return cluster_data_masked
            except np.linalg.LinAlgError as e:
                print(f"Cholesky分解失败: {e}，回退到简单均值插补")
                # 如果Cholesky分解失败，回退到均值插补
                pass
        
        # 样本太少或Cholesky分解失败，使用均值插补
        mean_values = np.nanmean(cluster_data_masked, axis=0)
        
        # 处理可能的NaN值
        if np.isnan(mean_values).any():
            mean_values = np.nan_to_num(mean_values)
        
        for i, row_mask in enumerate(cluster_mask):
            if row_mask == 0:  # 样本缺失
                cluster_data_masked[i] = mean_values
        
        print(f"使用均值插补 {np.sum(cluster_mask == 0)} 个样本")
        return cluster_data_masked
        
    def cross_view_impute(self, X_masked, masks, view_idx, sample_idx):
        """
        使用生成式模型进行跨视图插补
        Args:
            X_masked: 多视图数据列表
            masks: 多视图掩码
            view_idx: 当前视图索引
            sample_idx: 当前样本索引

        Returns:
            imputed_value: 插补值
        """

        pass
        
    
    def impute(self, X_masked: List[np.ndarray], masks: np.ndarray) -> List[np.ndarray]:
        """动态插补缺失值"""
        n_views = len(X_masked)
        n_samples = X_masked[0].shape[0]
        
        # 1. 聚类
        labels = self.clusterer.cluster(X_masked, masks)
        
        # 2. 初始化插补结果
        X_imputed = [x.copy() for x in X_masked]
        
        # 3. 对每个视图的每个样本进行插补
        for view_idx in range(n_views):
            for sample_idx in range(n_samples):
                # 如果不缺失，跳过
                if masks[sample_idx, view_idx] == 1:
                    continue
                    
                cluster_idx = labels[sample_idx]
                cluster_mask = (labels == cluster_idx)
                
                # 计算簇内信息量
                cluster_info = self.calculate_information(
                    X_masked[view_idx][cluster_mask], 
                    masks[cluster_mask, view_idx],
                    labels[cluster_mask],
                    cluster_idx
                )
                
                # 计算视图信息量
                view_info = 0
                view_pred = self.view_correlation_impute(
                    X_masked, masks, view_idx, sample_idx
                )
                if view_pred is not None:
                    view_info = 0.6  # 可以基于相关性动态计算
                
                # 动态选择插补策略
                if max(cluster_info, view_info) < self.threshold:
                    # 信息不足，使用零填充
                    X_imputed[view_idx][sample_idx] = np.zeros(X_masked[view_idx].shape[1])
                    print(f"Sample {sample_idx}, View {view_idx}: Zero imputation (insufficient info)")
                    
                elif cluster_info > view_info:
                    # 使用簇内KNN插补
                    if np.sum(cluster_mask) > 3:  # 确保簇内有足够样本
                        local_imputed = self.knn_impute(
                            X_masked[view_idx], 
                            masks[:, view_idx],
                            np.where(cluster_mask)[0]
                        )
                        # 找到样本在簇内的索引
                        local_idx = np.where(cluster_mask)[0].tolist().index(sample_idx)
                        X_imputed[view_idx][sample_idx] = local_imputed[local_idx]
                        print(f"Sample {sample_idx}, View {view_idx}: KNN imputation (cluster)")
                    else:
                        # 簇太小，使用全局KNN
                        X_imputed[view_idx][sample_idx] = self.global_knn_impute(
                            X_masked[view_idx],
                            masks[:, view_idx],
                            sample_idx
                        )
                        print(f"Sample {sample_idx}, View {view_idx}: KNN imputation (global)")
                        
                else:
                    # 使用视图相关性插补
                    X_imputed[view_idx][sample_idx] = view_pred
                    print(f"Sample {sample_idx}, View {view_idx}: Cross-view imputation")
        
        return X_imputed

if __name__ == "__main__":
    from config import DATA_CONFIG, MISSING_CONFIG
    from datasets import LoadDataset
    from missing_simulator import MissingSimulator
    
    # 加载数据
    loader = LoadDataset(DATA_CONFIG)
    X, Y = loader.load_dataset("Handwritten")
    
    # 生成缺失掩码
    simulator = MissingSimulator(MISSING_CONFIG, missing_rate=0.3)
    masks = simulator.generate_masks(X, mode="base_view")
    X_masked = simulator.apply_masks(X, masks)

    split_ratio = DATA_CONFIG["preprocess"]["split_ratio"]
    (train_X, train_Y, train_sn), _, (test_X, test_Y, test_sn) = split_dataset(X, Y, sn=masks, split_ratio=split_ratio, seed=42)
    
    # 聚类
    clusterer = Cluster({"n_clusters": 5}, n_clusters=5)
    labels = clusterer.cluster(X_masked, masks)
    
    # 计算视图0的所有缺失样本信息量
    view_idx = 0
    missing_info = calculate_view_missing_information(X_masked, masks, view_idx, labels)
    
    # 打印结果
    print(f"视图 {view_idx} 中共有 {len(missing_info['sample_indices'])} 个缺失样本")
    print(f"平均可插补信息量: {np.mean(missing_info['total_info']):.4f}")
    
    # 找出可插补和不可插补的样本
    threshold = 0.5
    imputable_samples = np.sum(missing_info['total_info'] >= threshold)
    print(f"可插补样本数: {imputable_samples} ({imputable_samples/len(missing_info['sample_indices'])*100:.2f}%)")
    print(f"不可插补样本数: {len(missing_info['sample_indices']) - imputable_samples}")
    
    # 测试单个样本
    sample_idx = missing_info['sample_indices'][0]
    sample_info = calculate_imputation_information(X_masked, masks, view_idx, sample_idx, labels)
    print(f"样本 {sample_idx} 信息量: {sample_info['total_info']:.4f}")
    print(f"应该插补: {should_impute(sample_info, threshold)}")

    # 可视化插补性能
    import matplotlib.pyplot as plt
    import seaborn as sns
    