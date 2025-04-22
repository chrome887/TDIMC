# missing_simulator.py
import numpy as np
from typing import List, Dict
from sklearn.preprocessing import OneHotEncoder
from utils import DatasetLoader
import os
import argparse


class MissingSimulator:
    """模拟多视图数据中的缺失模式"""
    
    def __init__(self, config: Dict, seed: int = 42):
        """
        初始化缺失模拟器
        Args:
            config: 缺失配置（如 MISSING_CONFIG）
            missing_rate: 总体缺失率
            seed: 随机种子
        """
        self.config = config
        self.missing_rate = config['missing_rate']
        self.mode = config['missing_mode']
        self.seed = seed
        np.random.seed(seed)

    def generate_masks(self, X: List[np.ndarray]) -> List[np.ndarray]:
        """
        生成缺失掩码
        Args:
            X: 多视图数据列表，每个视图为 [n_samples, n_features]
            mode: 缺失模式 ("base_view", "mcar")
        Returns:
            masks: 缺失掩码列表，1表示保留，0表示缺失
        """
        n_views = len(X)
        n_samples = X[0].shape[0]
        
        if self.mode == "base_view":
            masks = self._generate_base_view_masks(X, n_samples, n_views)
        
        elif self.mode == "mcar":
            # MCAR：所有视图按missing_rate缺失
            masks = self._get_sn(n_views, n_samples, self.missing_rate)
        
        else:
            raise ValueError(f"Unsupported missing mode: {self.mode}")
        
        # 调试：打印掩码样本
        print(f"Mask sample for view 0: {masks[:5, 0]}")
        
        return masks

    def _generate_base_view_masks(self, X: List[np.ndarray], n_samples: int, n_views: int) -> np.ndarray:
        """生成基础视角模式的掩码：基础视图不缺失，其他视图随机缺失后调整"""
        # 机制1：选择信息量最少的视图作为基础视图
        base_view_idx = self._select_base_view(X)
        
        # 初始掩码：基础视图不缺失，其他视图按missing_rate随机缺失
        base_masks = self._get_sn(1, n_samples, 0.0)  # 基础视图不缺失
        other_masks = self._get_sn(n_views - 1, n_samples, self.missing_rate)

        initial_masks = np.zeros((n_samples, n_views), dtype=np.float32)  # 初始化掩码矩阵
        initial_masks[:, base_view_idx] = base_masks.squeeze(1)   # 放入基础视图（不缺失）
        
        remaining_indices = list(range(n_views))
        remaining_indices.pop(base_view_idx)  # 剩下的视图索引
        initial_masks[:, remaining_indices] = other_masks  # 其他视图按缺失率设置
        
        # 机制2：检测基础视图异常并调整其他视图掩码（样本级别）
        anomaly_scores = self._detect_anomaly_statistical(X[base_view_idx])
        adjusted_masks = self._adjust_masks(initial_masks, base_view_idx, anomaly_scores)
        
        return adjusted_masks

    def _select_base_view(self, X: List[np.ndarray]) -> int:
        """选择信息量最少的视图作为基础视图（基于熵）"""
        entropies = []
        for view in X:
            std = np.std(view, axis=0)
            entropy = np.mean(-std * np.log(std + 1e-10))
            entropies.append(entropy)
        return np.argmin(entropies)

    def _detect_anomaly_statistical(self, view: np.ndarray, k: float = 1.5) -> np.ndarray:
        """方案1：样本级别的异常检测，返回每个样本的异常分数（0-1）
        Args:
        view: 视图数据
        k: Z分数阈值
        target_ratio: 目标异常样本比例
        """
        # k=2.0: 约4.6%样本异常
        # k=1.5: 约13.4%样本异常
        # k=1.0: 约31.7%样本异常
        # k=0.7: 约48.4%样本异常
        '''
        医疗场景：
            常规检查异常比例：约10-20%（k=1.3-1.0）
            高风险人群检查：约20-30%（k=1.0-0.8）
        商品评价场景：
            极端评分比例：约20-40%（k=1.0-0.7）
            不满意评价导致评论：约25%（k=0.9）
        一般异常检测：
            低异常率：5-10%（k=1.7-1.5）
            中等异常率：10-20%（k=1.5-1.0）
            高异常率：20-40%（k=1.0-0.7）
        '''
        mean = np.mean(view, axis=0)
        std = np.std(view, axis=0) + 1e-10
        z_scores = np.abs((view - mean) / std)  # [n_samples, n_features]
        anomaly_mask = z_scores > k  # [n_samples, n_features]
        # 每个样本的异常分数：异常特征的比例
        anomaly_scores = np.mean(anomaly_mask, axis=1)  # [n_samples]
        return anomaly_scores

    def _detect_anomaly_rule_based(self, view: np.ndarray, rules: Dict = None) -> np.ndarray:
        """方案2：预留接口，样本级别的预定义规则异常检测"""
        if rules is None:
            raise NotImplementedError("Rule-based anomaly detection not implemented yet.")
        return np.zeros(view.shape[0])

    def _adjust_masks(self, masks: np.ndarray, base_view_idx: int, anomaly_scores: np.ndarray) -> np.ndarray:
        """根据基础视图样本级异常程度调整其他视图的掩码"""
        adjusted_masks = masks.copy()
        # 基础视图不缺失
        adjusted_masks[:, base_view_idx] = 1.0
        # 其他视图根据样本异常程度减少缺失
        for i in range(masks.shape[1]):
            if i != base_view_idx:
                mask = adjusted_masks[:, i]
                zeros = mask == 0
                n_zeros = np.sum(zeros)
                if n_zeros > 0:
                    # 每个样本的翻转数量根据其异常分数调整
                    n_to_flip_per_sample = (zeros * anomaly_scores > np.random.rand(zeros.shape[0])).astype(np.int_)
                    for sample_idx in range(masks.shape[0]):
                        if zeros[sample_idx] and n_to_flip_per_sample[sample_idx] > 0:
                            adjusted_masks[sample_idx, i] = 1.0
        return adjusted_masks

    def _get_sn(self, view_num: int, alldata_len: int, missing_rate: float) -> np.ndarray:
        """随机生成不完整视图掩码"""
        one_rate = 1 - missing_rate
        if one_rate <= (1 / view_num):
            enc = OneHotEncoder()
            view_preserve = enc.fit_transform(np.random.randint(0, view_num, size=(alldata_len, 1))).toarray()
            return view_preserve
        if one_rate == 1:
            return np.ones((alldata_len, view_num), dtype=np.int_)
        
        error = 1
        while error >= 0.005:
            enc = OneHotEncoder()
            view_preserve = enc.fit_transform(np.random.randint(0, view_num, size=(alldata_len, 1))).toarray()
            one_num = view_num * alldata_len * one_rate - alldata_len
            ratio = one_num / (view_num * alldata_len)
            matrix_iter = (np.random.randint(0, 100, size=(alldata_len, view_num)) < int(ratio * 100)).astype(np.int_)
            a = np.sum(((matrix_iter + view_preserve) > 1).astype(np.int_))
            one_num_iter = one_num / (1 - a / one_num) if a < one_num else one_num
            ratio = one_num_iter / (view_num * alldata_len)
            matrix_iter = (np.random.randint(0, 100, size=(alldata_len, view_num)) < int(ratio * 100)).astype(np.int_)
            matrix = ((matrix_iter + view_preserve) > 0).astype(np.int_)
            ratio = np.sum(matrix) / (view_num * alldata_len)
            error = abs(one_rate - ratio)
        return matrix.astype(np.float32)

    def apply_masks(self, X: List[np.ndarray], masks: np.ndarray) -> List[np.ndarray]:
        """应用掩码到数据上，缺失部分置为NaN
        Args:
            X: 原始多视图数据列表 [view_1, view_2, ..., view_n]
            masks: 掩码矩阵 [n_samples, n_views]
        Returns:
            masked_X: 应用掩码后的数据列表
        """
        masked_X = []
        for view_idx, view_data in enumerate(X):
            # 提取当前视图的掩码列
            view_mask = masks[:, view_idx]
            # 将列向量扩展为与视图数据相匹配的形状
            expanded_mask = np.expand_dims(view_mask, axis=1)
            expanded_mask = np.tile(expanded_mask, (1, view_data.shape[1]))
            # 应用掩码
            masked_view = view_data * expanded_mask + (1 - expanded_mask) * np.nan
            masked_X.append(masked_view)
        return masked_X

    # def _adjust_ratios(self, base_ratios: List[int], n_views: int) -> List[int]:
    #     """动态调整比例以匹配视图数量（注释掉，保留供扩展）"""
    #     if len(base_ratios) == n_views:
    #         return base_ratios
    #     elif len(base_ratios) < n_views:
    #         return base_ratios + [base_ratios[-1]] * (n_views - len(base_ratios))
    #     else:
    #         return base_ratios[:n_views]

    # def _compute_view_missing_rates(self, ratios: List[int], total_missing_rate: float) -> List[float]:
    #     """从高到低动态计算每个视图的缺失率，最高不超过0.5（注释掉，保留供扩展）"""
    #     n_views = len(ratios)
    #     sorted_ratios = sorted(ratios, reverse=True)
    #     max_rate = 0.5
    #     
    #     total_ratio = sum(sorted_ratios)
    #     avg_missing_rate = total_missing_rate / n_views
    #     
    #     view_rates = []
    #     remaining_rate = total_missing_rate
    #     remaining_views = n_views
    #     
    #     for i, r in enumerate(sorted_ratios):
    #         if i == n_views - 1:
    #             rate = remaining_rate / remaining_views
    #         else:
    #             rate = min(avg_missing_rate * r / (total_ratio / n_views), max_rate)
    #             if rate * remaining_views > remaining_rate:
    #                 rate = remaining_rate / remaining_views
    #         view_rates.append(rate)
    #         remaining_rate -= rate
    #         remaining_views -= 1
    #     
    #     view_rates = [min(max(r, 0.0), max_rate) for r in view_rates]
    #     original_order = np.argsort(np.argsort(ratios))
    #     return [view_rates[i] for i in original_order]

    # def _generate_heterogeneous_masks(self, n_samples: int, n_views: int, view_missing_rates: List[float]) -> np.ndarray:
    #     """生成异质缺失模式的掩码（注释掉，保留供扩展）"""
    #     clusters = self._simulate_clusters(n_samples, len(view_missing_rates))
    #     return self._get_sn(n_views, n_samples, view_missing_rates[0])

def save_sn(masks: np.ndarray, filepath: str) -> None:
    """保存掩码矩阵到文件
    Args:
        masks: 掩码矩阵 [n_samples, n_views]
        filepath: 保存路径，如 'sn/masks.csv'
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    np.savetxt(filepath, masks, delimiter=",", fmt="%d")
    print(f"Successfully saved sn to {filepath}")

def load_sn(filepath: str) -> np.ndarray:
    """从文件加载掩码矩阵
    Args:
        filepath: 掩码文件路径，如 'sn/masks.csv'
    Returns:
        masks: 加载的掩码矩阵 [n_samples, n_views]
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"sn not found at {filepath}")
    masks = np.loadtxt(filepath, delimiter=",")
    print(f"Successfully loaded sn from {filepath}")
    return masks

# 测试代码
if __name__ == "__main__":
    dataset = {
    0: "Handwritten",
    1: "Scene15",
    }
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=int, default='1', help='dataset id')
    parser.add_argument('--devices', type=str, default='0', help='gpu device ids')
    parser.add_argument('--print_num', type=int, default='50', help='gap of print evaluations')
    parser.add_argument('--test_time', type=int, default='5', help='number of test times')
    parser.add_argument('--missing_rate', type=float, default='0.5', help='missing rate')
    parser.add_argument('--missing_mode', type=str, default='base_view', help='missing mode')

    args = parser.parse_args()

    dataset = dataset[args.dataset]
    # configure
    from utils import get_default_config, get_logger
    config = get_default_config(dataset)
    config['missing_rate'] = args.missing_rate
    config['print_num'] = args.print_num
    config['dataset'] = dataset
    config['missing_mode'] = args.missing_mode
    logger, plt_name = get_logger(config)

    logger.info('Dataset:' + str(dataset))
    for (k, v) in config.items():
        if isinstance(v, dict):
            logger.info("%s={" % (k))
            for (g, z) in v.items():
                logger.info("          %s = %s" % (g, z))
        else:
            logger.info("%s = %s" % (k, v))
    
    # load dataset
    loader = DatasetLoader(config)
    X_list, Y_list = loader.load_dataset(config['dataset'])
    
    # 初始化模拟器并生成掩码
    simulator = MissingSimulator(config)
    masks = simulator.generate_masks(X_list)
    masked_X = simulator.apply_masks(X_list, masks)
    
    # 获取保存路径
    filepath = os.path.join("sn", f"{config['dataset']}_{config['missing_mode']}_{config['missing_rate']:.1f}".replace(".", "") + ".csv")
    save_sn(masks, filepath)
    
    # filepath = os.path.join("sn", f"{dataname}_{args.mode}_{args.missing_rate:.1f}".replace(".", "") + ".csv")
    # # 加载掩码并打印输出信息
    # masks = load_sn(filepath)
    # print("Loaded masks shape:", masks.shape)
    # print("Sample of loaded masks:", masks[:5])

    # 统计各个视角缺失视图数和总缺失率
    n_views = masks.shape[1]
    n_samples = masks.shape[0]
    for view_idx in range(n_views):
        missing_count = n_samples - np.sum(masks[:, view_idx])
        missing_rate = missing_count / n_samples
        print(f"View {view_idx}: Missing count = {missing_count}, Missing rate = {missing_rate:.2%}")
    
    total_missing_count = np.sum(masks == 0)
    total_missing_rate = total_missing_count / (n_samples * n_views)
    print(f"Total missing count = {total_missing_count}, Total missing rate = {total_missing_rate:.2%}")

    # 结合标签统计不同簇的缺失率以及样本缺失情况的单独值
    unique_labels = np.unique(Y_list)
    for label in unique_labels:
        label_indices = np.where(Y_list == label)[0]
        label_masks = masks[label_indices]
        n_label_samples = label_masks.shape[0]
        for view_idx in range(n_views):
            missing_count = n_label_samples - np.sum(label_masks[:, view_idx])
            missing_rate = missing_count / n_label_samples
            print(f"Label {label}, View {view_idx}: Missing count = {missing_count}, Missing rate = {missing_rate:.2%}")
    # 统计每个样本的缺失视图数
    sample_missing_counts = np.sum(masks == 0, axis=1)
    unique_missing_counts, counts = np.unique(sample_missing_counts, return_counts=True)
    for missing_count, count in zip(unique_missing_counts, counts):
        print(f"Samples with {missing_count} missing views: {count}")