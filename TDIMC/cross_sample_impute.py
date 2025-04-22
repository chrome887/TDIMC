# cross_sample_impute.py
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

def cross_sample_impute(X_views, mask, cluster_indices, n_samples=5, k=10, reg_param=1e-3):
    """
    基于聚类的多视图插补，结合最近邻样本和多元高斯分布生成缺失值

    Args:
        X_views: 多视图数据列表，每个元素是一个视图的数据矩阵 (n_samples, n_features)
        mask: 缺失掩码矩阵 (n_samples, n_views)，1表示数据存在，0表示缺失
        cluster_indices: 当前簇内样本的索引
        n_samples: 每个缺失样本生成的采样次数
        k: 最近邻样本的数量
        reg_param: 正则化参数，防止协方差矩阵奇异

    Returns:
        imputed_views: 插补后的多视图数据列表
    """
    view_num = len(X_views)
    cluster_data = [X_view[cluster_indices].copy() for X_view in X_views]
    cluster_mask = mask[cluster_indices]

    # 计算每个视图中所有样本之间的欧氏距离
    print("计算视图间距离矩阵...")
    dist_all_set = [cdist(cluster_data[v], cluster_data[v], 'euclidean') for v in range(view_num)]

    # 初始化插补后的数据
    imputed_views = [data.copy() for data in cluster_data]

    # 遍历每个视图
    for v in range(view_num):
        print(f"处理视图 {v + 1}/{view_num} 的缺失值...")
        missing_indices = np.where(cluster_mask[:, v] == 0)[0]  # 找到当前视图缺失的样本索引
        present_indices = np.where(cluster_mask[:, v] == 1)[0]  # 找到当前视图存在的样本索引

        if len(present_indices) < 3:
            print(f"视图 {v} 的有效样本不足，跳过插补...")
            continue

        # 获取当前视图的有效样本
        valid_data = cluster_data[v][present_indices]

        # 计算均值和协方差
        mean = np.mean(valid_data, axis=0)
        cov = np.cov(valid_data, rowvar=0) + np.eye(valid_data.shape[1]) * reg_param

        # 对每个缺失样本进行插补
        for idx in missing_indices:
            # 找到最近邻样本
            distances = dist_all_set[v][idx, present_indices]
            nearest_indices = present_indices[np.argsort(distances)[:k]]
            neighbors = cluster_data[v][nearest_indices]

            # 使用最近邻样本计算均值和协方差
            neighbor_mean = np.mean(neighbors, axis=0)
            neighbor_cov = np.cov(neighbors, rowvar=0) + np.eye(neighbors.shape[1]) * reg_param

            try:
                # 使用Cholesky分解生成多元高斯分布样本
                L = np.linalg.cholesky(neighbor_cov)
                rng = np.random.default_rng()
                samples = rng.normal(size=(n_samples, neighbor_cov.shape[0])) @ L.T + neighbor_mean

                # 取均值作为插补值
                imputed_views[v][idx] = np.mean(samples, axis=0)
            except np.linalg.LinAlgError as e:
                print(f"视图 {v} 的样本 {idx} 的Cholesky分解失败，使用简单均值插补...")
                imputed_views[v][idx] = neighbor_mean

    print("插补完成")
    return imputed_views

# 修复后的插补代码
if __name__ == "__main__":
    # 示例数据
    from utils import DatasetLoader,get_default_config
    from missing_simulator import load_sn
    import os

    config = get_default_config("Handwritten")
    loader = DatasetLoader(config)

    X_views, Y = loader.load_dataset("Handwritten")
    np.random.seed(42)

    missing_rate = 0.5
    sn_path = os.path.join("sn", f"Handwritten_mcar_{missing_rate:.1f}".replace(".", "") + ".csv")
    mask = load_sn(sn_path)
    n_clusters = config['classes']
    n_views = len(X_views)
    
    # 确保每个样本至少有一个视图存在
    valid_samples = np.sum(mask, axis=1) > 0
    if not np.all(valid_samples):
        print(f"警告：有{np.sum(~valid_samples)}个样本在所有视图都缺失，这些样本将被移除")
        for v in range(n_views):
            X_views[v] = X_views[v][valid_samples]
        mask = mask[valid_samples]
    
    print(f"共有{len(X_views[0])}个有效样本进行聚类")
    
    # 拼接所有非缺失视图数据进行聚类
    combined_data = []
    for i in range(len(X_views[0])):
        sample_features = []
        for v in range(n_views):
            if mask[i, v] == 1:
                sample_features.append(X_views[v][i])
            else:
                # 如果视图缺失，用零向量填充
                sample_features.append(np.zeros(X_views[v].shape[1]))
        combined_feature = np.concatenate(sample_features)
        combined_data.append(combined_feature)
    
    # 转换为二维数组
    combined_data = np.array(combined_data)
    
    # 执行K均值聚类
    print("对所有视图的组合特征进行聚类...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(combined_data)
    
    # 为每个簇创建样本索引列表
    all_cluster_indices = []
    for cluster_id in range(n_clusters):
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        all_cluster_indices.append(cluster_indices)
        print(f"簇 {cluster_id} 包含 {len(cluster_indices)} 个样本")
    
    # 为了演示，我们选择第一个簇
    cluster_indices = all_cluster_indices[0]
    print(f"选择了簇 0 的 {len(cluster_indices)} 个样本进行插补示例")

    # 调用函数进行插补
    imputed_views_partial = cluster_base_impute_with_samples(X_views, mask, cluster_indices)

    # 初始化插补后的完整数据
    imputed_views = [view.copy() for view in X_views]
    
    # 将插补结果填充回原始数据
    for v in range(n_views):
        imputed_views[v][cluster_indices] = imputed_views_partial[v]

    # 打印插补后的结果
    for i, imputed_view in enumerate(imputed_views):
        print(f"Imputed view {i + 1}:")
        print(imputed_view)
        print("Shape:", imputed_view.shape)
        print("-" * 30)
    print("所有视图插补完成")

    # 计算所有视图的MSE
    total_mse = 0
    total_count = 0
    for v in range(n_views):
        # 只计算缺失的部分
        missing_indices = np.where(mask[:, v] == 0)[0]
        if len(missing_indices) > 0:
            view_mse = np.mean((X_views[v][missing_indices] - imputed_views[v][missing_indices]) ** 2)
            total_mse += view_mse * len(missing_indices)
            total_count += len(missing_indices)
            print(f"视图 {v} 缺失部分的均方误差 (MSE): {view_mse:.6f}")
    
    # 计算平均MSE
    avg_mse = total_mse / total_count if total_count > 0 else 0
    print(f"所有视图缺失部分的平均均方误差 (MSE): {avg_mse:.6f}")
    
    # 计算0填充的均方误差
    total_zero_mse = 0
    for v in range(n_views):
        missing_indices = np.where(mask[:, v] == 0)[0]
        if len(missing_indices) > 0:
            # 对缺失部分使用0填充
            zero_filled = np.zeros_like(X_views[v][missing_indices])
            view_zero_mse = np.mean((X_views[v][missing_indices] - zero_filled) ** 2)
            total_zero_mse += view_zero_mse * len(missing_indices)
            print(f"视图 {v} 0填充的均方误差 (MSE): {view_zero_mse:.6f}")
    
    # 计算平均0填充MSE
    avg_zero_mse = total_zero_mse / total_count if total_count > 0 else 0
    print(f"所有视图0填充的平均均方误差 (MSE): {avg_zero_mse:.6f}")
    print("评估完成")