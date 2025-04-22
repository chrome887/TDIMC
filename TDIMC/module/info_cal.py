# info_cal.py
import os
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import mutual_info_score
from sklearn.decomposition import PCA

def estimate_mutual_info(X_v, X_u, n_bins=20):
    """
    估算视图间互信息（离散化处理）。
    
    Args:
        X_v, X_u: 视图数据，形状 [num_samples, feature_dim_v/u]。
        n_bins: 离散化分箱数。
    
    Returns:
        mi: 互信息值。
    """
    # 排除缺失值
    valid_mask = ~np.isnan(X_v).any(axis=1) & ~np.isnan(X_u).any(axis=1)
    X_v_valid = X_v[valid_mask]
    X_u_valid = X_u[valid_mask]
    if len(X_v_valid) < 2:
        return 0.0
    
    # 离散化处理
    X_v_flat = np.mean(X_v_valid, axis=1)
    X_u_flat = np.mean(X_u_valid, axis=1)
    
    X_v_bins = np.histogram(X_v_flat, bins=n_bins, density=True)[1]
    X_u_bins = np.histogram(X_u_flat, bins=n_bins, density=True)[1]
    X_v_digitized = np.digitize(X_v_flat, bins=X_v_bins[:-1])
    X_u_digitized = np.digitize(X_u_flat, bins=X_u_bins[:-1])
    
    return mutual_info_score(X_v_digitized, X_u_digitized)


def compute_similarity_matrix(data, sn):
    """
    计算基于共同非缺失视图的样本间相似度矩阵
    """
    num_samples, num_views = sn.shape
    sim_matrix = np.zeros((num_samples, num_samples))
    
    # 对每对样本分别计算相似度
    for i in range(num_samples):
        for j in range(i, num_samples):  # 对称矩阵，只需计算上三角
            # 找出两个样本共同非缺失的视图
            common_views = np.where((sn[i] == 1) & (sn[j] == 1))[0]
            
            if len(common_views) == 0:
                # 没有共同视图时，相似度为0
                continue
                
            # 只在共同非缺失的视图上计算相似度
            features_i = []
            features_j = []
            for v in common_views:
                features_i.append(data[v][i])
                features_j.append(data[v][j])
                
            # 拼接共同视图的特征
            features_i = np.concatenate(features_i)
            features_j = np.concatenate(features_j)
            
            # 计算余弦相似度
            dot_product = np.sum(features_i * features_j)
            norm_i = np.sqrt(np.sum(features_i ** 2) + 1e-6)
            norm_j = np.sqrt(np.sum(features_j ** 2) + 1e-6)
            similarity = dot_product / (norm_i * norm_j)
            
            # 存储相似度（对称矩阵）
            sim_matrix[i, j] = sim_matrix[j, i] = max(0, min(similarity, 1))
    
    return sim_matrix

def cross_sample_info(sn, data, cluster_labels, knn_indices, similarity_matrix, k=5, alpha=0.5):
    """
    计算同视图信息量 I_sv(i, v)，基于局部熵减和图平滑性。
    
    Args:
        sn: 缺失索引矩阵，[num_samples, num_views]。
        data: 未插补数据，列表 [X_1, ..., X_num_views]。
        cluster_labels: 聚类标签，[num_samples]。
        knn_indices: k 近邻索引，[num_samples, k]。
        similarity_matrix: 样本间相似度矩阵，[num_samples, num_samples]。
        k: 近邻数。
        alpha: 熵减与平滑性的权重。
    
    Returns:
        info: 同视图信息量，[num_samples, num_views]。
    """
    num_samples, num_views = sn.shape
    info = np.zeros((num_samples, num_views))
    
    for v in range(num_views):
        # 仅对缺失位置计算
        missing_mask = sn[:, v] == 0
        if not np.any(missing_mask):
            continue
        
        # 1. 局部熵减
        entropy_info = np.zeros(num_samples)
        # 使用 k 近邻计算观测率
        knn_valid = sn[knn_indices, v]  # [num_samples, k]
        p_v = np.mean(knn_valid, axis=1)  # [num_samples]
        p_v = np.clip(p_v, 1e-6, 1 - 1e-6)  # 防止 log(0)
        H_v = - (p_v * np.log2(p_v) + (1 - p_v) * np.log2(1 - p_v))
        entropy_info = np.log2(2) - H_v  # 最大熵为 log(2)
        
        # 2. 局部结构一致性（图平滑性）
        smooth_info = np.zeros(num_samples)
        for i in range(num_samples):
            if not missing_mask[i]:
                continue
            # 结合 k 近邻和簇内样本
            knn_idx = knn_indices[i]  # [k]
            cluster_mask = cluster_labels == cluster_labels[i]
            neighbors = np.unique(np.concatenate([knn_idx, np.where(cluster_mask)[0]]))
            if len(neighbors) == 0:
                continue
            # 邻居的观测情况加权相似度
            neighbor_obs = sn[neighbors, v]  # [num_neighbors]
            neighbor_sim = similarity_matrix[i, neighbors]  # [num_neighbors]
            smooth_info[i] = np.sum(neighbor_sim * neighbor_obs) / (np.sum(neighbor_sim) + 1e-6)
        
        # 组合
        info[:, v] = alpha * entropy_info + (1 - alpha) * smooth_info
        info[~missing_mask, v] = 0  # 非缺失处清零
    
    return info, entropy_info, smooth_info

def cross_view_info(sn, data, view_corr=None):
    """
    计算跨视图信息量 I_cv(i, v)，基于视图互信息和观测率。
    
    Args:
        sn: 缺失索引矩阵，[num_samples, num_views]。
        data: 未插补数据，列表 [X_1, ..., X_num_views]。
        view_corr: 视图间互信息矩阵，[num_views, num_views]，若无则估算。
    
    Returns:
        info: 跨视图信息量，[num_samples, num_views]。
    """
    num_samples, num_views = sn.shape
    info = np.zeros((num_samples, num_views))
    
    # 估算视图间互信息
    if view_corr is None:
        view_corr = np.zeros((num_views, num_views))
        for v in range(num_views):
            for u in range(v + 1, num_views):
                mi = estimate_mutual_info(data[v], data[u])
                view_corr[v, u] = view_corr[u, v] = mi
        view_corr /= (view_corr.max() + 1e-6)  # 归一化
    
    # 计算跨视图信息量
    for v in range(num_views):
        missing_mask = sn[:, v] == 0
        if not np.any(missing_mask):
            continue
        
        # 其他视图的互信息加权观测情况
        other_views = np.arange(num_views) != v
        mi_v = view_corr[v, other_views]  # [num_views-1]
        for i in range(num_samples):
            if not missing_mask[i]:
                continue
            obs_vp = sn[i, other_views]  # [num_views-1]
            info[i, v] = np.sum(mi_v * obs_vp)
    
    return info

def total_info(cross_sample, cross_view, epsilon=1e-6):
    """
    计算总信息量 I_total(i, v)，采用调和平均。
    
    Args:
        cross_sample: 同视图信息量，[num_samples, num_views]。
        cross_view: 跨视图信息量，[num_samples, num_views]。
        epsilon: 防止除零的小常数。
    
    Returns:
        info: 总信息量，[num_samples, num_views]。
    """
    total = 2 * cross_sample * cross_view / (cross_sample + cross_view + epsilon)
    return total

def info_cal(sn, data, cluster_labels, knn_indices, similarity_matrix=None, k=5, alpha=0.5, view_corr=None):
    """
    主函数，计算插补信息量。
    
    Args:
        sn: 缺失索引矩阵，[num_samples, num_views]。
        data: 未插补数据，列表 [X_1, ..., X_num_views]。
        cluster_labels: 聚类标签，[num_samples]。
        knn_indices: k 近邻索引，[num_samples, k]。
        similarity_matrix: 样本间相似度矩阵，[num_samples, num_samples]，若无则计算。
        k: 近邻数。
        alpha: 同视图信息量中熵减与平滑性的权重。
        view_corr: 视图间互信息矩阵，[num_views, num_views]，可选。
    
    Returns:
        cross_sample_info: 同视图信息量，[num_samples, num_views]。
        cross_view_info: 跨视图信息量，[num_samples, num_views]。
        total_info: 总信息量，[num_samples, num_views]。
        obs_view_count: 可观测视图数，[num_samples]。
    """
    if similarity_matrix is None:
        similarity_matrix = compute_similarity_matrix(data, sn)
    
    cross_sample, entropy_info, smooth_info = cross_sample_info(sn, data, cluster_labels, knn_indices, similarity_matrix, k, alpha)
    cross_view = cross_view_info(sn, data, view_corr)
    total = total_info(cross_sample, cross_view)
    obs_view_count = np.sum(sn, axis=1)  # [num_samples]
    
    return cross_sample, cross_view, total, obs_view_count, entropy_info, smooth_info


# 示例用法
if __name__ == "__main__":
    # 示例数据
    from utils import DatasetLoader,get_default_config
    from missing_simulator import load_sn
    import os
    from sklearn.cluster import KMeans
    from sklearn.neighbors import NearestNeighbors

    # 定义参数
    k = 10  # 近邻数
    alpha = 0.5  # 熵减与平滑性的权重

    # 加载数据集
    # dataset = "Handwritten"
    dataset = "Scene15"
    config = get_default_config(dataset)
    loader = DatasetLoader(config)

    X_views, Y = loader.load_dataset(dataset)
    np.random.seed(42)

    missing_rate = 0.5
    sn_path = os.path.join("sn", f"{dataset}_mcar_{missing_rate:.1f}".replace(".", "") + ".csv")
    sn = load_sn(sn_path)
    # 应用缺失掩码生成 data
    data = []
    num_samples = len(sn)
    num_views = len(X_views)

    for v, X_v in enumerate(X_views):
        # 创建一个新的数组，用于存储应用了缺失掩码的视图数据
        X_v_missing = X_v.copy()
        # 根据缺失掩码设置缺失值为 NaN
        missing_indices = np.where(sn[:, v] == 0)[0]
        X_v_missing[missing_indices] = np.nan
        data.append(X_v_missing)

    # 使用KMeans进行聚类
    n_clusters = config['classes']
    # 对每个视图中的非缺失样本进行特征归一化
    data_for_clustering = []
    for v in range(len(X_views)):
        # 提取非缺失样本
        valid_samples = np.where(sn[:, v] == 1)[0]
        if len(valid_samples) > 0:
            # 归一化特征
            X_valid = X_views[v][valid_samples]
            X_norm = (X_valid - np.mean(X_valid, axis=0)) / (np.std(X_valid, axis=0) + 1e-8)
            data_for_clustering.append((valid_samples, X_norm))

    # 聚类基于有效样本
    all_features = np.zeros((num_samples, 0))
    all_valid_mask = np.zeros(num_samples, dtype=bool)

    for valid_samples, X_norm in data_for_clustering:
        temp_features = np.zeros((num_samples, X_norm.shape[1]))
        temp_features[valid_samples] = X_norm
        all_features = np.hstack((all_features, temp_features))
        all_valid_mask |= np.isin(np.arange(num_samples), valid_samples)

    # 在有效样本上执行聚类
    valid_indices = np.where(all_valid_mask)[0]
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    valid_cluster_labels = kmeans.fit_predict(all_features[valid_indices])

    # 将聚类标签分配给所有样本
    cluster_labels = np.zeros(num_samples, dtype=int)
    cluster_labels[valid_indices] = valid_cluster_labels

    # 查看每个簇的样本数，验证效果
    unique, counts = np.unique(cluster_labels, return_counts=True)
    print("各簇样本数：", dict(zip(unique, counts)))

    # 计算K近邻，考虑缺失数据
    # 创建相似度矩阵
    similarity_matrix = compute_similarity_matrix(data, sn)
    
    # 从相似度矩阵计算距离矩阵
    distance_matrix = 1 - similarity_matrix
    np.fill_diagonal(distance_matrix, np.inf)  # 排除自身作为近邻
    
    # 对每个样本找k近邻
    knn_indices = np.zeros((num_samples, k), dtype=int)
    for i in range(num_samples):
        # 选择最相似的k个样本作为近邻
        knn_indices[i] = np.argsort(distance_matrix[i])[:k]
    
    n_clusters = config['classes']
    n_views = len(X_views)
    
    # 计算
    cross_sample, cross_view, total, obs_count, entropy_info, smooth_info = info_cal(
        sn, data, cluster_labels, knn_indices, similarity_matrix, k=k, alpha=0.5
    )
    # 拼接所有信息
    info_all = np.dstack([cross_sample, cross_view, total, sn])  # [num_samples, num_views, 4]
    
    # 显示结果
    print(f"信息矩阵维度: {info_all.shape}")
    print("信息矩阵前3行:")
    for i in range(min(3, num_samples)):
        print(f"样本 {i}:")
        for v in range(num_views):
            print(f"  视图 {v}: 同视图信息={info_all[i,v,0]:.4f}, " + 
                  f"跨视图信息={info_all[i,v,1]:.4f}, " +
                  f"总信息={info_all[i,v,2]:.4f}, " +
                  f"观测状态={int(info_all[i,v,3])}")
    
    # 选择一个簇中的样本
    unique_clusters = np.unique(cluster_labels)
    selected_cluster = unique_clusters[0]  # 选择第一个簇，可以根据需要更改
    samples_in_cluster = np.where(cluster_labels == selected_cluster)[0]
    
    # 从所选簇中取三个样本
    selected_samples = samples_in_cluster[:3] if len(samples_in_cluster) >= 3 else samples_in_cluster
    
    print(f"\n从簇 {selected_cluster} 中选择样本 {selected_samples} 的缺失数据示例及信息来源分析:")
    
    # 分析选定样本的所有缺失数据
    for i in selected_samples:
        for v in range(num_views):
            if sn[i, v] == 0:  # 只有缺失数据
                # 计算同视图来源：k近邻和同簇中观测到的样本数
                knn_obs_count = np.sum(sn[knn_indices[i], v])
                cluster_mask = cluster_labels == cluster_labels[i]
                cluster_obs_count = np.sum(sn[cluster_mask, v]) - (0 if not cluster_mask[i] else sn[i, v])
                
                # 计算跨视图来源：其他可观测视图数
                cross_view_obs_count = np.sum(sn[i, :])  # 当前样本的可观测视图数
                
                print(f"样本 {i}, 视图 {v}: " +
                      f"同视图信息={info_all[i,v,0]:.4f} (熵减信息={entropy_info[i]:.4f}, 平滑信息={smooth_info[i]:.4f}, " +
                      f"k近邻可观测={knn_obs_count}/{k}, 同簇可观测={cluster_obs_count}), " +
                      f"跨视图信息={info_all[i,v,1]:.4f} (可观测视图数={cross_view_obs_count}/{num_views-1}), " +
                      f"总信息={info_all[i,v,2]:.4f}")
    
    # 分析信息量与可用数据量的相关性
    print("\n信息量与可用数据量相关性分析:")
    missing_idxs = [(i, v) for i in range(num_samples) for v in range(num_views) if sn[i, v] == 0]
    i_list, v_list = zip(*missing_idxs)
    knn_obs_counts = np.array([np.sum(sn[knn_indices[i], v]) for i, v in missing_idxs])
    cross_view_obs_counts = np.array([np.sum(sn[i, :]) for i, _ in missing_idxs])
    
    cross_sample_info_values = np.array([info_all[i, v, 0] for i, v in missing_idxs])
    cross_view_info_values = np.array([info_all[i, v, 1] for i, v in missing_idxs])
    entropy_info_values = np.array([entropy_info[i] for i, _ in missing_idxs])
    smooth_info_values = np.array([smooth_info[i] for i, _ in missing_idxs])
    
    print(f"同视图信息与k近邻可观测数相关性: {np.corrcoef(knn_obs_counts, cross_sample_info_values)[0,1]:.4f}")
    print(f"熵减信息与k近邻可观测数相关性: {np.corrcoef(knn_obs_counts, entropy_info_values)[0,1]:.4f}")
    print(f"平滑信息与k近邻可观测数相关性: {np.corrcoef(knn_obs_counts, smooth_info_values)[0,1]:.4f}")
    print(f"跨视图信息与可观测视图数相关性: {np.corrcoef(cross_view_obs_counts, cross_view_info_values)[0,1]:.4f}")
