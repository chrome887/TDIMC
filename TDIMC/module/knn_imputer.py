# knn_imputer.py
import numpy as np
import torch
import faiss

class LatentKNNImputer:
    def __init__(self, k=10, alpha=0.5, interval=1, metric="cosine", topk=None):
        """
        基于潜在空间的KNN插补器 - 专为视图级别缺失设计
        
        参数:
        k: 用于插补的邻居数量
        alpha: 更新特征的权重参数
        interval: 更新间隔
        metric: 相似度度量方式 (现在只支持cosine)
        topk: 搜索的最近邻数量 (默认与k相同)
        """
        self.k = k
        self.topk = topk if topk is not None else k
        self.alpha = alpha
        self.interval = interval
        self.metric = metric
        self.features = None
        self.neighbor_indices = None
        self.iteration = 0

    def fit(self, latent_reps, masks=None):
        """
        基于潜在表示拟合模型
        
        参数:
        latent_reps: 视图潜在表示的列表 [V1, V2, ..., Vn]
        masks: 每个视图的缺失掩码列表 (可选)
        
        返回:
        neighbor_indices: 每个视图中每个样本的最近邻索引
        """
        self.view_num = len(latent_reps)
        self.features = []
        self.neighbor_indices = []
        
        # 计算每个视图的邻居
        for v, z in enumerate(latent_reps):
            # 转换为numpy数组
            z = z.detach().cpu().numpy()
            
            # 归一化特征向量
            z = z / np.linalg.norm(z, axis=1, keepdims=True)
            
            # 初始化或更新特征存储
            if self.features is None or len(self.features) <= v:
                self.features.append(z)
            else:
                # 按照alpha权重更新特征
                self.features[v] = (1 - self.alpha) * self.features[v] + self.alpha * z
                # 重新归一化
                self.features[v] = self.features[v] / np.linalg.norm(self.features[v], axis=1, keepdims=True)
            
            # 使用faiss加速KNN搜索
            n, dim = self.features[v].shape
            index = faiss.IndexFlatIP(dim)  # 内积索引 (归一化后等价于余弦相似度)
            index.add(self.features[v])
            
            # 查找topk+1个最近邻 (包括样本自身)
            _, indices = index.search(self.features[v], self.topk + 1)
            
            # 排除自身，只保留topk个邻居
            self.neighbor_indices.append(indices[:, 1:])
        
        self.iteration += 1
        return self.neighbor_indices
    
    def update_features(self, new_latent_reps):
        """
        根据间隔更新潜在特征
        
        参数:
        new_latent_reps: 新的潜在表示列表
        
        返回:
        neighbor_indices: 更新后的邻居索引
        """
        self.iteration += 1
        if self.iteration % self.interval != 0:
            return self.neighbor_indices
        
        return self.fit(new_latent_reps)
    
    def impute(self, data_list, mask_list):
        """
        处理视图级别缺失 - 使用其他可用视图中的相似样本进行插补
        
        参数:
        data_list: 每个视图的原始数据列表
        mask_list: 视图级别的缺失掩码 (0表示整个视图缺失，1表示可用)
        
        返回:
        imputed_data: 插补后的数据列表
        """
        if self.neighbor_indices is None:
            raise ValueError("模型尚未拟合，请先调用fit方法")
        
        imputed_data = []
        data_size = data_list[0].shape[0]
        view_num = len(data_list)
        
        # 对每个视图进行处理
        for v in range(view_num):
            data = data_list[v].clone() if isinstance(data_list[v], torch.Tensor) else torch.tensor(data_list[v]).clone()
            
            # 对每个样本进行处理
            for i in range(data_size):
                # 如果该样本在当前视图中不可用(视图级缺失)
                if mask_list[v][i] == 0:
                    predictions = []
                    
                    # 找出哪些视图对该样本可用
                    available_views = [w for w in range(view_num) if w != v and mask_list[w][i] == 1]
                    
                    # 如果没有任何可用视图，则跳过
                    if not available_views:
                        continue
                    
                    # 从可用视图中获取邻居
                    for w in available_views:
                        # 获取样本i在视图w中的邻居
                        neighbors = self.neighbor_indices[w][i]
                        
                        # 查找这些邻居在视图v中的数据(如果可用)
                        for n_idx in neighbors:
                            if mask_list[v][n_idx] == 1:  # 如果邻居在视图v中可用
                                predictions.append(data_list[v][n_idx])
                            
                            if len(predictions) >= self.k:
                                break
                        
                        if len(predictions) >= self.k:
                            break
                    
                    # 如果收集到预测，进行插补
                    if len(predictions) > 0:
                        # 计算平均值
                        if isinstance(predictions[0], torch.Tensor):
                            fill_sample = torch.stack(predictions).mean(dim=0)
                        else:
                            fill_sample = torch.tensor(np.mean(predictions, axis=0))
                        
                        # 插补整个视图的数据
                        data[i] = fill_sample
            
            imputed_data.append(data)
        
        return imputed_data
    

def cosine_similarity(X, Y=None):
    """
    计算余弦相似度
    
    参数:
    X: 形状为 (n_samples_X, n_features) 的矩阵
    Y: 形状为 (n_samples_Y, n_features) 的矩阵 (默认为None，则Y=X)
    
    返回:
    S: 形状为 (n_samples_X, n_samples_Y) 的相似度矩阵
    """
    if Y is None:
        Y = X
    
    # 归一化
    X_normalized = X / np.linalg.norm(X, axis=1, keepdims=True)
    Y_normalized = Y / np.linalg.norm(Y, axis=1, keepdims=True)
    
    # 计算余弦相似度
    return np.dot(X_normalized, Y_normalized.T)

