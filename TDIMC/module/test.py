# 测试文件: test_knn_imputer.py
import numpy as np
import torch
from knn_imputer import LatentKNNImputer
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

def test_latent_knn_imputer():
    """测试LatentKNNImputer在多视图数据上的表现"""
    
    # 1. 生成模拟数据
    np.random.seed(42)
    n_samples = 100
    
    # 生成两个视图的数据
    # 视图1: 2D数据，包含两个聚类
    view1 = np.concatenate([
        np.random.randn(n_samples//2, 2) + np.array([3, 3]),  # 第一个聚类
        np.random.randn(n_samples//2, 2) + np.array([-3, -3]) # 第二个聚类
    ])
    
    # 视图2: 2D数据，也包含两个聚类 (稍微旋转一下)
    view2 = np.concatenate([
        np.random.randn(n_samples//2, 2) + np.array([3, -3]),  # 第一个聚类
        np.random.randn(n_samples//2, 2) + np.array([-3, 3])   # 第二个聚类
    ])
    
    # 原始完整数据
    original_data = [view1, view2]
    
    # 2. 生成缺失掩码 (视图级别缺失)
    miss_rate = 0.3  # 30%的视图级缺失率
    
    # 每个视图的缺失掩码 (1表示可用，0表示缺失)
    mask_list = []
    for v in range(len(original_data)):
        mask = np.ones(n_samples)
        # 随机选择30%的样本标记为缺失
        miss_indices = np.random.choice(n_samples, size=int(n_samples * miss_rate), replace=False)
        mask[miss_indices] = 0
        mask_list.append(mask)
    
    # 3. 创建带缺失的数据
    data_list = []
    for v, data in enumerate(original_data):
        # 复制原始数据
        data_with_missing = data.copy()
        # 记录缺失样本的原始值，用于验证
        missing_ground_truth = []
        for i in range(n_samples):
            if mask_list[v][i] == 0:
                missing_ground_truth.append((i, data_with_missing[i].copy()))
        data_list.append(data_with_missing)
    
    # 4. 模拟潜在表示
    # 假设我们有一个模型将原始数据映射到潜在空间
    # 这里简单起见，我们使用PCA模拟潜在表示
    latent_dim = 3  # 潜在空间维度
    latent_reps = []
    
    for v, data in enumerate(data_list):
        # 简化模拟：将原始数据添加随机投影作为潜在表示
        projection = np.random.randn(data.shape[1], latent_dim)
        latent = np.dot(data, projection)  # 简单线性投影
        latent_reps.append(torch.tensor(latent, dtype=torch.float32))
    
    # 5. 初始化KNN插补器
    imputer = LatentKNNImputer(k=5, alpha=0.6, interval=1, topk=10)
    
    # 6. 拟合模型
    imputer.fit(latent_reps)
    
    # 7. 插补缺失数据
    imputed_data = imputer.impute(data_list, mask_list)
    
    # 8. 评估插补质量
    mse_per_view = []
    for v, (orig, imputed) in enumerate(zip(original_data, imputed_data)):
        errors = []
        for i in range(n_samples):
            if mask_list[v][i] == 0:  # 只评估缺失值
                if isinstance(imputed, torch.Tensor):
                    imputed_val = imputed[i].detach().numpy()
                else:
                    imputed_val = imputed[i]
                error = np.mean((orig[i] - imputed_val) ** 2)
                errors.append(error)
        if errors:
            mse = np.mean(errors)
            mse_per_view.append(mse)
            print(f"视图 {v+1} 的平均MSE: {mse:.6f}")
    
    # 9. 更新潜在表示并重新插补
    print("\n更新潜在表示...\n")
    # 模拟更新后的潜在表示 (轻微变化)
    updated_latent_reps = []
    for latent in latent_reps:
        updated = latent + torch.randn_like(latent) * 0.1  # 添加小扰动
        updated_latent_reps.append(updated)
    
    # 更新插补器
    imputer.update_features(updated_latent_reps)
    
    # 再次插补
    updated_imputed_data = imputer.impute(data_list, mask_list)
    
    # 10. 评估更新后的插补质量
    mse_per_view_updated = []
    for v, (orig, imputed) in enumerate(zip(original_data, updated_imputed_data)):
        errors = []
        for i in range(n_samples):
            if mask_list[v][i] == 0:  # 只评估缺失值
                if isinstance(imputed, torch.Tensor):
                    imputed_val = imputed[i].detach().numpy()
                else:
                    imputed_val = imputed[i]
                error = np.mean((orig[i] - imputed_val) ** 2)
                errors.append(error)
        if errors:
            mse = np.mean(errors)
            mse_per_view_updated.append(mse)
            print(f"更新后，视图 {v+1} 的平均MSE: {mse:.6f}")
    
    # 11. 可视化插补效果
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    
    # 视图1
    # 原始数据
    axs[0, 0].scatter(original_data[0][:, 0], original_data[0][:, 1], c='blue', alpha=0.5)
    axs[0, 0].set_title('视图1: 原始数据')
    
    # 缺失数据
    present_mask = mask_list[0] == 1
    missing_mask = mask_list[0] == 0
    axs[0, 1].scatter(original_data[0][present_mask, 0], original_data[0][present_mask, 1], c='blue', alpha=0.5)
    axs[0, 1].set_title('视图1: 缺失后数据')
    
    # 插补后数据
    if isinstance(imputed_data[0], torch.Tensor):
        imputed_np = imputed_data[0].detach().numpy()
    else:
        imputed_np = imputed_data[0]
    axs[0, 2].scatter(original_data[0][present_mask, 0], original_data[0][present_mask, 1], c='blue', alpha=0.5)
    axs[0, 2].scatter(imputed_np[missing_mask, 0], imputed_np[missing_mask, 1], c='red', marker='x')
    axs[0, 2].set_title('视图1: 插补后数据')
    
    # 视图2
    # 原始数据
    axs[1, 0].scatter(original_data[1][:, 0], original_data[1][:, 1], c='green', alpha=0.5)
    axs[1, 0].set_title('视图2: 原始数据')
    
    # 缺失数据
    present_mask = mask_list[1] == 1
    missing_mask = mask_list[1] == 0
    axs[1, 1].scatter(original_data[1][present_mask, 0], original_data[1][present_mask, 1], c='green', alpha=0.5)
    axs[1, 1].set_title('视图2: 缺失后数据')
    
    # 插补后数据
    if isinstance(imputed_data[1], torch.Tensor):
        imputed_np = imputed_data[1].detach().numpy()
    else:
        imputed_np = imputed_data[1]
    axs[1, 2].scatter(original_data[1][present_mask, 0], original_data[1][present_mask, 1], c='green', alpha=0.5)
    axs[1, 2].scatter(imputed_np[missing_mask, 0], imputed_np[missing_mask, 1], c='red', marker='x')
    axs[1, 2].set_title('视图2: 插补后数据')
    
    plt.tight_layout()
    plt.savefig('imputation_results.png')
    print("可视化结果已保存为 'imputation_results.png'")
    
    # 返回评估结果
    return {
        "initial_mse": mse_per_view,
        "updated_mse": mse_per_view_updated
    }

if __name__ == "__main__":
    results = test_latent_knn_imputer()
    
    # 比较更新前后的插补结果
    initial_avg_mse = np.mean(results["initial_mse"])
    updated_avg_mse = np.mean(results["updated_mse"])
    
    print(f"\n初始平均MSE: {initial_avg_mse:.6f}")
    print(f"更新后平均MSE: {updated_avg_mse:.6f}")
    change = (updated_avg_mse - initial_avg_mse) / initial_avg_mse * 100
    print(f"变化: {change:.2f}%")