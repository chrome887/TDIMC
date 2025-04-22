import numpy as np
# 解决中文显示问题
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
import matplotlib.pyplot as plt
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import torch


def visualize_uncertainty_distribution(uncertainties, preds, true_labels, epoch=None):
    """可视化不确定性分布与聚类性能的关系"""
    plt.figure(figsize=(15, 10))
    
    # 确保输入数据是一维数组
    uncertainties = np.ravel(uncertainties)
    preds = np.ravel(preds)
    true_labels = np.ravel(true_labels)
    
    # 1. 不确定性直方图
    plt.subplot(2, 2, 1)
    plt.hist(uncertainties, bins=30, alpha=0.7)
    plt.xlabel('不确定性')
    plt.ylabel('样本数量')
    plt.title('不确定性分布直方图')
    
    # 2. 不确定性与聚类正确性关系
    correct = (preds == true_labels)
    plt.subplot(2, 2, 2)
    plt.scatter(uncertainties[correct], np.ones_like(uncertainties[correct]), 
                c='g', alpha=0.5, label='聚类正确')
    plt.scatter(uncertainties[~correct], np.zeros_like(uncertainties[~correct]), 
                c='r', alpha=0.5, label='聚类错误')
    plt.yticks([0, 1], ['错误', '正确'])
    plt.xlabel('不确定性')
    plt.legend()
    plt.title('不确定性与聚类正确性关系')
    
    # 3. 按不确定性排序的性能
    # 按不确定性分组计算性能
    sorted_indices = np.argsort(uncertainties)
    window_size = len(uncertainties) // 10
    if window_size < 10:
        window_size = 10
    
    x_windows = []
    nmi_values = []
    ari_values = []
    
    for i in range(0, len(uncertainties), window_size):
        if i + window_size > len(uncertainties):
            break
            
        window_indices = sorted_indices[i:i+window_size]
        window_true = true_labels[window_indices].ravel()  # 确保是一维数组
        window_preds = preds[window_indices].ravel()       # 确保是一维数组
        window_uncertainty = np.mean(uncertainties[window_indices])
        
        x_windows.append(window_uncertainty)
        nmi_values.append(normalized_mutual_info_score(window_true, window_preds))
        ari_values.append(adjusted_rand_score(window_true, window_preds))
    
    plt.subplot(2, 2, 3)
    plt.plot(x_windows, nmi_values, 'o-', label='NMI')
    plt.xlabel('平均不确定性')
    plt.ylabel('NMI')
    plt.title('不确定性与NMI的关系')
    
    plt.subplot(2, 2, 4)
    plt.plot(x_windows, ari_values, 'o-', label='ARI')
    plt.xlabel('平均不确定性')
    plt.ylabel('ARI')
    plt.title('不确定性与ARI的关系')
    
    plt.tight_layout()
    if epoch is not None:
        plt.savefig(f'uncertainty_analysis_epoch{epoch}.png')
    else:
        plt.savefig('uncertainty_analysis_final.png')
    plt.close()

def analyze_view_uncertainties(model, data_loader, epoch=None):
    """分析不同视图的不确定性估计"""
    device = next(model.parameters()).device
    model.eval()
    
    all_view_uncertainties = [[] for _ in range(model.n_views)]
    fusion_uncertainties = []
    
    with torch.no_grad():
        for batch_x, _ in data_loader:
            views = [view.float().to(device) for view in batch_x]
            outputs, fusion_output = model.forward(views)
            
            # 收集每个视图的不确定性
            for i in range(model.n_views):
                all_view_uncertainties[i].extend(outputs[i]['uncertainty'].cpu().numpy().flatten())
            
            # 收集融合后的不确定性
            fusion_uncertainties.extend(fusion_output['uncertainty'].cpu().numpy().flatten())
    
    # 转换为numpy数组 - 确保是一维数据
    all_view_uncertainties = [np.array(x) for x in all_view_uncertainties]
    fusion_uncertainties = np.array(fusion_uncertainties)
    
    # 可视化
    plt.figure(figsize=(12, 6))
    
    # 1. 各视图不确定性箱线图
    plt.subplot(1, 2, 1)
    box_data = all_view_uncertainties + [fusion_uncertainties]  # 合并为一个列表
    labels = [f'视图 {i+1}' for i in range(model.n_views)] + ['融合结果']
    plt.boxplot(box_data, labels=labels)
    plt.title('各视图不确定性分布')
    plt.ylabel('不确定性')
    
    # 2. 视图不确定性相关性热图
    plt.subplot(1, 2, 2)
    
    # 计算相关性矩阵 - 确保每个数组长度相同
    min_length = min(len(arr) for arr in all_view_uncertainties + [fusion_uncertainties])
    correlation_data = np.vstack([arr[:min_length] for arr in all_view_uncertainties + [fusion_uncertainties]])
    corr_matrix = np.corrcoef(correlation_data)
    
    plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(label='相关系数')
    tick_labels = [f'视图 {i+1}' for i in range(model.n_views)] + ['融合']
    plt.xticks(np.arange(len(tick_labels)), tick_labels, rotation=45)
    plt.yticks(np.arange(len(tick_labels)), tick_labels)
    plt.title('不确定性相关性热图')
    
    plt.tight_layout()
    if epoch is not None:
        plt.savefig(f'view_uncertainty_analysis_epoch{epoch}.png')
    else:
        plt.savefig('view_uncertainty_analysis_final.png')
    plt.close()
    
    return all_view_uncertainties, fusion_uncertainties

def debug_DS_Combin(alpha_list, view_idx=None, sample_idx=None):
    """调试DS组合规则融合过程
    
    Args:
        alpha_list: 视图alpha参数列表
        view_idx: 要分析的视图索引 (如果为None则分析所有)
        sample_idx: 要分析的样本索引 (如果为None则取批次第一个样本)
    """
    def print_dirichlet_stats(alpha, name):
        """打印Dirichlet分布的统计信息"""
        if sample_idx is not None:
            alpha_sample = alpha[sample_idx].detach().cpu().numpy()
        else:
            alpha_sample = alpha[0].detach().cpu().numpy()  # 第一个样本
        
        S = np.sum(alpha_sample)
        uncertainty = alpha_sample.shape[0] / S
        probs = alpha_sample / S
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        
        print(f"{name}:")
        print(f"  Alpha参数: {alpha_sample}")
        print(f"  总和S: {S:.4f}")
        print(f"  不确定性: {uncertainty:.4f}")
        print(f"  期望概率: {probs}")
        print(f"  熵: {entropy:.4f}")
        print(f"  最大概率: {np.max(probs):.4f}, 索引: {np.argmax(probs)}")
        print()
        
        return {
            'alpha': alpha_sample,
            'S': S,
            'uncertainty': uncertainty,
            'probs': probs,
            'entropy': entropy,
            'max_prob': np.max(probs),
            'max_idx': np.argmax(probs)
        }
    
    results = []
    
    # 分析各视图
    if view_idx is None:
        for i, alpha in enumerate(alpha_list):
            stats = print_dirichlet_stats(alpha, f"视图 {i+1}")
            results.append(stats)
    else:
        stats = print_dirichlet_stats(alpha_list[view_idx], f"视图 {view_idx+1}")
        results.append(stats)
    
    return results

# 修改evaluate_edl_clustering函数，添加更多分析
def evaluate_edl_clustering(model, data_loader, true_labels=None, epoch=None, visualize=True):
    device = next(model.parameters()).device
    model.eval()
    all_preds = []
    all_uncertainties = []
    all_view_outputs = [[] for _ in range(model.n_views)]  # 收集每个视图的输出
    all_fusion_outputs = []
    
    with torch.no_grad():
        for batch_x, _ in data_loader:
            views = [view.float().to(device) for view in batch_x]
            outputs, fusion_output = model.forward(views)
            
            # 收集各视图输出
            for i in range(model.n_views):
                all_view_outputs[i].append({
                    'uncertainty': outputs[i]['uncertainty'].cpu(),
                    'probs': outputs[i]['cluster_probs'].cpu(),
                    'alpha': outputs[i]['alpha'].cpu()
                })
            
            # 收集融合结果
            all_fusion_outputs.append({
                'uncertainty': fusion_output['uncertainty'].cpu(),
                'probs': fusion_output['cluster_probs'].cpu(),
                'alpha': fusion_output['alpha'].cpu()
            })
            
            # 使用融合结果进行预测
            combined_probs = fusion_output['cluster_probs']
            uncertainty = fusion_output['uncertainty']
            
            preds = torch.argmax(combined_probs, dim=1)
            
            all_preds.append(preds.cpu())
            all_uncertainties.append(uncertainty.cpu())
    
    all_preds = torch.cat(all_preds).numpy()
    all_uncertainties = torch.cat(all_uncertainties).numpy()
    
    # 打印基本结果
    if true_labels is not None:
        nmi = normalized_mutual_info_score(true_labels, all_preds)
        ari = adjusted_rand_score(true_labels, all_preds)
        print(f"NMI: {nmi:.4f}, ARI: {ari:.4f}, Avg Uncertainty: {np.mean(all_uncertainties):.4f}")
        
        # 分析不同不确定性水平的性能
        low_uncert_threshold = np.median(all_uncertainties)
        low_uncert_idx = np.where(all_uncertainties < low_uncert_threshold)[0]
        high_uncert_idx = np.where(all_uncertainties >= low_uncert_threshold)[0]
        
        if len(low_uncert_idx) > 0:
            low_uncert_preds = all_preds[low_uncert_idx]
            low_uncert_true = true_labels[low_uncert_idx]
            
            low_nmi = normalized_mutual_info_score(low_uncert_true, low_uncert_preds)
            low_ari = adjusted_rand_score(low_uncert_true, low_uncert_preds)
            
            print(f"低不确定性样本 (uncertainty < {low_uncert_threshold:.4f}):")
            print(f"  数量: {len(low_uncert_idx)}, NMI: {low_nmi:.4f}, ARI: {low_ari:.4f}")
            
        if len(high_uncert_idx) > 0:
            high_uncert_preds = all_preds[high_uncert_idx]
            high_uncert_true = true_labels[high_uncert_idx]
            
            high_nmi = normalized_mutual_info_score(high_uncert_true, high_uncert_preds)
            high_ari = adjusted_rand_score(high_uncert_true, high_uncert_preds)
            
            print(f"高不确定性样本 (uncertainty >= {low_uncert_threshold:.4f}):")
            print(f"  数量: {len(high_uncert_idx)}, NMI: {high_nmi:.4f}, ARI: {high_ari:.4f}")
            
        # 计算每个真实类别的平均不确定性
        class_uncertainties = {}
        for cls in np.unique(true_labels):
            cls_idx = (true_labels == cls)
            class_uncertainties[cls] = np.mean(all_uncertainties[cls_idx])
        
        print("\n各类别平均不确定性:")
        for cls, uncert in sorted(class_uncertainties.items(), key=lambda x: x[1]):
            print(f"  类别 {cls}: {uncert:.4f}")
        
        # 添加可视化
        if visualize:
            visualize_uncertainty_distribution(all_uncertainties, all_preds, true_labels, epoch)
            analyze_view_uncertainties(model, data_loader, epoch)
    
    return all_preds, all_uncertainties

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.colors import ListedColormap
import torch

def visualize_clustering_results(model, data_loader, true_labels=None, epoch=None):
    """可视化聚类结果与不确定性分布"""
    device = next(model.parameters()).device
    model.eval()
    
    # 收集数据、预测和不确定性
    all_features = [[] for _ in range(model.n_views)]
    all_preds = []
    all_uncertainties = []
    all_view_preds = [[] for _ in range(model.n_views)]
    
    with torch.no_grad():
        for batch_x, _ in data_loader:
            views = [view.float().to(device) for view in batch_x]
            outputs, fusion_output = model.forward(views)
            
            # 收集各视图特征
            for i in range(model.n_views):
                all_features[i].append(outputs[i]['features'].cpu().numpy())
                all_view_preds[i].append(torch.argmax(outputs[i]['cluster_probs'], dim=1).cpu().numpy())
            
            # 收集融合结果
            preds = torch.argmax(fusion_output['cluster_probs'], dim=1)
            uncertainties = fusion_output['uncertainty']
            
            all_preds.append(preds.cpu().numpy())
            all_uncertainties.append(uncertainties.cpu().numpy())
    
    # 转换为numpy数组
    all_features = [np.vstack(features) for features in all_features]
    all_preds = np.concatenate(all_preds)
    all_uncertainties = np.concatenate(all_uncertainties)
    all_view_preds = [np.concatenate(preds) for preds in all_view_preds]
    
    # 创建多面板可视化
    fig = plt.figure(figsize=(20, 15))
    
    # 1. 融合结果的t-SNE可视化
    # 选择第一个视图特征进行降维(或可以使用所有视图特征的平均)
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(all_features[0])
    
    # 1.1 按预测类别着色
    ax1 = fig.add_subplot(231)
    scatter = ax1.scatter(features_2d[:, 0], features_2d[:, 1], c=all_preds, 
               cmap='tab10', alpha=0.7, s=30)
    ax1.set_title('聚类结果')
    ax1.set_xlabel('t-SNE维度1')
    ax1.set_ylabel('t-SNE维度2')
    plt.colorbar(scatter, ax=ax1)
    
    # 1.2 按不确定性着色
    ax2 = fig.add_subplot(232)
    scatter = ax2.scatter(features_2d[:, 0], features_2d[:, 1], c=all_uncertainties,
               cmap='viridis', alpha=0.7, s=30)
    ax2.set_title('不确定性分布')
    ax2.set_xlabel('t-SNE维度1')
    ax2.set_ylabel('t-SNE维度2')
    plt.colorbar(scatter, ax=ax2)
    
    # 1.3 如果有真实标签，按真实标签着色
    if true_labels is not None:
        ax3 = fig.add_subplot(233)
        scatter = ax3.scatter(features_2d[:, 0], features_2d[:, 1], c=true_labels,
                   cmap='tab10', alpha=0.7, s=30)
        ax3.set_title('真实类别')
        ax3.set_xlabel('t-SNE维度1')
        ax3.set_ylabel('t-SNE维度2')
        plt.colorbar(scatter, ax=ax3)
    
    # 2. 各视图聚类结果对比
    # 使用混淆矩阵来可视化各视图间的一致性
    from sklearn.metrics import confusion_matrix
    
    # 2.1 视图1和融合结果的一致性
    ax4 = fig.add_subplot(234)
    cm = confusion_matrix(all_view_preds[0], all_preds, normalize='true')
    im = ax4.imshow(cm, interpolation='nearest', cmap='Blues')
    ax4.set_title('视图1与融合结果一致性')
    ax4.set_xlabel('融合聚类')
    ax4.set_ylabel('视图1聚类')
    plt.colorbar(im, ax=ax4)
    
    # 2.2 如果有多个视图，视图2和融合结果的一致性
    if model.n_views > 1:
        ax5 = fig.add_subplot(235)
        cm = confusion_matrix(all_view_preds[1], all_preds, normalize='true')
        im = ax5.imshow(cm, interpolation='nearest', cmap='Blues')
        ax5.set_title('视图2与融合结果一致性')
        ax5.set_xlabel('融合聚类')
        ax5.set_ylabel('视图2聚类')
        plt.colorbar(im, ax=ax5)
    
    # 3. 不确定性与聚类边界关系
    ax6 = fig.add_subplot(236)
    
    # 计算每个类别的平均不确定性
    cluster_uncertainties = {}
    for cluster in np.unique(all_preds):
        cluster_mask = (all_preds == cluster)
        cluster_uncertainties[cluster] = np.mean(all_uncertainties[cluster_mask])
    
    # 绘制条形图
    clusters = list(cluster_uncertainties.keys())
    uncertainties = [cluster_uncertainties[c] for c in clusters]
    ax6.bar(clusters, uncertainties, alpha=0.7)
    ax6.set_title('各聚类的平均不确定性')
    ax6.set_xlabel('聚类ID')
    ax6.set_ylabel('平均不确定性')
    
    plt.tight_layout()
    if epoch is not None:
        plt.savefig(f'clustering_visualization_epoch{epoch}.png')
    else:
        plt.savefig('clustering_visualization_final.png')
    plt.close()
    
    return features_2d, all_preds, all_uncertainties