import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, normalized_mutual_info_score, adjusted_rand_score, f1_score
import numpy as np

class CrossViewAutoencoder(nn.Module):
    def __init__(self, input_dims, latent_dim):
        super(CrossViewAutoencoder, self).__init__()
        self.encoders = nn.ModuleList([nn.Sequential(
            nn.Linear(input_dim[0], 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        ) for input_dim in input_dims])
        
        self.decoders = nn.ModuleList([nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        ) for input_dim in input_dims])

    def forward(self, views):
        latent_representations = [encoder(view) for encoder, view in zip(self.encoders, views)]
        reconstructed_views = [decoder(latent) for decoder, latent in zip(self.decoders, latent_representations)]
        return latent_representations, reconstructed_views

    def cross_view_mapping(self, observed_views, missing_view_index):
        # Encode observed views
        latent_representations = [self.encoders[i](view) for i, view in enumerate(observed_views)]
        # Average latent representations (simple fusion strategy)
        fused_latent = torch.mean(torch.stack(latent_representations), dim=0)
        # Decode to predict the missing view
        predicted_view = self.decoders[missing_view_index](fused_latent)
        return predicted_view

def train_model(model, data_loader, num_views, num_epochs=50, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        for batch in data_loader:
            views = [batch[f'view_{i}'] for i in range(num_views)]
            views = [view.to(torch.float32) for view in views]

            # Forward pass
            _, reconstructed_views = model(views)
            loss = sum(criterion(reconstructed, view) for reconstructed, view in zip(reconstructed_views, views))

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

def evaluate_clustering(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    return acc, nmi, ari, f1

def test_model(model, data_loader, mask, num_views, device):
    model.eval()
    imputed_features = []
    zero_filled_features = []
    y_true = []

    with torch.no_grad():
        for batch in data_loader:
            views = [batch[f'view_{i}'].to(device) for i in range(num_views)]
            y_true.extend(batch['labels'].cpu().numpy())
            mask_batch = mask.to(device)

            # Impute missing views
            imputed_batch = []
            for i in range(num_views):
                observed_views = [views[j] for j in range(num_views) if j != i]
                imputed_view = model.cross_view_mapping(observed_views, i)
                imputed_batch.append(imputed_view)
            imputed_features.append(torch.cat(imputed_batch, dim=1).cpu().numpy())

            # Zero-fill missing views
            zero_filled_batch = []
            for i in range(num_views):
                zero_filled_view = views[i].clone()
                zero_filled_view[~mask_batch[:, i]] = 0
                zero_filled_batch.append(zero_filled_view)
            zero_filled_features.append(torch.cat(zero_filled_batch, dim=1).cpu().numpy())

    # Concatenate all batches
    imputed_features = np.concatenate(imputed_features, axis=0)
    zero_filled_features = np.concatenate(zero_filled_features, axis=0)
    y_true = np.array(y_true)

    # Perform clustering
    kmeans_imputed = KMeans(n_clusters=len(np.unique(y_true)), random_state=0).fit(imputed_features)
    kmeans_zero = KMeans(n_clusters=len(np.unique(y_true)), random_state=0).fit(zero_filled_features)

    # Evaluate clustering
    y_pred_imputed = kmeans_imputed.labels_
    y_pred_zero = kmeans_zero.labels_

    acc_imputed, nmi_imputed, ari_imputed, f1_imputed = evaluate_clustering(y_true, y_pred_imputed)
    acc_zero, nmi_zero, ari_zero, f1_zero = evaluate_clustering(y_true, y_pred_zero)

    print("Imputed Features:")
    print(f"ACC: {acc_imputed:.4f}, NMI: {nmi_imputed:.4f}, ARI: {ari_imputed:.4f}, F1: {f1_imputed:.4f}")
    print("Zero-Filled Features:")
    print(f"ACC: {acc_zero:.4f}, NMI: {nmi_zero:.4f}, ARI: {ari_zero:.4f}, F1: {f1_zero:.4f}")

if __name__ == "__main__":
    import os
    import numpy as np
    import random
    import statistics
    from sklearn.cluster import KMeans
    from utils import get_default_config, get_logger, DatasetLoader
    from missing_simulator import load_sn

    # 配置参数
    dataset = {0: "Handwritten", 1: "Scene15"}
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=int, default=1, help='dataset id')
    parser.add_argument('--devices', type=str, default='0', help='gpu device ids')
    parser.add_argument('--print_num', type=int, default=50, help='gap of print evaluations')
    parser.add_argument('--test_time', type=int, default=5, help='number of test times')
    parser.add_argument('--missing_rate', type=float, default=0.5, help='missing rate')
    parser.add_argument('--missing_mode', type=str, default='base_view', help='missing mode')

    args = parser.parse_args()
    dataset = dataset[args.dataset]

    # 加载配置
    config = get_default_config(dataset)
    config['missing_rate'] = args.missing_rate
    config['print_num'] = args.print_num
    config['dataset'] = dataset
    config['missing_mode'] = args.missing_mode
    config['sn_path'] = os.path.join("sn", f"{dataset}_{args.missing_mode}_{args.missing_rate:.1f}".replace(".", "") + ".csv")
    logger, plt_name = get_logger(config)

    logger.info('Crossview Autoencoder Test Log')
    logger.info('Dataset:' + str(dataset))
    for k, v in config.items():
        logger.info(f"{k} = {v}")

    # 加载数据集
    loader = DatasetLoader(config)
    X_list, Y_list = loader.load_dataset(config["dataset"])
    num_views = config["num_views"]
    dims = config["dims"]

    # 设置随机种子
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # 设置设备
    os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化模型
    model = CrossViewAutoencoder(input_dims=dims, latent_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['lr'])

    # 测试多次
    accumulated = {
        'acc': [], 'nmi': [], 'ARI': [], 'f-mea': [],
        'zero_acc': [], 'zero_nmi': [], 'zero_ARI': [], 'zero_f-mea': []
    }

    for i in range(args.test_time):
        logger.info(f"Test time: {i+1}/{args.test_time}")

        # 模拟缺失数据
        mask = load_sn(config['sn_path'])
        mask_tensor = torch.FloatTensor(mask).bool().to(device)
        X_tensor_list = [torch.FloatTensor(X).to(device) for X in X_list]
        Y_tensor = torch.LongTensor(Y_list).to(device)

        # 训练模型
        logger.info("Training CrossViewAutoencoder...")
        train_model(model, data_loader=None, num_views=num_views, num_epochs=config['training']['pretrain_epochs'], lr=config['training']['lr'])

        # 测试模型
        logger.info("Testing with cross-view imputed features...")
        model.eval()
        with torch.no_grad():
            # 提取补全特征
            imputed_features = []
            zero_filled_features = []
            for v in range(num_views):
                observed_views = [X_tensor_list[j] for j in range(num_views) if j != v]
                imputed_view = model.cross_view_mapping(observed_views, v)
                imputed_features.append(imputed_view.cpu().numpy())

                zero_filled_view = X_tensor_list[v].clone()
                zero_filled_view[~mask_tensor[:, v]] = 0
                zero_filled_features.append(zero_filled_view.cpu().numpy())

            # 聚类
            concat_imputed = np.concatenate(imputed_features, axis=1)
            concat_zero = np.concatenate(zero_filled_features, axis=1)
            y_true = Y_list

            kmeans_imputed = KMeans(n_clusters=config['classes'], random_state=0).fit(concat_imputed)
            kmeans_zero = KMeans(n_clusters=config['classes'], random_state=0).fit(concat_zero)

            # 评估聚类性能
            from clustering import clustering_metric
            scores_imputed, _ = clustering_metric(y_true, kmeans_imputed.labels_, config['classes'])
            scores_zero, _ = clustering_metric(y_true, kmeans_zero.labels_, config['classes'])

            # 记录结果
            accumulated['acc'].append(scores_imputed['accuracy'])
            accumulated['nmi'].append(scores_imputed['NMI'])
            accumulated['ARI'].append(scores_imputed['ARI'])
            accumulated['f-mea'].append(scores_imputed['f_measure'])
            accumulated['zero_acc'].append(scores_zero['accuracy'])
            accumulated['zero_nmi'].append(scores_zero['NMI'])
            accumulated['zero_ARI'].append(scores_zero['ARI'])
            accumulated['zero_f-mea'].append(scores_zero['f_measure'])

    # 输出最终结果
    logger.info("\nFinal Results (Mean ± Std):")
    logger.info("Cross-view imputation:")
    logger.info(f"ACC: {statistics.mean(accumulated['acc']):.4f} ± {statistics.stdev(accumulated['acc']):.4f}")
    logger.info(f"NMI: {statistics.mean(accumulated['nmi']):.4f} ± {statistics.stdev(accumulated['nmi']):.4f}")
    logger.info(f"ARI: {statistics.mean(accumulated['ARI']):.4f} ± {statistics.stdev(accumulated['ARI']):.4f}")
    logger.info(f"F-score: {statistics.mean(accumulated['f-mea']):.4f} ± {statistics.stdev(accumulated['f-mea']):.4f}")

    logger.info("\nZero-filling:")
    logger.info(f"ACC: {statistics.mean(accumulated['zero_acc']):.4f} ± {statistics.stdev(accumulated['zero_acc']):.4f}")
    logger.info(f"NMI: {statistics.mean(accumulated['zero_nmi']):.4f} ± {statistics.stdev(accumulated['zero_nmi']):.4f}")
    logger.info(f"ARI: {statistics.mean(accumulated['zero_ARI']):.4f} ± {statistics.stdev(accumulated['zero_ARI']):.4f}")
    logger.info(f"F-score: {statistics.mean(accumulated['zero_f-mea']):.4f} ± {statistics.stdev(accumulated['zero_f-mea']):.4f}")
