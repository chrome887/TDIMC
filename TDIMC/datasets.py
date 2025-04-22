# datasets.py
import os
import numpy as np
import scipy.io as scio
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Dict
import torch
from utils import DatasetLoader


# 示例用法
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

    args = parser.parse_args()

    dataset = dataset[args.dataset]
    # configure
    from utils import get_default_config, get_logger
    config = get_default_config(dataset)
    config['missing_rate'] = args.missing_rate
    config['print_num'] = args.print_num
    config['dataset'] = dataset
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
    
    # 加载Handwritten
    X_list, Y_list = loader.load_dataset(config["dataset"])
    num_views = config["num_views"]
    dims = config["dims"]
    
    print("Number of views:", num_views)
    print("Dimensions of each view:", dims)
    print("Number of samples:", X_list[0].shape[0])