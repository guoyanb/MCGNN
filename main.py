# -*- coding: utf-8 -*-
import warnings
from data_process import data_lode
from train import Train
from utils import *
import os
import torch

warnings.filterwarnings("ignore")


def main_indep(args):
    # 设置设备和随机种子
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() and args.use_gpu else 'cpu')
    print(f"Using device: {device}")
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
    
    features, in_size = data_lode()
    
    # 将特征移到GPU
    for key in features:
        features[key] = features[key].to(device)
    
    NDCG_5, NDCG_3, NDCG_1, MRR = [list() for x in range(4)]
    
    train_data_pos = np.array(
        pd.read_csv('./MCGNN/Data/indepent_data/train_data_pos.csv', header=None))
    train_data_neg = np.array(
        pd.read_csv('./MCGNN/Data/indepent_data/train_data_neg.csv',header=None))
    val_data_pos = np.array(
        pd.read_csv('./MCGNN/Data/indepent_data/test_data_pos.csv', header=None))
    val_data_neg = np.array(
        pd.read_csv('./MCGNN/Data/indepent_data/test_data_neg.csv', header=None))
    
    print("="*80)
    print("Enhanced MCGNN with Wavelet Features in First Layer")
    print("="*80)
    print(f"Device: {device}")
    print(f"Wavelet type: {args.wavelet_type}")
    print(f"Number of layers: {args.num_layers}")
    print("="*80)
    
    hg = construct_hg(train_data_pos)
    hg = hg.to(device)
    
    train_data = np.vstack((train_data_pos, train_data_neg))
    np.random.shuffle(train_data)
    val_data = np.vstack((val_data_pos, val_data_neg))
    
    result = Train(train_data, val_data, in_size, args, hg, features, device)
    
    NDCG_5.append(result[0])
    NDCG_3.append(result[1])
    NDCG_1.append(result[2])
    MRR.append(result[3])
    
    print('----------independent test finished-----------')
    print("="*80)
    print("Enhanced MCGNN Results Summary")
    print("="*80)
    print('Independent test result：')
    print('  NDCG@5: %.6f' % np.mean(NDCG_5))
    print('  NDCG@3: %.6f' % np.mean(NDCG_3))
    print('  NDCG@1: %.6f' % np.mean(NDCG_1))
    print('  MRR:    %.6f' % np.mean(MRR))
    print("="*80)
    
    return np.mean(NDCG_5), np.mean(NDCG_3), np.mean(NDCG_1), np.mean(MRR)


def main_CV(args):
    # 设置设备和随机种子
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() and args.use_gpu else 'cpu')
    print(f"Using device: {device}")
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
    
    features, in_size = data_lode()
    
    # 将特征移到GPU
    for key in features:
        features[key] = features[key].to(device)
    
    NDCG_5, NDCG_3, NDCG_1, MRR = [list() for x in range(4)]
    fold_num = 0
    
    for i in range(5):
        fold_num += 1
        train_data_pos = np.array(
            pd.read_csv('./MCGNN/Data/CV_data/CV_' + str(fold_num) + '/train_data_pos.csv', header=None))
        train_data_neg = np.array(
            pd.read_csv('./MCGNN/Data/CV_data/CV_' + str(fold_num) + '/train_data_neg.csv', header=None))
        val_data_pos = np.array(
            pd.read_csv('./MCGNN/Data/CV_data/CV_' + str(fold_num) + '/val_data_pos.csv', header=None))
        val_data_neg = np.array(
            pd.read_csv('./MCGNN/Data/CV_data/CV_' + str(fold_num) + '/val_data_neg.csv',header=None))

        print(f"\nProcessing Fold {fold_num}")
        print(f"Using Enhanced MCGNN with Wavelet Features in First Layer")
        print(f"Device: {device}")
        
        hg = construct_hg(train_data_pos)
        hg = hg.to(device)
        
        train_data = np.vstack((train_data_pos, train_data_neg))
        np.random.shuffle(train_data)
        val_data = np.vstack((val_data_pos, val_data_neg))
        
        result = Train(train_data, val_data, in_size, args, hg, features, device)
        
        NDCG_5.append(result[0])
        NDCG_3.append(result[1])
        NDCG_1.append(result[2])
        MRR.append(result[3])
    
    print('----------5 fold CV finished-----------')
    print("="*80)
    print("MCGNN 5-Fold Cross Validation Results")
    print("="*80)
    print('5-fold CV result：')
    print('  NDCG@5: %.6f ± %.6f' % (np.mean(NDCG_5), np.std(NDCG_5)))
    print('  NDCG@3: %.6f ± %.6f' % (np.mean(NDCG_3), np.std(NDCG_3)))
    print('  NDCG@1: %.6f ± %.6f' % (np.mean(NDCG_1), np.std(NDCG_1)))
    print('  MRR:    %.6f ± %.6f' % (np.mean(MRR), np.std(MRR)))
    print("="*80)
    
    return np.mean(NDCG_5), np.mean(NDCG_3), np.mean(NDCG_1), np.mean(MRR)


if __name__ == '__main__':
    args = parameters_set()
    
    if not os.path.exists('./Result'):
        os.makedirs('./Result', exist_ok=True)
    
    print("="*80)
    print("MCGNN WITH WAVELET FEATURES IN FIRST LAYER")
    print("="*80)
    print("Features:")
    print("  1. Wavelet Transform applied only in the first layer")
    print("  2. Wavelet features fused with original features")
    print("  3. Subsequent layers use Fourier convolution and causal learning")
    print("  4. Simplified architecture for better efficiency")
    print("="*80)
    print(f"GPU enabled: {args.use_gpu}")
    if args.use_gpu:
        print(f"GPU ID: {args.gpu_id}")
        print(f"GPU Wavelet Transform: ENABLED")
    else:
        print(f"GPU Wavelet Transform: DISABLED (using CPU fallback)")
    print(f"Wavelet type: {args.wavelet_type}")
    print(f"Number of layers: {args.num_layers}")
    print("="*80)
    
    print('\nStarting the 5-fold CV experiment')
    CV_NDCG_5, CV_NDCG_3, CV_NDCG_1, CV_MRR_num = main_CV(args)
    
    with open('./Result/Enhanced_MCGNN_CV_Results.txt', 'a') as f:
        f.write("Enhanced MCGNN with Wavelet in First Layer\n")
        f.write(f"NDCG@5: {CV_NDCG_5:.6f}\tNDCG@3: {CV_NDCG_3:.6f}\tNDCG@1: {CV_NDCG_1:.6f}\tMRR: {CV_MRR_num:.6f}\n")
        f.write("-"*80 + "\n")

    print('\nStarting the independent test experiment')
    indep_NDCG_5, indep_NDCG_3, indep_NDCG_1, indep_MRR_num = main_indep(args)
    
    with open('./Result/Enhanced_MCGNN_Indep_Results.txt', 'a') as f:
        f.write("Enhanced MCGNN with Wavelet in First Layer\n")
        f.write(f"NDCG@5: {indep_NDCG_5:.6f}\tNDCG@3: {indep_NDCG_3:.6f}\tNDCG@1: {indep_NDCG_1:.6f}\tMRR: {indep_MRR_num:.6f}\n")
        f.write("-"*80 + "\n")
    
    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("="*80)