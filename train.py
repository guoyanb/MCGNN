# train.py
import os
import numpy as np
import torch
import warnings
from model import MCGNN
from utils import *

warnings.filterwarnings("ignore")


def Train(train_data, test_data, in_size, args, hg, features, device):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(args.seed)
    
    os.makedirs('./Result', exist_ok=True)
    best_model_path = './Result/best_model.pth'
    model = MCGNN(
        meta_paths=args.metapaths,
        test_data=test_data,
        in_size=in_size,
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        dropout=args.dropout,
        etypes=args.etypes,
        device=device,
        wavelet_type=args.wavelet_type,
        num_layers=args.num_layers
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)

    myloss = Myloss()
    
    mrr = MRR()
    matrix = Matrix()
    
    trainloss = []
    valloss = []
    result_list = []
    NDCG_max_matrix = np.zeros((1, 3))
    patience_num_matrix = np.zeros((1, 1))
    MRR_max_matrix = np.zeros((1, 1))
    epoch_max_matrix = np.zeros((1, 1))
    
    # Prepare validation data
    val_data_pos = test_data[np.where(test_data[:, -1] == 1)]
    shuffle_index = np.random.choice(range(len(test_data)), len(test_data), replace=False)
    task_test_data = test_data[shuffle_index]

    for epoch in range(args.num_epochs):
        model.train()
        optimizer.zero_grad()
        
        # Move training data to device
        train_label = torch.unsqueeze(torch.from_numpy(train_data[:, 3]).float(), 1).to(device)
        
        # Forward pass
        main_score = model(hg, features, train_data)
        
        # Compute main loss
        train_loss = myloss(main_score, train_label, args.alpha)

        trainloss.append(train_loss.item())
        train_loss.backward()
        
        # Gradient clipping to prevent gradient explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        model.eval()
        with torch.no_grad():
            # Move validation data to device
            val_label = torch.unsqueeze(torch.from_numpy(task_test_data[:, 3]).float(), 1).to(device)
            
            # Forward pass on validation set
            main_score_val = model(hg, features, task_test_data)
            
            # Compute validation loss
            val_loss = myloss(main_score_val, val_label, args.alpha)

            valloss.append(val_loss.item())
            
            # Move predictions to CPU for computation
            predict_val = np.squeeze(main_score_val.cpu().detach().numpy())
            ndcg5, sample_ndcg5 = matrix(5, 30, predict_val, len(val_data_pos), shuffle_index)
            ndcg3, sample_ndcg3 = matrix(3, 30, predict_val, len(val_data_pos), shuffle_index)
            ndcg1, sample_ndcg1 = matrix(1, 30, predict_val, len(val_data_pos), shuffle_index)
            MRR_num, sample_mrr = mrr(30, predict_val, len(val_data_pos), shuffle_index)
            
            result = [val_loss.item()] + [ndcg5] + [ndcg3] + [ndcg1] + [MRR_num]
            result_list.append(result)
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch: {epoch + 1:4d} | '
                      f'Train Loss: {train_loss.item():.4f} | '
                      f'Val Loss: {val_loss.item():.4f} | '
                      f'Val NDCG@5: {ndcg5:.6f} | '
                      f'Val NDCG@3: {ndcg3:.6f} | '
                      f'Val NDCG@1: {ndcg1:.6f} | '
                      f'Val MRR: {MRR_num:.6f}')
            
            # Early stopping and save best model
            prev_best_ndcg1 = NDCG_max_matrix[0][0]
            patience_num_matrix = ealy_stop(NDCG_max_matrix, patience_num_matrix, epoch_max_matrix,
                                            epoch, ndcg1, ndcg3, ndcg5, MRR_num)
            
            # If NDCG@1 improves, save model
            if ndcg1 > prev_best_ndcg1:
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'ndcg1': ndcg1,
                    'ndcg3': ndcg3,
                    'ndcg5': ndcg5,
                    'mrr': MRR_num,
                    'val_loss': val_loss.item()
                }, best_model_path)
                print(f'  --> Saved best model (NDCG@1: {ndcg1:.6f}) at epoch {epoch + 1}')
            
            if patience_num_matrix[0][0] >= args.patience:
                print(f'Early stopping triggered at epoch {epoch + 1}')
                break
    
    # Clear GPU cache
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    max_epoch = int(epoch_max_matrix[0][0])
    if max_epoch < len(result_list):
        print('Saving train result:', result_list[max_epoch][1:])
        print(f'the optimal epoch {max_epoch + 1}')
        return result_list[max_epoch][1:]
    else:
        print('Saving train result:', result_list[-1][1:])
        print(f'the optimal epoch {len(result_list)}')
        return result_list[-1][1:]