# -*- coding: utf-8 -*-
import warnings
from utils import *
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")


def data_lode():
    'Read the initial features files'
    microbe_feat = pd.read_csv('./MCGNN/Data/mic_sim176.txt', header=None, sep='\t').values
    gene_feat = pd.read_csv('./MCGNN/Data/gene_sim_BP301.csv', header=None).values
    disease_feat = pd.read_csv('./MCGNN/Data/dis_sim153.txt', header=None, sep='\t').values

    gene_features = torch.FloatTensor(gene_feat)
    disease_features = torch.FloatTensor(disease_feat)
    microbe_features = torch.FloatTensor(microbe_feat)

    features = {'g': gene_features, 'm': microbe_features, 'd': disease_features}
    in_size = {'g': gene_features.shape[1], 'm': microbe_features.shape[1],
               'd': disease_features.shape[1]}
    return features, in_size


def get_train_val_data(all_data, train_ind, val_ind, adj, seed):
    'To generate negative samples for the training set for 5-fold CV'
    neg_num_test = 30
    train_data_pos, val_data_pos = all_data[train_ind], all_data[val_ind]
    train_data_all, tr_neg_1_ls, tr_neg_2_ls, tr_neg_3_ls, tr_neg_4_ls, te_neg_1_ls, te_neg_2_ls, te_neg_3_ls, te_neg_4_ls = neg_data_generate(
        adj, train_data_pos, val_data_pos, neg_num_test, seed)
    np.random.shuffle(train_data_all)
    val_neg_data = []
    for i in te_neg_1_ls:
        if list(i)[-1] == 0:
            val_neg_data.append(i)
    train_data_pos = train_data_pos.copy().astype(int)
    val_data_pos = val_data_pos.copy().astype(int)
    return train_data_pos, np.array(tr_neg_1_ls), val_data_pos, np.array(val_neg_data)


def get_indep_data(adj, train_data_pos, val_data_pos, seed):
    'To generate negative samples for the training set for independent test'
    neg_num_test = 30
    train_data_all, tr_neg_1_ls, tr_neg_2_ls, tr_neg_3_ls, tr_neg_4_ls, te_neg_1_ls, te_neg_2_ls, te_neg_3_ls, te_neg_4_ls = neg_data_generate(
        adj, train_data_pos, val_data_pos, neg_num_test, seed)
    np.random.shuffle(train_data_all)
    val_neg_data = []
    for i in te_neg_1_ls:
        if list(i)[-1] == 0:
            val_neg_data.append(i)
    train_data_pos = train_data_pos.copy().astype(int)
    val_data_pos = val_data_pos.copy().astype(int)
    return train_data_pos, tr_neg_1_ls, val_data_pos, np.array(val_neg_data)


def neg_data_generate(adj_data_all, train_data_fix, val_data_fix, neg_num_test, seed):
    'A function used to generate negative samples'
    neg_num_train = 1
    np.random.seed(seed)
    train_neg_1_ls = []
    train_neg_2_ls = []
    train_neg_3_ls = []
    train_neg_4_ls = []
    val_neg_1_ls = []
    val_neg_2_ls = []
    val_neg_3_ls = []
    val_neg_4_ls = []
    arr_true = np.zeros((301, 176, 153))
    for line in adj_data_all:
        arr_true[int(line[0]), int(line[1]), int(line[2])] = 1
    arr_false_train = np.zeros((len(set(adj_data_all[:, 0])), len(set(adj_data_all[:, 1])),
                                len(set(adj_data_all[:, 2]))))
    
    # Generate negative samples for training
    for i in train_data_fix:
        ctn_1 = 0
        ctn_2 = 0
        ctn_3 = 0
        ctn_4 = 0
        tr_gene_ls = [j for j in range(0, arr_true.shape[0])]
        tr_mic_ls = [j for j in range(0, arr_true.shape[1])]
        tr_dis_ls = [j for j in range(0, arr_true.shape[2])]
        
        while ctn_1 < neg_num_train:
            a = np.random.randint(0, arr_true.shape[0] - 1)
            b = np.random.randint(0, arr_true.shape[1] - 1)
            c = np.random.randint(0, arr_true.shape[2] - 1)
            if arr_true[a, b, c] != 1 and arr_false_train[a, b, c] != 1:
                arr_false_train[a, b, c] = 1
                ctn_1 += 1
                train_neg_1_ls.append((a, b, c, 0))
        
        while ctn_2 < neg_num_train:
            b = int(i[1])
            c = int(i[2])
            a = np.random.choice(tr_gene_ls)
            if arr_true[a, b, c] != 1 and arr_false_train[a, b, c] != 1:
                arr_false_train[a, b, c] = 1
                ctn_2 += 1
                train_neg_2_ls.append((a, b, c, 0))
        
        while ctn_3 < neg_num_train:
            a = int(i[0])
            b = np.random.randint(0, arr_true.shape[1] - 1)
            c = int(i[2])
            if arr_true[a, b, c] != 1 and arr_false_train[a, b, c] != 1:
                arr_false_train[a, b, c] = 1
                ctn_3 += 1
                train_neg_3_ls.append((a, b, c, 0))
        
        while ctn_4 < neg_num_train:
            a = int(i[0])
            b = int(i[1])
            if len(tr_dis_ls) > 0:
                c = np.random.choice(tr_dis_ls)
                tr_dis_ls.remove(c)
                if arr_true[a, b, c] != 1 and arr_false_train[a, b, c] != 1:
                    arr_false_train[a, b, c] = 1
                    ctn_4 += 1
                    train_neg_4_ls.append((a, b, c, 0))
            else:
                break

    # Combine all training negative samples
    train_neg_arrays = []
    if len(train_neg_1_ls) > 0:
        train_neg_arrays.append(np.array(train_neg_1_ls))
    if len(train_neg_2_ls) > 0:
        train_neg_arrays.append(np.array(train_neg_2_ls))
    if len(train_neg_3_ls) > 0:
        train_neg_arrays.append(np.array(train_neg_3_ls))
    if len(train_neg_4_ls) > 0:
        train_neg_arrays.append(np.array(train_neg_4_ls))
    
    if train_neg_arrays:
        train_neg_all = np.vstack(train_neg_arrays)
        train_neg_all = np.vstack((train_neg_all, train_data_fix))
    else:
        train_neg_all = train_data_fix
    
    np.random.shuffle(train_neg_all)
    
    # Generate negative samples for validation
    for i in val_data_fix:
        neg_1_i = []
        neg_2_i = []
        neg_3_i = []
        neg_4_i = []
        
        arr_false_val_1 = np.zeros(
            (len(set(adj_data_all[:, 0])), len(set(adj_data_all[:, 1])), len(set(adj_data_all[:, 2]))))
        arr_false_val_2 = np.zeros(
            (len(set(adj_data_all[:, 0])), len(set(adj_data_all[:, 1])), len(set(adj_data_all[:, 2]))))
        arr_false_val_3 = np.zeros(
            (len(set(adj_data_all[:, 0])), len(set(adj_data_all[:, 1])), len(set(adj_data_all[:, 2]))))
        arr_false_val_4 = np.zeros(
            (len(set(adj_data_all[:, 0])), len(set(adj_data_all[:, 1])), len(set(adj_data_all[:, 2]))))
        
        neg_1_i.append(i)
        neg_2_i.append(i)
        neg_3_i.append(i)
        neg_4_i.append(i)
        
        cva_1 = 0
        cva_2 = 0
        cva_3 = 0
        cva_4 = 0
        
        gene_ls = [j for j in range(0, arr_true.shape[0])]
        mic_ls = [j for j in range(0, arr_true.shape[1])]
        dis_ls = [j for j in range(0, arr_true.shape[2])]
        
        while cva_1 < neg_num_test:
            a_1 = np.random.randint(0, arr_true.shape[0] - 1)
            b_1 = np.random.randint(0, arr_true.shape[1] - 1)
            c_1 = np.random.randint(0, arr_true.shape[2] - 1)
            if arr_true[a_1, b_1, c_1] != 1 and arr_false_train[a_1, b_1, c_1] != 1 and arr_false_val_1[
                a_1, b_1, c_1] != 1:
                arr_false_val_1[a_1, b_1, c_1] = 1
                cva_1 += 1
                neg_1_i.append((a_1, b_1, c_1, 0))
        
        np.random.shuffle(neg_1_i)
        val_neg_1_ls.extend(neg_1_i)
        
        while cva_2 < neg_num_test:
            b_2 = int(i[1])
            c_2 = int(i[2])
            if len(gene_ls) > 0:
                a_2 = np.random.choice(gene_ls)
                gene_ls.remove(a_2)
                if arr_true[a_2, b_2, c_2] != 1 and arr_false_train[a_2, b_2, c_2] != 1 and arr_false_val_2[
                    a_2, b_2, c_2] != 1:
                    arr_false_val_2[a_2, b_2, c_2] = 1
                    cva_2 += 1
                    neg_2_i.append((a_2, b_2, c_2, 0))
            else:
                break
        
        np.random.shuffle(neg_2_i)
        val_neg_2_ls.extend(neg_2_i)
        
        while cva_3 < neg_num_test:
            a_3 = int(i[0])
            c_3 = int(i[2])
            if len(mic_ls) > 0:
                b_3 = np.random.choice(mic_ls)
                mic_ls.remove(b_3)
                if arr_true[a_3, b_3, c_3] != 1 and arr_false_train[a_3, b_3, c_3] != 1 and arr_false_val_3[
                    a_3, b_3, c_3] != 1:
                    arr_false_val_3[a_3, b_3, c_3] = 1
                    cva_3 += 1
                    neg_3_i.append((a_3, b_3, c_3, 0))
            else:
                break
        
        np.random.shuffle(neg_3_i)
        val_neg_3_ls.extend(neg_3_i)
        
        while cva_4 < neg_num_test:
            a_4 = int(i[0])
            b_4 = int(i[1])
            if len(dis_ls) > 0:
                c_4 = np.random.choice(dis_ls)
                dis_ls.remove(c_4)
                if arr_true[a_4, b_4, c_4] != 1 and arr_false_train[a_4, b_4, c_4] != 1 and arr_false_val_4[
                    a_4, b_4, c_4] != 1:
                    arr_false_val_4[a_4, b_4, c_4] = 1
                    cva_4 += 1
                    neg_4_i.append((a_4, b_4, c_4, 0))
            else:
                break
        
        np.random.shuffle(neg_4_i)
        val_neg_4_ls.extend(neg_4_i)
    
    return train_neg_all, train_neg_1_ls, train_neg_2_ls, train_neg_3_ls, train_neg_4_ls, val_neg_1_ls, val_neg_2_ls, val_neg_3_ls, val_neg_4_ls


def generate_dataset():
    args = parameters_set()
    np.random.seed(args.seed)
    adj_data = np.loadtxt('./MCGNN/Data/g_m_d_pos_pairs.txt')
    np.random.shuffle(adj_data)
    cv_data = adj_data[int(0.1 * len(adj_data)):, :]
    indep_data = adj_data[:int(0.1 * len(adj_data)), :]
    fold_num = 0
    
    # Create directories
    import os
    for i in range(1, 6):
        os.makedirs(f'./MCGNN/Data/CV_data/CV_{i}', exist_ok=True)
    os.makedirs('./MCGNN/Data/indepent_data', exist_ok=True)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)
    for train_index, val_index in kf.split(cv_data):
        fold_num += 1
        train_data_pos, train_data_neg, val_data_pos, val_data_neg = get_train_val_data(cv_data, train_index, val_index,
                                                                                        adj_data, args.seed)
        np.savetxt('./MCGNN/Data/CV_data/CV_' + str(fold_num) + '/train_data_pos.csv', train_data_pos, delimiter=",",
                   fmt='%d')
        np.savetxt('./MCGNN/Data/CV_data/CV_' + str(fold_num) + '/train_data_neg.csv', train_data_neg, delimiter=',',
                   fmt='%d')
        np.savetxt('./MCGNN/Data/CV_data/CV_' + str(fold_num) + '/val_data_pos.csv', val_data_pos, delimiter=',', fmt='%d')
        np.savetxt('./MCGNN/Data/CV_data/CV_' + str(fold_num) + '/val_data_neg.csv', val_data_neg, delimiter=',', fmt='%d')
    
    train_data_pos, train_data_neg, test_data_pos, test_data_neg = get_indep_data(adj_data, cv_data, indep_data,
                                                                                  args.seed)
    np.savetxt('./MCGNN/Data/indepent_data/train_data_pos.csv', train_data_pos, delimiter=",", fmt='%d')
    np.savetxt('./MCGNN/Data/indepent_data/train_data_neg.csv', train_data_neg, delimiter=",", fmt='%d')
    np.savetxt('./MCGNN/Data/indepent_data/test_data_pos.csv', test_data_pos, delimiter=",", fmt='%d')
    np.savetxt('./MCGNN/Data/indepent_data/test_data_neg.csv', test_data_neg, delimiter=",", fmt='%d')