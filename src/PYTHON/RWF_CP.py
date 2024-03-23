# -*- coding: utf-8 -*-
from __future__ import print_function,division

import networkx as nx
import scipy.io as sio
from scipy import sparse
from os import path
import scipy.sparse as sp
from collections import deque
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch
import os
from itertools import combinations


import numpy as np
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split, cross_validate, RepeatedStratifiedKFold, PredefinedSplit
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV
from scipy.spatial import distance
from tqdm import tqdm
import cpnet



def compute_gamma(embeddings, samples=100):
    indices = np.random.choice(embeddings.shape[0], 100, replace=False)
    sampled_embeddings = embeddings[indices, :]

    pairwise_distances = distance.cdist(sampled_embeddings, sampled_embeddings) ** 2
    median_of_distances = np.median(pairwise_distances)
    gamma = 1 / median_of_distances

    return gamma


def run_classification(args, features, labels, kernel='precomputed', outer_iters=10, scaling='std'):
    if kernel == 'precomputed':
        scaling = 'none'

    if scaling == 'minmax':
        scaler = MinMaxScaler()
        X = scaler.fit_transform(features)
    elif scaling == 'std':
        scaler = StandardScaler()
        X = scaler.fit_transform(features)
    elif scaling == 'kernel':
        X = features
    else:
        X = features

    y = labels

    model = SVC(kernel=kernel)

    if kernel == "rbf":
        parameter_space = {
            'C': [10 ** i for i in range(args.lowc, args.highc)],
            # 'gamma': [10**i for i in range(-3,4)]+[compute_gamma(X)],
            'gamma': [compute_gamma(X)],
        }
    else:
        parameter_space = {
            'C': [10 ** i for i in range(args.lowc, args.highc)]
        }

    rskfcv = RepeatedStratifiedKFold(n_splits=10, n_repeats=outer_iters, random_state=4702149)
    clf = GridSearchCV(model, parameter_space, cv=rskfcv, verbose=3, return_train_score=True, n_jobs=-1)
    clf.fit(X, y)

    means = clf.cv_results_['mean_test_score']
    train_means = clf.cv_results_['mean_train_score']
    stds = clf.cv_results_['std_test_score']
    train_stds = clf.cv_results_['std_train_score']
    param_list = clf.cv_results_['params']

    return means, train_means, stds, train_stds, param_list


def run_single_split_classification(features, labels, TVT_split, kernel='precomputed', scaling='std'):
    if kernel == 'precomputed':
        scaling = 'none'

    if scaling == 'minmax':
        scaler = MinMaxScaler()
        X = scaler.fit_transform(features)
    elif scaling == 'std':
        scaler = StandardScaler()
        X = scaler.fit_transform(features)
    else:
        X = features

    y = labels

    num_train, num_val, num_test = TVT_split

    if kernel == "precomputed":
        X = X[:, :num_train + num_val]

    X_trainval = X[:num_train + num_val, :]
    y_trainval = y[:num_train + num_val]

    X_test = X[num_train + num_val:, :]
    y_test = y[num_train + num_val:]

    model = SVC(kernel=kernel)

    if kernel == "rbf":
        parameter_space = {
            'C': [10 ** i for i in range(-3, 4)],
            'gamma': [compute_gamma(X)]
        }
    else:
        parameter_space = {
            'C': [10 ** i for i in range(-3, 4)]
        }

    validation_fold = [-1] * num_train + [0] * num_val
    ps = PredefinedSplit(test_fold=validation_fold)

    clf = GridSearchCV(model, parameter_space, cv=ps, verbose=3, return_train_score=True, n_jobs=-1)
    clf.fit(X_trainval, y_trainval)

    validation_means = clf.cv_results_['mean_test_score']
    train_means = clf.cv_results_['mean_train_score']
    validation_stds = clf.cv_results_['std_test_score']
    train_stds = clf.cv_results_['std_train_score']
    param_list = clf.cv_results_['params']

    test_score = clf.best_estimator_.score(X_test, y_test)

    return validation_means, train_means, validation_stds, train_stds, param_list, test_score


def sys_normalized_adjacency(adj):
    adj = sp.coo_matrix(adj)
    # adj = adj + sp.eye(adj.shape[0])
    row_sum = np.array(adj.sum(1))
    row_sum = (row_sum == 0) * 1 + row_sum
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def normalize_to_0_1(array):
    min_val = np.min(array)
    max_val = np.max(array)
    range_val = max_val - min_val
    if range_val == 0:
        return np.ones_like(array)
    else:
        normalized_array = (array - min_val) / range_val
        return normalized_array

def reorder_adjacency_matrix_with_x(adj_matrix, x_features):
    ## ranking by degree
    # degrees = np.asarray(adj_matrix.sum(axis=0))[0]
    # sorted_indices = np.argsort(degrees)[::-1]
    # reordered_adj = adj_matrix[:, sorted_indices][sorted_indices, :]
    # degrees = sorted(degrees, reverse=True)
    # degrees = 1 + np.array(degrees)
    # d_inv_sqrt = np.power(degrees, -0.5).flatten()
    # d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    # d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    ## use core-periphery heuristic
    G = nx.from_scipy_sparse_array(adj_matrix)
    algorithm = cpnet.KM_config()
    algorithm.detect(G)
    c = algorithm.get_pair_id()
    x = algorithm.get_coreness()
    # print(c) # indicates group ids
    # print(x) # indicates core and periphery
    x_id = list(x.values())
    sorted_indices = np.argsort(x_id)[::-1]
    reordered_adj = adj_matrix[:, sorted_indices][sorted_indices, :]
    reordered_x_features = x_features[sorted_indices,:]
    degrees = np.asarray(reordered_adj.sum(axis=0))[0]
    degrees = 1 + np.array(degrees)
    d_inv_sqrt = np.power(degrees, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    num_of_one = sum(x_id)
    return reordered_adj, num_of_one, reordered_x_features, d_mat_inv_sqrt.dot(reordered_adj).dot(d_mat_inv_sqrt).tocoo()


def extract_LM_embeddings_label(args, root, dataset):
    print("Extracting embeddings for dataset", dataset)

    As = sio.loadmat(root + dataset + '/' + dataset + '_all_graphs.mat')['all_graphs'].flatten()

    data_dir1 = root + dataset + '/' + dataset + '_all_OH_labels.mat'
    if os.path.exists(data_dir1):
        x_onehot_label_all = sio.loadmat(data_dir1)['all_OH_labels'].flatten()
    data_dir2 = root + dataset + '/' + dataset + '_all_attributes.mat'
    if os.path.exists(data_dir2):
        x_attribute_all = sio.loadmat(data_dir2)['all_attributes'].flatten()

    all_embeddings = []
    ### number of splits
    k = args.k
    ### lengh of random walk
    tau = args.tau

    for i in tqdm(range(len(As))):
        print(i, end='\r')
        A = As[i]

        num_nodes = A.shape[0]
        num_edges = A.count_nonzero()

        x_feat1 = np.maximum(1, np.asarray(np.sum(A, axis=1)))
        x_feat = normalize_to_0_1(x_feat1)

        A_new, num_of_one, X, P = reorder_adjacency_matrix_with_x(A,x_feat)

        mat_p = sparse_mx_to_torch_sparse_tensor(P)
        x_p = torch.eye(num_nodes, dtype=torch.float32)
        X = torch.from_numpy(X)
        X = X.type(torch.float32)
        start = {}
        end = {}

        start[0] = 0
        end[0] = num_of_one - 1
        start[1] = num_of_one
        end[1] = num_nodes - 1
        emd = []
        tuples = list(combinations(range(k), 2))

        for j in range(tau):
            x_p = torch.spmm(mat_p, x_p)
            for id in range(k):
                emd.append(torch.diag(x_p[start[id]:end[id], start[id]:end[id]]).mean())
                emd.append(x_p[start[id]:end[id], start[id]:end[id]].mean())

            for tup in tuples:
                emd.append(x_p[start[tup[0]]:end[tup[0]], start[tup[1]]:end[tup[1]]].mean())
        all_embeddings.append(emd)
    return np.vstack(all_embeddings)

def save_array_to_text(array, filename, folder):
    # Check if the folder exists, create it if not
    if not os.path.exists(folder):
        os.makedirs(folder)
    # Create the full path for the file
    full_path = os.path.join(folder, filename)
    # Save the NumPy array to a text file
    np.savetxt(full_path, array, delimiter=",")

def main():

    print("exp with CP partitioning, without node feature")
    ### COLLAB PROTEINS

    parser = argparse.ArgumentParser(description='Generate RWF embeddings for graph dataset')
    parser.add_argument('--dataset', type=str, default="MUTAG",
                        help='name of dataset')

    parser.add_argument('--embedder', type=str, default='RWF',
                        help='type of embedder/kernel')

    parser.add_argument('--k', type=int, default=2,
                        help='number of node separations')
    parser.add_argument('--tau', type=int, default=6,
                        help='length of random walk')
    parser.add_argument('--lowc', type=int, default=-3,
                        help='minimum of C')
    parser.add_argument('--highc', type=int, default=4,
                        help='maximum of C')

    args = parser.parse_args()

    best = []
    for args.tau in [3,4,5,6,7,8,9,10]:

        root_orig = '../../data/processed/'
        root_emb = '../../embeddings/'

        embeddings = extract_LM_embeddings_label(args, root=root_orig, dataset=args.dataset)

        embedding_file = args.dataset + '_RWF' + '_' + str(args.k) + '_' + str(args.tau) + '_CP.csv'
        save_array_to_text(embeddings, embedding_file, root_emb + args.dataset)


        print("Dataset:", args.dataset)
        print("Embedder:", "RWF Embedder")

        root_emb = '../../embeddings/' + args.dataset + '/'
        root_orig = '../../data/processed/' + args.dataset + '/'


        feat = pd.read_csv(root_emb + embedding_file, header=None)
        labels = pd.read_csv(root_orig + args.dataset + "_graph_labels.txt", header=None)


        feat = feat.fillna(0)
        nonzero_columns = feat.any()
        feat = feat.loc[:, nonzero_columns]
        feat = feat.values
        labels = labels.values.flatten()

        kernel = 'rbf'
        scaling = 'std' # none minmax std

        print("Feature matrix size:", feat.shape)
        print("Running classification with", kernel, "kernel.")
        if args.dataset in ["BANDPASS", "DBLP_v1", "reddit_threads", "twitch_egos", "github_stargazers", "deezer_ego_nets"]:
            TVT_split = (3000, 1000, 1000)
            means, train_means, stds, train_stds, param_list, test_score = run_single_split_classification(feat, labels,
                                                                                                           TVT_split,
                                                                                                           kernel=kernel,
                                                                                                           scaling=scaling)
        else:
            means, train_means, stds, train_stds, param_list = run_classification(args, feat, labels, kernel=kernel, scaling=scaling)

        best_mean = []
        best_std = []
        best_svm_params = []
        for mean, std, train_mean, train_std, params in zip(means, stds, train_means, train_stds, param_list):
            print("Train: %0.3f (+/-%0.03f) for %r" % (train_mean, train_std, params), end='\t')
            print("Test: %0.3f (+/-%0.03f) for %r" % (mean, std, params))
            best_mean.append(mean)
            best_std.append(std)
            best_svm_params.append(params)

        idx, _ = max(enumerate(best_mean), key=lambda x: x[1])

        file_out = '../../results/' + args.dataset + '_' + str(round(best_mean[idx],3)) + '_' + str(args.k) + '_' + str(args.tau) + '_CP.txt'

        argsDict = args.__dict__
        with open(file_out, 'w') as f:
            f.write("mean and std: ")
            f.write('\n')
            f.write(str(round(best_mean[idx],3)) + '(' + str(round(best_std[idx],3)) + ')')
            f.write('\n')
            f.write(str(best_mean[idx]) + '(' + str(best_std[idx]) + ')')
            f.write('\n')

            f.writelines('------------------ svm parameters ------------------' + '\n')
            for eachArg, value in best_svm_params[idx].items():
                f.writelines(eachArg + ' : ' + str(value) + '\n')
            f.writelines('------------------- end -------------------')
            f.write('\n')
            f.writelines('------------------ model parameters ------------------' + '\n')
            for eachArg, value in argsDict.items():
                f.writelines(eachArg + ' : ' + str(value) + '\n')
            f.writelines('------------------- end -------------------')
        f.close()
        best.append(best_mean[idx])

    print("exp with CP partitioning, without node feature, varying tau: ")
    print(best)
    best_file = '../../results/' + args.dataset + '_' + str(args.k) + '_CP_results.txt'
    with open(best_file, 'w') as f:
        for item in best:
            f.write(str(item) + '\n')
    f.close()

if __name__ == '__main__':
    main()