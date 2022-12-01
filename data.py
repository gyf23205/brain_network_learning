import numpy as np
import os
import torch
from torch_geometric.data import Data, Batch


def pre_load(X_dir, Y_dir):
    filenames = os.listdir(X_dir)
    X = []
    Y = []
    num_node = []
    for file in filenames:
        X_full_path = os.path.join(X_dir, '{}'.format(file))
        Y_full_path = os.path.join(Y_dir, '{}'.format(file))
        x_tmp = np.load(X_full_path)
        y_tmp = np.load(Y_full_path)
        X.append(x_tmp)
        Y.append(y_tmp)
        num_node.append(x_tmp.shape[0])
    return X, Y, num_node


def adj_to_coo(adjs):
    """
    :param adj: Type: list, shape of elements in the list:(a, a)
    :return: Type: list, shape of elements in the list is a pytorch sparse tensor
    """
    edge_index = []
    edge_attr = []
    for i, adj in enumerate(adjs):
        row_idx, col_idx = np.nonzero(adj)
        idx = torch.tensor(np.array([row_idx, col_idx]), dtype=torch.long)
        v = torch.tensor(adj[row_idx, col_idx], dtype=torch.float)
        # for r, c in zip(row_idx, col_idx):
        #     temp.append(np.array([r, c, adj[r, c]]))
        edge_index.append(idx)
        edge_attr.append(v)  # (v[:, None])
    return edge_index, edge_attr


def preprocess(adj_dir, label_dir, target_dir):
    X, Y, num_node = pre_load(adj_dir, label_dir)
    index, value = adj_to_coo(X)
    for i, (edge_idx, edge_attr) in enumerate(zip(index, value)):
        node_attr = torch.ones(num_node[i], 2)
        d = Data(x=node_attr, edge_index=edge_idx, edge_attr=edge_attr)
        d.label = int(Y[i])
        # print(os.path.join(target_dir, f'network{i}.pt'))
        torch.save(d, os.path.join(target_dir, f'network{i}.pt'))


def load(dir):
    data_list = []
    filenames = os.listdir(dir)
    for file in filenames:
        path = os.path.join(dir, file)
        data_list.append(torch.load(path))
    return data_list


if __name__ == '__main__':
    # preprocess('./modal1_networks', './labels/gender', 'model1_COO')
    adj_dir = './modal1_networks'
    label_dir = './labels/gender'
    X, Y, num_node = pre_load(adj_dir, label_dir)
    data_list = []
    index, value = adj_to_coo(X[:10])
    for i, (edge_idx, edge_attr) in enumerate(zip(index, value)):
        node_attr = torch.ones(num_node[i], 2)
        d = Data(x=node_attr, edge_index=edge_idx, edge_attr=edge_attr)
        d.label = int(Y[i])
        data_list.append(d)
    batch1 = Batch.from_data_list(data_list)

    train_batch1 = batch1[:130]
    test_batch1 = batch1[130:]
    epoch = 100
    # for i in range(epoch):
    #     train_data = training_set[i % 2]
    #     out = model(train_data.x, train_data.edge_index,...)
    #     loss = F.loss(out, train_data.label)
    #     loss.backward()
    #     .step
    print()