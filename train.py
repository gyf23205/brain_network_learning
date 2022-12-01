from GCN import MyGCN
import torch
import numpy as np
import data
from torch_geometric.data import DataLoader, Batch
import matplotlib.pyplot as plt


def train_step(model, batch_graph, optimizer, cross_entropy):
    logits = model(batch_graph.x, batch_graph.edge_index, batch_graph.edge_attr, batch=batch_graph.batch)
    # Loss for MyGCN
    loss = cross_entropy(logits, batch_graph.label)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Get accuracy
    pred = torch.argmax(logits, dim=1)
    acc = sum(pred == batch_graph.label) / len(pred)
    return loss, acc


def val_step(model, batch_graph, cross_entropy):
    logits = model(batch_graph.x, batch_graph.edge_index, batch_graph.edge_attr, batch=batch_graph.batch)
    loss = cross_entropy(logits, batch_graph.label)
    pred = torch.argmax(logits, dim=1)
    acc = sum(pred == batch_graph.label) / len(pred)
    return loss, acc


def train(data_list, model, iters, lr, train_ratio, batch_size):
    model.train()
    split = int(len(data_list) * train_ratio)
    data_train = data_list[:split]
    data_val = data_list[split:]
    loader_train = DataLoader(data_train, batch_size=batch_size, shuffle=True)
    dataset_val = Batch.from_data_list(data_val).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    cross_entropy = torch.nn.CrossEntropyLoss()
    best_so_far = np.inf
    train_loss_hist = []
    train_acc_hist = []
    val_loss_hist = []
    val_acc_hist = []
    for i in range(iters):
        # model.train()
        loss_batch_hist = []
        acc_batch_hist = []
        for batch_graph in iter(loader_train):
            batch_graph.to(device)
            loss_train, acc_train = train_step(model, batch_graph, optimizer, cross_entropy)
            loss_batch_hist.append(loss_train.cpu().detach().numpy())
            acc_batch_hist.append(acc_train.cpu().detach().numpy())
        loss_average = np.mean(loss_batch_hist)
        acc_average = np.mean(acc_batch_hist)
        train_loss_hist.append(np.mean(loss_average))
        train_acc_hist.append(np.mean(acc_average))
        print(f'Epoch {i}, training loss: {loss_average}, training acc: {acc_average}')

        # Validation and save the best so far
        if i % 5 == 0:
            model.eval()
            loss_val, acc_val = val_step(model, dataset_val, cross_entropy)
            val_loss_hist.append(loss_val.cpu().detach().numpy())
            val_acc_hist.append(acc_val.cpu().detach().numpy())
            print(f'Val loss: {loss_val}, val acc: {acc_val}')
            if loss_val < best_so_far:
                best_so_far = loss_val
                print('save')
                # torch.save(model.state_dict(), 'saved_model/MyGCN12.pth')
    return train_loss_hist, val_loss_hist, train_acc_hist, val_acc_hist


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_list = data.load('./model1_COO')
    model = MyGCN(input_channels=2, output_channels=2, hidden_state=16).to(device)
    train_loss_hist, val_loss_hist, train_acc_hist, val_acc_hist = \
        train(data_list, model, iters=3000, lr=0.0001, train_ratio=0.9, batch_size=128)
    plt.subplot(221)
    plt.plot(train_loss_hist)
    plt.title('Training loss')
    plt.subplot(222)
    plt.plot(train_acc_hist)
    plt.title('Training accuracy')
    plt.subplot(223)
    plt.plot(val_loss_hist)
    plt.title('Validation loss')
    plt.subplot(224)
    plt.plot(val_acc_hist)
    plt.title('Validation accuracy')
    plt.show()

