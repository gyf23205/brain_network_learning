import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool


class MyGCN(torch.nn.Module):
    def __init__(self, input_channels, output_channels, hidden_state):
        super(MyGCN, self).__init__()
        self.conv1 = GCNConv(input_channels, hidden_state)
        self.bn1 = torch.nn.BatchNorm1d(hidden_state)
        self.conv2 = GCNConv(hidden_state, hidden_state)
        # self.bn2 = torch.nn.BatchNorm1d(hidden_state)
        self.lin1 = torch.nn.Linear(hidden_state, output_channels)
        # self.linears = torch.nn.ModuleList()
        # for _ in range(2):  # Number of conv layers
        #     self.linears.append(torch.nn.Linear(hidden_state, hidden_state))

    def forward(self, x, edge_index, edge_weights, batch):
        # h_s = []
        h = self.conv1(x=x, edge_index=edge_index, edge_weight=edge_weights)
        h = self.bn1(h)
        h = F.relu(h)
        h = F.dropout(h, 0.5, training=self.training)
        # h_s.append(h)

        h = self.conv2(x=h, edge_index=edge_index, edge_weight=edge_weights)
        # h = self.bn2(h)
        # h = F.relu(h)
        # h_s.append(h)
        # Graph pooling
        h_graph = global_add_pool(h, batch)
        # for layer, layer_h in enumerate(h_s):
        #     layer_pool = global_add_pool(h_s[layer], batch)
        #     h_graph += F.dropout(self.linears[layer](layer_pool), 0.5, training=self.training)

        return self.lin1(h_graph)

