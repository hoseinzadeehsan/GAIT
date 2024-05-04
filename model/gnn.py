import torch
import torch.nn as nn
from dgl import DGLGraph
from dgl.nn.pytorch import GATConv
from dgl.nn.pytorch import GatedGraphConv
from dgl.nn.pytorch import GraphConv


class GAT(nn.Module):
    def __init__(self,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual):
        super(GAT, self).__init__()
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation

        # There is no hidden layer
        if num_layers == 0:
            self.gat_layers.append(GATConv(
                in_dim, num_classes, heads[0],
                feat_drop, attn_drop, negative_slope, residual, None))

        else:
            # input projection
            self.gat_layers.append(GATConv(
                in_dim, num_hidden[0], heads[0],
                feat_drop, attn_drop, negative_slope, residual, self.activation))
            # hidden layers
            for l in range(1, num_layers):
                # due to multi-head, the in_dim = num_hidden * num_heads
                self.gat_layers.append(GATConv(
                    num_hidden[l - 1] * heads[l - 1], num_hidden[l], heads[l],
                    feat_drop, attn_drop, negative_slope, residual, self.activation))
            # output projection
            self.gat_layers.append(GATConv(
                num_hidden[-1] * heads[-2], num_classes, heads[-1],
                feat_drop, attn_drop, negative_slope, residual, None))

    def forward(self, g: DGLGraph, inputs):
        h = inputs
        for l in range(self.num_layers):
            h = self.gat_layers[l](g, h).flatten(1)
        # output projection
        logits = self.gat_layers[-1](g, h).mean(1)
        return logits


class GGNN(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 shared_weight=False,
                 dropout=0):
        super(GGNN, self).__init__()
        self.layers = nn.ModuleList()

        if shared_weight:
            self.layers.append(GatedGraphConv(in_feats, n_classes, n_layers + 1, 1))
        else:
            # input layer
            self.layers.append(GatedGraphConv(in_feats, n_hidden[0], 1, 1))
            # hidden layers
            for i in range(n_layers - 1):
                self.layers.append(GatedGraphConv(n_hidden[i - 1], n_hidden[i], 1, 1))
            # output layer
            self.layers.append(GatedGraphConv(n_hidden[-1], n_classes, 1, 1))

            self.dropout = nn.Dropout(p=dropout)

    def forward(self, g: DGLGraph, features, edges):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h, edges)
        return h

class GCNLayer(nn.Module):

    def __init__(self, in_feats, out_feats,
                 residual=True, batchnorm=True, dropout=0, activation=None):
        super(GCNLayer, self).__init__()

        self.activation = activation
        self.graph_conv = GraphConv(in_feats=in_feats, out_feats=out_feats, activation=activation)
        self.dropout = nn.Dropout(dropout)

        self.residual = residual
        if residual:
            if in_feats == out_feats:
                self.res_connection = nn.Identity()
            else:
                self.res_connection = nn.Linear(in_feats, out_feats)

        self.bn = batchnorm
        if batchnorm:
            self.bn_layer = nn.BatchNorm1d(out_feats)

    def forward(self, g, feats):
        new_feats = self.graph_conv(g, feats)
        if self.residual:
            res_feats = self.res_connection(feats)
            new_feats = new_feats + res_feats
        new_feats = self.dropout(new_feats)

        if self.bn:
            new_feats = self.bn_layer(new_feats)
        return new_feats


class GCN(nn.Module):

    def __init__(self, in_feats, hidden_feats, n_classes, n_layers, activation=None, residual=True, batchnorm=False,
                 dropout=0):
        super(GCN, self).__init__()
        self.gnn_layers = nn.ModuleList()
        if n_layers == 0:
            self.gnn_layers.append(GCNLayer(in_feats, n_classes,
                                            residual, batchnorm, 0))
        else:
            for i in range(n_layers):
                self.gnn_layers.append(GCNLayer(in_feats, hidden_feats[i],
                                                residual, batchnorm, dropout, activation))
                in_feats = hidden_feats[i]

            self.gnn_layers.append(GCNLayer(in_feats, n_classes,
                                            residual, batchnorm, 0))

    def forward(self, g, feats):
        for layer in self.gnn_layers:
            feats = layer(g, feats)
        return feats
