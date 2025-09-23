from enum import Enum
from inspect import signature
from typing import Callable, List

import torch
import torch.nn as nn

from dhg.nn import GCNConv, HGNNPConv
from torch.nn import TransformerEncoderLayer
from torch_geometric.nn import SAGEConv, GraphSAGE, GCN, GAT


class LayerType(Enum):
    TRANSFORMERENCODER = 0
    GCNCONV = 1
    HGNNPCONV = 2
    LINEAR = 3
    SAGECONV = 4
    GRAPHSAGE = 5
    GCN = 6
    GAT = 7


class Layer(nn.Module):
    _PARAM_MAPPING = {
        LayerType.GCNCONV: {"drop_rate": "drop_rate"},
        LayerType.HGNNPCONV: {"drop_rate": "drop_rate"},
        LayerType.GRAPHSAGE: {"drop_rate": "dropout"},
    }

    def __init__(self, layer_type: LayerType, in_channels: int, out_channels: int, act: Callable = nn.ReLU(), **kwargs):
        super().__init__()
        self.layer_type = layer_type

        params = self._adapt_parameters(kwargs)

        if layer_type == LayerType.GCNCONV:
            self.layer = GCNConv(
                in_channels,
                out_channels,
                use_bn=params.get("use_bn", True),
                drop_rate=params.get("dropout", 0.0),
                is_last=params.get("last_conv", False),
            )
        elif layer_type == LayerType.HGNNPCONV:
            self.layer = HGNNPConv(
                in_channels,
                out_channels,
                use_bn=params.get("use_bn", True),
                drop_rate=params.get("dropout", 0.0),
                is_last=params.get("last_conv", False),
            )
        elif layer_type == LayerType.SAGECONV:
            self.layer = SAGEConv(in_channels=in_channels, out_channels=out_channels, **self._filter_params(SAGEConv, params))
        elif layer_type == LayerType.GRAPHSAGE:
            self.layer = GraphSAGE(in_channels=in_channels, out_channels=out_channels, **self._filter_params(GraphSAGE, params))
        elif layer_type == LayerType.GCN:
            self.layer = GCN(in_channels=in_channels, out_channels=out_channels, **self._filter_params(GCN, params))
        elif layer_type == LayerType.GAT:
            self.layer = GAT(in_channels=in_channels, out_channels=out_channels, **self._filter_params(GCN, params))
        elif layer_type == LayerType.TRANSFORMERENCODER:
            if in_channels != out_channels:
                raise ValueError("TransformerEncoderLayer need the same dim between in and out channels!")
            self.layer = TransformerEncoderLayer(d_model=in_channels, **self._filter_params(TransformerEncoderLayer, params))
        elif layer_type == LayerType.LINEAR:
            self.linear = nn.Linear(in_channels, out_channels)
            self.bn = nn.BatchNorm1d(in_channels) if params.get("use_bn", True) else None
            self.dropout = nn.Dropout(params.get("dropout", 0.0)) if params.get("dropout", 0.0) > 0.0 else None
        else:
            raise ValueError(f"Unsupported layer type: {layer_type}")

        self.activation = act

    def _adapt_parameters(self, params: dict) -> dict:
        mapping = self._PARAM_MAPPING.get(self.layer_type, {})
        return {mapping.get(k, k): v for k, v in params.items()}

    def _filter_params(self, cls, params: dict) -> dict:
        sig = signature(cls.__init__)
        valid_params = sig.parameters.keys()
        return {k: v for k, v in params.items() if k in valid_params}

    def forward(self, x: torch.Tensor, graph=None, edge_index=None, edge_weight=None, **kwargs) -> torch.Tensor:
        if self.layer_type in [LayerType.GCNCONV, LayerType.HGNNPCONV]:
            x = self.layer(x, graph)
        elif self.layer_type in [LayerType.SAGECONV, LayerType.GRAPHSAGE, LayerType.GCN, LayerType.GAT]:
            x = self.layer(x, edge_index, edge_weight=edge_weight)
        elif self.layer_type in [LayerType.TRANSFORMERENCODER]:
            x = self.layer(x)
        elif self.layer_type == LayerType.LINEAR:
            if self.bn is not None:
                x = self.bn(x)
            x = self.activation(x)
            if self.dropout is not None:
                x = self.dropout(x)
            x = self.linear(x)

        return x


class Net(nn.Module):

    def __init__(self, layers: List[Layer]):
        super(Net, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, graph=None, edge_index=None, edge_weight=None, **kwargs):
            for layer in self.layers:
                x = layer(x, graph, edge_index, edge_weight, **kwargs)
            return self.sigmoid(x)
