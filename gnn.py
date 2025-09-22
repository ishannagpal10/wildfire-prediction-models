import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
from typing import Optional, Union, Tuple
import math


class GraphSAGELayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 aggregation: str = 'mean', dropout: float = 0.0):
        super(GraphSAGELayer, self).__init__()
        self.aggregation = aggregation
        self.dropout = dropout
        self.sage_conv = SAGEConv(in_channels, out_channels, aggr=aggregation)
        self.layer_norm = nn.LayerNorm(out_channels)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.sage_conv(x, edge_index)
        x = self.layer_norm(x)
        x = F.relu(x)
        x = self.dropout_layer(x)
        return x


class GraphSAGE(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int,
                 aggregation: str = 'mean', dropout: float = 0.1,
                 residual: bool = True):
        super(GraphSAGE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.aggregation = aggregation
        self.dropout = dropout
        self.residual = residual
        layer_dims = [input_dim] + hidden_dims + [output_dim]
        self.layers = nn.ModuleList()
        for i in range(len(layer_dims) - 1):
            layer = GraphSAGELayer(
                in_channels=layer_dims[i],
                out_channels=layer_dims[i + 1],
                aggregation=aggregation,
                dropout=dropout if i < len(layer_dims) - 2 else 0.0
            )
            self.layers.append(layer)
        self.use_residual = residual and (input_dim == output_dim)
        if self.use_residual:
            self.residual_transform = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        original_x = x
        for layer in self.layers:
            x = layer(x, edge_index)
        if self.use_residual and x.shape == original_x.shape:
            x = x + self.residual_transform(original_x)
        return x


class SpatialGNN(nn.Module):
    def __init__(self, config: dict):
        super(SpatialGNN, self).__init__()
        self.input_dim = config.get('input_dim', 84)
        self.hidden_dims = config.get('hidden_dims', [128, 64])
        self.output_dim = config.get('output_dim', 32)
        self.aggregation = config.get('aggregation', 'mean')
        self.dropout = config.get('dropout', 0.1)
        self.residual = config.get('residual', True)
        self.pool_method = config.get('pool_method', 'mean')
        print(f"SpatialGNN: input_dim={self.input_dim}, hidden_dims={self.hidden_dims}, output_dim={self.output_dim}")
        if self.input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {self.input_dim}")
        if len(self.hidden_dims) < 1:
            raise ValueError(f"hidden_dims must have at least 1 layer, got {self.hidden_dims}")
        self.input_projection = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        self.graphsage = GraphSAGE(
            input_dim=self.hidden_dims[0],
            hidden_dims=self.hidden_dims[1:],
            output_dim=self.output_dim,
            aggregation=self.aggregation,
            dropout=self.dropout,
            residual=self.residual
        )
        self.output_projection = nn.Sequential(
            nn.Linear(self.output_dim, self.output_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout / 2),
            nn.Linear(self.output_dim // 2, 1)
        )
        self.graph_projection = nn.Sequential(
            nn.Linear(self.output_dim, self.output_dim),
            nn.ReLU(),
            nn.Linear(self.output_dim, 1)
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, data: Data, return_embeddings: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x, edge_index = data.x, data.edge_index
        x = self.input_projection(x)
        embeddings = self.graphsage(x, edge_index)
        node_predictions = self.output_projection(embeddings)
        if return_embeddings:
            return node_predictions, embeddings
        else:
            return node_predictions

    def forward_batch(self, batch: Batch, return_embeddings: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        return self.forward(batch, return_embeddings)

    def get_graph_embedding(self, data: Data) -> torch.Tensor:
        _, embeddings = self.forward(data, return_embeddings=True)
        batch = getattr(data, 'batch', None)
        if batch is None:
            batch = torch.zeros(embeddings.shape[0], dtype=torch.long, device=embeddings.device)
        if self.pool_method == 'mean':
            graph_embedding = global_mean_pool(embeddings, batch)
        elif self.pool_method == 'max':
            graph_embedding = global_max_pool(embeddings, batch)
        else:
            if batch.max() == 0:
                graph_embedding = torch.mean(embeddings, dim=0, keepdim=True)
            else:
                graph_embedding = global_mean_pool(embeddings, batch)
        return graph_embedding

    def predict_graph_level(self, data: Data) -> torch.Tensor:
        graph_embedding = self.get_graph_embedding(data)
        return self.graph_projection(graph_embedding)


class GNNConfig:
    def __init__(self):
        self.input_dim = 84
        self.hidden_dims = [128, 64, 32]
        self.output_dim = 32
        self.aggregation = 'mean'
        self.residual = True
        self.dropout = 0.1
        self.learning_rate = 1e-3
        self.weight_decay = 1e-4
        self.pool_method = 'mean'

    def to_dict(self) -> dict:
        return {
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims,
            'output_dim': self.output_dim,
            'aggregation': self.aggregation,
            'residual': self.residual,
            'dropout': self.dropout,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'pool_method': self.pool_method
        }

    @classmethod
    def from_dict(cls, config_dict: dict):
        config = cls()
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config


def create_gnn_model(config: Union[dict, GNNConfig] = None) -> SpatialGNN:
    if config is None:
        config = GNNConfig()
    if isinstance(config, GNNConfig):
        config = config.to_dict()
    return SpatialGNN(config)


def test_graphsage_layer():
    print("Testing GraphSAGE Layer...")
    num_nodes = 10
    in_channels = 84
    out_channels = 64
    x = torch.randn(num_nodes, in_channels)
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long)
    layer = GraphSAGELayer(in_channels, out_channels, aggregation='mean', dropout=0.1)
    layer.eval()
    with torch.no_grad():
        output = layer(x, edge_index)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Expected shape: ({num_nodes}, {out_channels})")
    assert output.shape == (num_nodes, out_channels), f"Expected {(num_nodes, out_channels)}, got {output.shape}"
    print("  GraphSAGE Layer test passed!")


def test_spatial_gnn():
    print("\nTesting SpatialGNN Model...")
    config = {
        'input_dim': 84,
        'hidden_dims': [128, 64],
        'output_dim': 32,
        'aggregation': 'mean',
        'dropout': 0.1,
        'residual': True,
        'pool_method': 'mean'
    }
    model = SpatialGNN(config)
    model.eval()
    num_nodes = 20
    x = torch.randn(num_nodes, config['input_dim'])
    edge_index = torch.randint(0, num_nodes, (2, 40))
    data = Data(x=x, edge_index=edge_index)
    with torch.no_grad():
        predictions = model(data)
        predictions_with_emb, embeddings = model(data, return_embeddings=True)
        graph_embedding = model.get_graph_embedding(data)
        graph_prediction = model.predict_graph_level(data)
    print(f"  Node predictions shape: {predictions.shape}")
    print(f"  Node embeddings shape: {embeddings.shape}")
    print(f"  Graph embedding shape: {graph_embedding.shape}")
    print(f"  Graph prediction shape: {graph_prediction.shape}")
    assert predictions.shape == (num_nodes, 1), f"Expected {(num_nodes, 1)}, got {predictions.shape}"
    assert embeddings.shape == (num_nodes, 32), f"Expected {(num_nodes, 32)}, got {embeddings.shape}"
    assert graph_embedding.shape == (1, 32), f"Expected {(1, 32)}, got {graph_embedding.shape}"
    assert graph_prediction.shape == (1, 1), f"Expected {(1, 1)}, got {graph_prediction.shape}"
    print("  SpatialGNN Model test passed!")


def test_model_integration():
    print("\nTesting Model Integration...")
    try:
        from utils import WildfireDataLoader, GraphConstructor
        loader = WildfireDataLoader()
        graph_constructor = GraphConstructor(loader)
        sample_df = loader.load_data_split('train', as_pandas=True, sample_size=30)
        batch_df = sample_df.head(20)
        graph_data = graph_constructor.create_graph_data(batch_df, include_target=True)
        model = create_gnn_model()
        model.eval()
        with torch.no_grad():
            predictions = model(graph_data)
            _, embeddings = model(graph_data, return_embeddings=True)
        print(f"  Real data - Nodes: {graph_data.x.shape[0]}")
        print(f"  Real data - Edges: {graph_data.edge_index.shape[1]}")
        print(f"  Real data - Features: {graph_data.x.shape[1]}")
        print(f"  Model predictions: {predictions.shape}")
        print(f"  Model embeddings: {embeddings.shape}")
        assert predictions.shape[0] == graph_data.x.shape[0], "Prediction count mismatch"
        assert embeddings.shape[0] == graph_data.x.shape[0], "Embedding count mismatch"
        print("  Model Integration test passed!")
    except ImportError as e:
        print(f"  Model Integration test skipped: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("TESTING GRAPHSAGE GNN IMPLEMENTATION")
    print("=" * 60)
    test_graphsage_layer()
    test_spatial_gnn()
    test_model_integration()
    print("\n" + "=" * 60)
    print("ALL GNN TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 60)
