import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from models.gnn import SpatialGNN
from models.lstm import TemporalLSTM


@dataclass
class PipelineConfig:
    input_dim: int
    gnn_hidden_dims: List[int]
    gnn_output_dim: int
    gnn_aggregation: str
    gnn_dropout: float
    lstm_hidden_dim: int
    lstm_num_layers: int
    lstm_output_dim: int
    lstm_dropout: float
    lstm_bidirectional: bool
    sequence_length: int
    forecast_horizon: int
    learning_rate: float
    weight_decay: float


class WildfireForecastingPipeline(nn.Module):
    """
    GNN + LSTM pipeline:
      - SpatialGNN encodes each day's graph -> node embeddings
      - TemporalLSTM encodes a historical sequence (global context)
      - Concatenate node embeddings with temporal context -> predict FRP per node
    """

    def __init__(self, cfg: PipelineConfig):
        super().__init__()
        self.cfg = cfg

        self.gnn = SpatialGNN(
            config={
                'input_dim': cfg.input_dim,
                'hidden_dims': cfg.gnn_hidden_dims,
                'output_dim': cfg.gnn_output_dim,
                'aggregation': cfg.gnn_aggregation,
                'dropout': cfg.gnn_dropout,
                'residual': True,
                'pool_method': 'mean'
            }
        )

        self.temporal = TemporalLSTM(
            input_dim=cfg.input_dim,
            hidden_dim=cfg.lstm_hidden_dim,
            num_layers=cfg.lstm_num_layers,
            output_dim=cfg.lstm_output_dim,
            dropout=cfg.lstm_dropout,
            bidirectional=cfg.lstm_bidirectional
        )

        self.pred_head = nn.Sequential(
            nn.Linear(cfg.gnn_output_dim + cfg.lstm_output_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

    def _temporal_context(self, hist_sequences: torch.Tensor) -> torch.Tensor:
        """
        hist_sequences: (1, seq_len, input_dim)
        return: (1, lstm_output_dim)
        """
        return self.temporal(hist_sequences)

    def forecast_one_day(self,
                         graph_data,
                         hist_sequences: torch.Tensor,
                         temporal_ctx: torch.Tensor) -> torch.Tensor:
        """
        graph_data: torch_geometric.data.Data
        hist_sequences: (1, seq_len, input_dim)
        temporal_ctx: (1, lstm_output_dim)
        returns: (N_nodes, 1) predictions
        """
        _, node_emb = self.gnn(graph_data, return_embeddings=True)  # (N_nodes, gnn_output_dim)
        ctx = temporal_ctx.repeat(node_emb.size(0), 1)              # (N_nodes, lstm_output_dim)
        fused = torch.cat([node_emb, ctx], dim=1)                   # (N_nodes, gnn_out + lstm_out)
        return self.pred_head(fused)                                # (N_nodes, 1)

    def forecast_sequence(self,
                          initial_graph_data,
                          initial_sequences: torch.Tensor,
                          forecast_days: int,
                          future_graph_data: List[Any]) -> Dict[str, Any]:
        """
        Roll forward day by day using provided future graph snapshots.
        """
        preds = []
        temporal_ctx = self._temporal_context(initial_sequences)
        for day_idx in range(forecast_days):
            g = future_graph_data[day_idx]
            p = self.forecast_one_day(g, initial_sequences, temporal_ctx)
            preds.append(p)
        return {'predictions': preds}

    def compute_loss(self,
                     preds: List[torch.Tensor],
                     targets: List[torch.Tensor]) -> torch.Tensor:
        loss = 0.0
        count = 0
        for p, t in zip(preds, targets):
            loss += F.mse_loss(p, t)
            count += 1
        return loss / max(count, 1)
