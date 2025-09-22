import torch
import torch.nn as nn


class TemporalLSTM(nn.Module):
    """
    Simple wrapper around an LSTM to encode a sequence of per-node/global features.

    Expected input: (batch=1, seq_len, input_dim)
    Output: (batch=1, output_dim)  # final hidden state mapped to output_dim
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 num_layers: int,
                 output_dim: int,
                 dropout: float = 0.0,
                 bidirectional: bool = False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )

        lstm_out_dim = hidden_dim * (2 if bidirectional else 1)
        self.fc = nn.Linear(lstm_out_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (1, seq_len, input_dim)
        Returns: (1, output_dim)
        """
        out, (h_n, c_n) = self.lstm(x)  # h_n: (num_layers*num_directions, batch, hidden_dim)
        # Take the last layer's hidden state(s)
        if self.bidirectional:
            h_last_fwd = h_n[-2, :, :]  # last forward
            h_last_bwd = h_n[-1, :, :]  # last backward
            h = torch.cat([h_last_fwd, h_last_bwd], dim=1)  # (1, hidden_dim*2)
        else:
            h = h_n[-1, :, :]  # (1, hidden_dim)
        return self.fc(h)  # (1, output_dim)
