#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Wildfire forecasting training script (revised):
 - Uses real forecast offsets instead of generic "Day 1"
 - Handles missing/irregular future days
 - Supports target log1p transform and Huber loss
 - Prints best RMSE/MAE/R^2 (summary + single-offset)
 - CLI flags for everything
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import argparse
import optuna
from optuna.samplers import TPESampler
import joblib
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')
sys.path.append(os.getcwd())

# Optional speed hint (PyTorch 2.x)
try:
    torch.set_float32_matmul_precision('medium')
except Exception:
    pass

# ----------------- Local imports -----------------
from utils import (
    WildfireDataLoader,
    GraphConstructor,
    set_random_seeds,
    get_device
)
from models.pipeline import WildfireForecastingPipeline, PipelineConfig

# We still import these to keep compatibility,
# but we will compute offset-based metrics manually.
from metrics import (
    time_aware_split,
    EvaluationResults
)

# ============================================================
# Utility helpers
# ============================================================
def optimize_for_gpu() -> bool:
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"   VRAM: {mem:.1f} GB")
        torch.backends.cudnn.benchmark = True
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'
        return True
    else:
        print("No GPU detected, using CPU")
        return False


def move_graph_to_device(graph, device):
    """Move torch_geometric Data to device (no pin_memory to avoid Win issues)."""
    if hasattr(graph, 'x') and graph.x is not None:
        graph.x = graph.x.to(device, non_blocking=True)
    if hasattr(graph, 'edge_index') and graph.edge_index is not None:
        graph.edge_index = graph.edge_index.to(device, non_blocking=True)
    if hasattr(graph, 'edge_attr') and graph.edge_attr is not None:
        graph.edge_attr = graph.edge_attr.to(device, non_blocking=True)
    return graph


def compute_regression_metrics(preds: np.ndarray, trues: np.ndarray) -> Tuple[float, float, float]:
    """Return (rmse, mae, r2) for 1-D arrays."""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    # Scrub NaN/inf values to prevent sklearn crashes
    preds = np.nan_to_num(preds, nan=0.0, posinf=1e6, neginf=0.0)
    trues = np.nan_to_num(trues, nan=0.0, posinf=1e6, neginf=0.0)
    rmse = float(np.sqrt(mean_squared_error(trues, preds)))
    mae  = float(mean_absolute_error(trues, preds))
    r2   = float(r2_score(trues, preds)) if np.var(trues) > 1e-12 else 0.0
    return rmse, mae, r2


def _safe_expm1(x: torch.Tensor) -> torch.Tensor:
    return torch.expm1(x)


def inverse_target_transform(t: torch.Tensor, use_log1p: bool) -> torch.Tensor:
    """Safely inverse the log1p transform to avoid infinity values."""
    if not use_log1p:
        return t
    # Clamp to avoid overflow in expm1 (exp(20) ≈ 485M, manageable)
    return torch.expm1(torch.clamp(t, max=20.0))


# ============================================================
# Dataset
# ============================================================
class WildfireDataset(Dataset):
    """
    For each base_date:
      - initial_sequences: (1, seq_len, input_dim)
      - future_graph_data: list of graph snapshots (<= horizon)
      - targets: list of (N_nodes, 1) FRP tensors per day
      - offsets: list of integers (days after base_date) for each element
    Missing days are skipped; we do NOT fill by default here.
    """

    def __init__(self,
                 data_loader: WildfireDataLoader,
                 graph_constructor: GraphConstructor,
                 base_dates: List[datetime],
                 split_name: str,
                 forecast_horizon: int = 31,
                 sequence_length: int = 7,
                 fill_missing: bool = False):
        self.data_loader = data_loader
        self.graph_constructor = graph_constructor
        self.base_dates = base_dates
        self.split_name = split_name
        self.forecast_horizon = forecast_horizon
        self.sequence_length = sequence_length
        self.fill_missing = fill_missing

        data_path = f"data/processed/{split_name}.parquet"
        self.df = pd.read_parquet(data_path)
        if not pd.api.types.is_datetime64_any_dtype(self.df['date']):
            self.df['date'] = pd.to_datetime(self.df['date'])

        self.feature_cols = [c for c in self.df.columns if c not in ['date', 'FRP']]
        self._graph_cache: Dict[pd.Timestamp, Any] = {}
        
        # Precompute date groups for faster lookups
        self.date_groups = {d: df for d, df in self.df.groupby('date')}
        
        # Coordinate tolerance for checking if nodes are the same (in degrees)
        self.coord_tolerance = 1e-6  # About 0.1 meters at equator

    def _coords_within_tolerance(self, lat1, lon1, lat2, lon2):
        """Check if two coordinate pairs are within tolerance (same node)"""
        return (abs(lat1 - lat2) <= self.coord_tolerance and 
                abs(lon1 - lon2) <= self.coord_tolerance)

    def _round_coords_for_sorting(self, df):
        """Round coordinates to avoid floating point precision issues while maintaining tolerance"""
        df = df.copy()
        # Round to 6 decimal places (~0.1m precision) for consistent sorting
        df['LAT_rounded'] = df['LAT'].round(6)
        df['LON_rounded'] = df['LON'].round(6)
        return df.sort_values(['LAT_rounded', 'LON_rounded'])

    def __len__(self):
        return len(self.base_dates)

    def _graph_for_date(self, day_df: pd.DataFrame, date_key: pd.Timestamp):
        if date_key in self._graph_cache:
            return self._graph_cache[date_key]
        g = self.graph_constructor.create_graph_data(day_df, include_target=True)
        self._graph_cache[date_key] = g
        return g

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        base_date = self.base_dates[idx]
        try:
            # ------ FUTURE PART ------
            real_future_dates = []
            for i in range(1, self.forecast_horizon + 1):
                f_date = base_date + timedelta(days=i)
                # only use days that truly exist (with rows) in this split
                day_df = self.df[self.df['date'] == f_date]
                if len(day_df) > 0:
                    real_future_dates.append(f_date)

            if len(real_future_dates) == 0:
                return None

            # Skip samples with too few future days to ensure diverse offsets
            if len(real_future_dates) < 3:
                return None

            # Make sure they are sorted by actual date
            real_future_dates = sorted(real_future_dates)

            future_graph_data, targets, offsets = [], [], []
            for f_date in real_future_dates:
                # Use precomputed date groups for faster lookup
                if f_date not in self.date_groups:
                    continue
                day_df = self.date_groups[f_date]
                # skip any empty
                if len(day_df) == 0:
                    continue
                    
                # Sort by LAT, LON to guarantee stable order across days (same nodes in same order)
                # Use tolerance-based rounding to avoid floating point precision issues
                day_df = self._round_coords_for_sorting(day_df)
                    
                g = self._graph_for_date(day_df, f_date)
                future_graph_data.append(g)

                # N_nodes x 1
                t = torch.tensor(day_df['FRP'].values, dtype=torch.float32).unsqueeze(1)
                targets.append(t)

                # real offset (days after base_date)
                offsets.append((f_date - base_date).days)

            # Debug print for first few samples only
            if idx < 3:  # Reduced from 5 to minimize spam
                print(f"[DATA DEBUG] base_date={base_date.strftime('%Y-%m-%d')} -> offsets={offsets}")

            # Optionally fill missing offsets with dummy if requested
            if self.fill_missing and len(future_graph_data) < self.forecast_horizon:
                missing_needed = self.forecast_horizon - len(future_graph_data)
                if len(future_graph_data) > 0:
                    last_g = future_graph_data[-1]
                    last_t = targets[-1]
                else:
                    # If completely missing, we can't fill properly
                    return None
                for m in range(missing_needed):
                    future_graph_data.append(last_g)
                    targets.append(torch.zeros_like(last_t))
                    # fake offset: last offset + 1, then +2, etc.
                    offsets.append(offsets[-1] + 1)

            # Build historical sequences
            hist_tensors = []
            for i in range(self.sequence_length, 0, -1):
                h_date = base_date - timedelta(days=i)
                # Use precomputed date groups for faster lookup
                if h_date in self.date_groups:
                    hist_df = self.date_groups[h_date]
                    if len(hist_df) > 0:
                        # Sort by LAT, LON to guarantee stable order (same nodes in same order)
                        # Use tolerance-based rounding to avoid floating point precision issues
                        hist_df = self._round_coords_for_sorting(hist_df)
                        arr = hist_df.iloc[0][self.feature_cols].values.astype(np.float32)
                        hist_tensors.append(torch.tensor(arr))

            if len(hist_tensors) == 0:
                feat_ct = len(self.feature_cols)
                initial_sequences = torch.zeros(1, self.sequence_length, feat_ct, dtype=torch.float32)
            else:
                while len(hist_tensors) < self.sequence_length:
                    hist_tensors.append(hist_tensors[-1].clone())
                initial_sequences = torch.stack(hist_tensors[-self.sequence_length:]).unsqueeze(0).float()

            return {
                'base_date': base_date,
                'initial_sequences': initial_sequences,
                'future_graph_data': future_graph_data,
                'targets': targets,
                'offsets': offsets  # list[int]
            }
        except Exception:
            return None


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return batch[0]


# ============================================================
# Trainer
# ============================================================
class WildfireTrainer:
    def __init__(self,
                 config: Dict[str, Any],
                 val_every: int = 1,
                 quiet: bool = False):
        self.config = config
        self.device = get_device()
        set_random_seeds(config.get('random_seed', 42))

        self.data_loader = WildfireDataLoader()
        self.graph_constructor = GraphConstructor(self.data_loader)

        self.quiet = quiet
        self.val_every = max(1, val_every)

        # DataLoader settings for speed optimization
        self.num_workers = 4 if self.device.type == 'cuda' else 0
        self.pin_memory = self.device.type == 'cuda'

        # Loss & transform settings
        self.target_log1p = bool(config.get('target_log1p', False))
        self.loss_type = config.get('loss_type', 'mse')
        self.huber_delta = float(config.get('huber_delta', 1.0))

        self.setup_splits()

        self.model: Optional[WildfireForecastingPipeline] = None
        self.optimizer = None
        self.scheduler = None

        self.train_losses: List[float] = []
        self.val_results_raw: List[Dict[str, Any]] = []  # store raw dict for debugging if needed

        # Track best metrics across epochs
        self.best_by_metric = {
            'rmse_sum': (float('inf'), -1),  # minimize summary
            'mae_sum':  (float('inf'), -1),
            'r2_sum':   (-float('inf'), -1), # maximize summary
            'rmse_off': (float('inf'), -1, -1),  # value, epoch, offset
            'mae_off':  (float('inf'), -1, -1),
            'r2_off':   (-float('inf'), -1, -1),
            'train_loss': (float('inf'), -1)
        }
        self.best_model_path: Optional[str] = None
        self.patience_counter = 0

    # ------------------ Splits ------------------
    def setup_splits(self):
        print("Using pre-processed train/val splits from data/processed/")

        # Load the already processed train and validation sets
        train_df = pd.read_parquet("data/processed/train.parquet")
        val_df = pd.read_parquet("data/processed/val.parquet")

        # Ensure date is datetime
        if not pd.api.types.is_datetime64_any_dtype(train_df['date']):
            train_df['date'] = pd.to_datetime(train_df['date'])
        if not pd.api.types.is_datetime64_any_dtype(val_df['date']):
            val_df['date'] = pd.to_datetime(val_df['date'])

        # Report spans & sizes
        tr_min, tr_max = train_df['date'].min(), train_df['date'].max()
        va_min, va_max = val_df['date'].min(), val_df['date'].max()
        print(f"Train: {tr_min} -> {tr_max}  ({len(train_df)} rows)")
        print(f"Val:   {va_min} -> {va_max}  ({len(val_df)} rows)")

        self.train_base_dates = self._make_base_dates(train_df)
        self.val_base_dates = self._make_base_dates(val_df)

        print(f"Train base dates: {len(self.train_base_dates)}")
        print(f"Val base dates:   {len(self.val_base_dates)}")

    def _make_base_dates(self, df: pd.DataFrame) -> List[datetime]:
        dates = sorted(df['date'].unique())
        base_dates = dates[:-1]  # exclude last, can't forecast past it
        max_samples = self.config.get('max_training_samples', None)
        if max_samples is not None and max_samples > 0 and len(base_dates) > max_samples:
            # Use random sampling instead of systematic stepping to avoid seasonal bias
            rng = np.random.default_rng(self.config.get('random_seed', 42))
            base_dates = rng.choice(base_dates, size=max_samples, replace=False)
            base_dates = sorted(base_dates)
        return base_dates

    # ------------------ Model ------------------
    def create_model(self, trial_params: Optional[Dict] = None) -> WildfireForecastingPipeline:
        if trial_params is None:
            trial_params = self.config.get('default_params', {})
        
        # Get the actual feature count from the processed dataset
        sample_df = pd.read_parquet("data/processed/train.parquet")
        feature_cols = [c for c in sample_df.columns if c not in ['date', 'FRP']]
        input_dim = len(feature_cols)
        
        print(f"Model input_dim set to {input_dim} (from processed dataset)")
        
        model_conf = PipelineConfig(
            input_dim=input_dim,
            gnn_hidden_dims=trial_params.get('gnn_hidden_dims', [24, 12]),
            gnn_output_dim=trial_params.get('gnn_output_dim', 16),
            gnn_aggregation=trial_params.get('gnn_aggregation', 'mean'),
            gnn_dropout=trial_params.get('gnn_dropout', 0.2),
            lstm_hidden_dim=trial_params.get('lstm_hidden_dim', 32),
            lstm_num_layers=trial_params.get('lstm_num_layers', 1),
            lstm_output_dim=trial_params.get('lstm_output_dim', 32),
            lstm_dropout=trial_params.get('lstm_dropout', 0.2),
            lstm_bidirectional=trial_params.get('lstm_bidirectional', False),
            sequence_length=trial_params.get('sequence_length', 3),
            forecast_horizon=self.config['forecast_horizon'],
            learning_rate=trial_params.get('learning_rate', 0.001),
            weight_decay=trial_params.get('weight_decay', 1e-5)
        )
        model = WildfireForecastingPipeline(model_conf).to(self.device)
        print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
        return model

    def _loader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=1,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=(self.num_workers > 0),
            prefetch_factor=2 if self.num_workers > 0 else None
        )

    # ------------------ Loss / Transform ------------------
    def _apply_transform(self, t: torch.Tensor) -> torch.Tensor:
        if self.target_log1p:
            return torch.log1p(torch.clamp(t, min=0))
        return t

    def _compute_loss(self, preds: List[torch.Tensor], targets: List[torch.Tensor]) -> torch.Tensor:
        loss_total = 0.0
        for p, t in zip(preds, targets):
            # transform both to same space
            p_tr = self._apply_transform(p)
            t_tr = self._apply_transform(t)
            if self.loss_type == 'huber':
                loss_total += F.huber_loss(p_tr, t_tr, delta=self.huber_delta, reduction='mean')
            else:
                loss_total += F.mse_loss(p_tr, t_tr, reduction='mean')
        return loss_total / max(1, len(preds))

    # ------------------ Train/Val loops ------------------
    def train_epoch(self, dataset: WildfireDataset, epoch: int) -> float:
        self.model.train()
        total_loss, batches = 0.0, 0
        scaler = torch.cuda.amp.GradScaler() if self.device.type == 'cuda' else None
        loader = self._loader(dataset, shuffle=True)

        pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=False, disable=self.quiet, 
                   mininterval=5.0, miniters=50)

        if self.device.type == 'cuda':
            autocast = torch.cuda.amp.autocast
        else:
            from contextlib import nullcontext
            autocast = nullcontext

        for i, batch in enumerate(pbar):
            if batch is None:
                continue
            try:
                future_graph_data = batch['future_graph_data']
                initial_sequences = batch['initial_sequences'].to(self.device, non_blocking=True)
                targets = [t.to(self.device, non_blocking=True) for t in batch['targets']]

                for g in future_graph_data:
                    move_graph_to_device(g, self.device)

                self.optimizer.zero_grad(set_to_none=True)
                with autocast():
                    out = self.model.forecast_sequence(
                        initial_graph_data=future_graph_data[0],
                        initial_sequences=initial_sequences,
                        forecast_days=len(future_graph_data),
                        future_graph_data=future_graph_data
                    )
                    preds = out['predictions']
                    
                # --- Ensure preds is a list of Tensors, length == len(future_graph_data) ---
                if isinstance(preds, torch.Tensor):
                    # common shapes you might see:
                    #   (forecast_days, N_nodes, 1) or (N_nodes, forecast_days) etc.
                    if preds.dim() == 3 and preds.shape[0] == len(future_graph_data):
                        preds = [preds[d] for d in range(preds.shape[0])]
                    elif preds.dim() == 3 and preds.shape[1] == len(future_graph_data):
                        preds = [preds[:, d] for d in range(preds.shape[1])]
                    elif preds.dim() == 2 and preds.shape[1] == len(future_graph_data):
                        preds = [preds[:, d:d+1] for d in range(preds.shape[1])]
                    else:
                        # safest fallback: unbind along 0
                        preds = list(torch.unbind(preds, dim=0))

                # Sanity check for training
                assert len(preds) == len(future_graph_data), \
                    f"Train pred length {len(preds)} != future_graph_data {len(future_graph_data)}"
                assert len(preds) == len(targets), \
                    f"Train pred length {len(preds)} != targets {len(targets)}"
                
                loss = self._compute_loss(preds, targets)
                if scaler:
                    scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

                total_loss += loss.item()
                batches += 1
                if not self.quiet:
                    pbar.set_postfix(loss=f"{loss.item():.4f}")

                # Removed frequent CUDA cache clearing for speed

            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()
                    continue
                else:
                    print(f"Error in training batch {i}: {e}")
                    continue
            except Exception as e:
                print(f"Error in training batch {i}: {e}")
                continue

        avg_loss = total_loss / max(batches, 1)
        if avg_loss < self.best_by_metric['train_loss'][0]:
            self.best_by_metric['train_loss'] = (avg_loss, epoch)
        return avg_loss

    def validate(self, dataset: WildfireDataset) -> Dict[str, Any]:
        """
        Returns a dict with:
          {
            'offset_metrics': List[{'offset': int, 'rmse':..., 'mae':..., 'r2':...}, ...],
            'summary_metrics': {'mean_rmse':..., 'mean_mae':..., 'mean_r2':...}
          }
        """
        self.model.eval()
        loader = self._loader(dataset, shuffle=False)

        # We'll store predictions by offset:
        # offset_dict[offset] = { 'preds': [np arrays], 'tgts': [np arrays] }
        offset_dict: Dict[int, Dict[str, List[np.ndarray]]] = {}

        if self.device.type == 'cuda':
            autocast = torch.cuda.amp.autocast
        else:
            from contextlib import nullcontext
            autocast = nullcontext

        with torch.inference_mode():
            for i, batch in enumerate(loader):
                if batch is None:
                    continue
                try:
                    future_graph_data = batch['future_graph_data']
                    offsets = batch['offsets']
                    initial_sequences = batch['initial_sequences'].to(self.device, non_blocking=True)
                    targets = [t.to(self.device, non_blocking=True) for t in batch['targets']]

                    for g in future_graph_data:
                        move_graph_to_device(g, self.device)

                    with autocast():
                        out = self.model.forecast_sequence(
                            initial_graph_data=future_graph_data[0],
                            initial_sequences=initial_sequences,
                            forecast_days=len(future_graph_data),
                            future_graph_data=future_graph_data
                        )
                        preds = out['predictions']

                    # --- Ensure preds is a list of Tensors, length == len(future_graph_data) ---
                    if isinstance(preds, torch.Tensor):
                        # common shapes you might see:
                        #   (forecast_days, N_nodes, 1) or (N_nodes, forecast_days) etc.
                        if preds.dim() == 3 and preds.shape[0] == len(future_graph_data):
                            preds = [preds[d] for d in range(preds.shape[0])]
                        elif preds.dim() == 3 and preds.shape[1] == len(future_graph_data):
                            preds = [preds[:, d] for d in range(preds.shape[1])]
                        elif preds.dim() == 2 and preds.shape[1] == len(future_graph_data):
                            preds = [preds[:, d:d+1] for d in range(preds.shape[1])]
                        else:
                            # safest fallback: unbind along 0
                            preds = list(torch.unbind(preds, dim=0))

                    # Sanity check
                    assert len(preds) == len(future_graph_data), \
                        f"Pred length {len(preds)} != future_graph_data {len(future_graph_data)}"
                    assert len(preds) == len(targets), \
                        f"Pred length {len(preds)} != targets {len(targets)}"

                    # Reduced debug spam - only show validation progress occasionally
                    if i < 3:  # Reduced from showing all to first 3 only
                        print(f"[VAL DEBUG] base_date={batch['base_date']} offsets={offsets} preds={len(preds)}")
                    
                    # Ensure exact alignment - no truncation
                    assert len(preds) == len(offsets) == len(targets), \
                        f"Mismatch: preds={len(preds)}, offsets={len(offsets)}, targets={len(targets)}"

                    for j in range(len(preds)):
                        p_np = preds[j].detach().cpu().numpy().flatten()
                        t_np = targets[j].detach().cpu().numpy().flatten()
                        off = offsets[j]

                        if off not in offset_dict:
                            offset_dict[off] = {'preds': [], 'tgts': []}
                        offset_dict[off]['preds'].append(p_np)
                        offset_dict[off]['tgts'].append(t_np)

                    # Removed frequent CUDA cache clearing for speed

                except RuntimeError as e:
                    if "CUDA out of memory" in str(e):
                        if self.device.type == 'cuda':
                            torch.cuda.empty_cache()
                        continue
                    else:
                        print(f"Error in validation batch {i}: {e}")
                        continue
                except Exception as e:
                    print(f"Error in validation batch {i}: {e}")
                    continue

        # Compute metrics for each offset
        if not offset_dict:
            # No predictions at all?
            return {
                'offset_metrics': [],
                'summary_metrics': {
                    'mean_rmse': float('inf'),
                    'mean_mae': float('inf'),
                    'mean_r2':  -float('inf')
                }
            }

        offset_metrics = []
        for off in sorted(offset_dict.keys()):
            preds = np.concatenate(offset_dict[off]['preds'])
            tgts  = np.concatenate(offset_dict[off]['tgts'])
            
            # Apply inverse transform safely to avoid infinity values
            if self.target_log1p:
                # Convert back to torch for safe inverse transform
                preds_torch = torch.from_numpy(preds)
                tgts_torch = torch.from_numpy(tgts)
                preds_torch = inverse_target_transform(preds_torch, True)
                tgts_torch = inverse_target_transform(tgts_torch, True)
                preds = preds_torch.numpy()
                tgts = tgts_torch.numpy()
            
            rmse, mae, r2 = compute_regression_metrics(preds, tgts)
            offset_metrics.append({'offset': off, 'rmse': rmse, 'mae': mae, 'r2': r2})

        # Summary = mean across offsets
        mean_rmse = float(np.mean([m['rmse'] for m in offset_metrics]))
        mean_mae  = float(np.mean([m['mae']  for m in offset_metrics]))
        mean_r2   = float(np.mean([m['r2']   for m in offset_metrics]))

        if not self.quiet:
            print("\nPer-offset validation metrics (offset = days after base_date):")
            for m in offset_metrics:
                print(f"  Offset {m['offset']:2d}d: RMSE={m['rmse']:.4f}  MAE={m['mae']:.4f}  R^2={m['r2']:.4f}")

        # Track best metrics
        epoch = len(self.train_losses)
        if mean_rmse < self.best_by_metric['rmse_sum'][0]:
            self.best_by_metric['rmse_sum'] = (mean_rmse, epoch)
        if mean_mae < self.best_by_metric['mae_sum'][0]:
            self.best_by_metric['mae_sum'] = (mean_mae, epoch)
        if mean_r2 > self.best_by_metric['r2_sum'][0]:
            self.best_by_metric['r2_sum'] = (mean_r2, epoch)

        for m in offset_metrics:
            off = m['offset']
            if m['rmse'] < self.best_by_metric['rmse_off'][0]:
                self.best_by_metric['rmse_off'] = (m['rmse'], epoch, off)
            if m['mae'] < self.best_by_metric['mae_off'][0]:
                self.best_by_metric['mae_off']  = (m['mae'],  epoch, off)
            if m['r2']  > self.best_by_metric['r2_off'][0]:
                self.best_by_metric['r2_off']   = (m['r2'],   epoch, off)

        return {
            'offset_metrics': offset_metrics,
            'summary_metrics': {'mean_rmse': mean_rmse, 'mean_mae': mean_mae, 'mean_r2': mean_r2}
        }

    def train(self, trial_params: Optional[Dict] = None) -> Dict[str, float]:
        print("Starting training...")
        self.model = self.create_model(trial_params)

        # Optional torch.compile for speed (PyTorch 2.x)
        if hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model)
                print("Model compiled with torch.compile for optimization")
            except Exception as e:
                print(f"torch.compile failed, continuing without: {e}")

        lr = trial_params.get('learning_rate', 0.001) if trial_params else 0.001
        wd = trial_params.get('weight_decay', 1e-5)     if trial_params else 1e-5
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.5, patience=5)

        train_ds = WildfireDataset(
            self.data_loader,
            self.graph_constructor,
            self.train_base_dates,
            split_name='train',
            forecast_horizon=self.config['forecast_horizon'],
            sequence_length=self.config['default_params']['sequence_length'],
            fill_missing=False
        )
        val_ds = WildfireDataset(
            self.data_loader,
            self.graph_constructor,
            self.val_base_dates,
            split_name='val',
            forecast_horizon=self.config['forecast_horizon'],
            sequence_length=self.config['default_params']['sequence_length'],
            fill_missing=False
        )

        for epoch in range(1, self.config['num_epochs'] + 1):
            train_loss = self.train_epoch(train_ds, epoch)
            self.train_losses.append(train_loss)
            print(f"\nEpoch {epoch} - Train Loss: {train_loss:.6f}")

            do_val = ((epoch % self.val_every) == 0) or (epoch == self.config['num_epochs'])
            if do_val:
                val_res = self.validate(val_ds)  # dict
                self.val_results_raw.append(val_res)

                rmse = val_res['summary_metrics']['mean_rmse']
                # Early stopping based on RMSE summary
                if rmse < self.best_by_metric['rmse_sum'][0]:
                    self.save_checkpoint(epoch, val_res, is_best=True)
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.config.get('patience', 10):
                        print("Early stopping (no improvement)")
                        break

                self.scheduler.step(rmse)

            # Clear CUDA cache once per epoch instead of per batch
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

        # Print best metrics
        print("\nTraining complete")
        print("===== Best Validation Summary Metrics (mean across offsets) =====")
        print(f"  Best RMSE: {self.best_by_metric['rmse_sum'][0]:.4f} at epoch {self.best_by_metric['rmse_sum'][1]}")
        print(f"  Best MAE : {self.best_by_metric['mae_sum'][0]:.4f} at epoch {self.best_by_metric['mae_sum'][1]}")
        print(f"  Best R^2 : {self.best_by_metric['r2_sum'][0]:.4f} at epoch {self.best_by_metric['r2_sum'][1]}")

        print("\n===== Best Single-Offset Validation Metrics =====")
        rmse_val, rmse_ep, rmse_off = self.best_by_metric['rmse_off']
        mae_val,  mae_ep,  mae_off  = self.best_by_metric['mae_off']
        r2_val,   r2_ep,   r2_off   = self.best_by_metric['r2_off']
        print(f"  Best RMSE@offset: {rmse_val:.4f} (epoch {rmse_ep}, offset {rmse_off}d)")
        print(f"  Best MAE @offset: {mae_val:.4f} (epoch {mae_ep}, offset {mae_off}d)")
        print(f"  Best R^2 @offset: {r2_val:.4f} (epoch {r2_ep}, offset {r2_off}d)")

        print("\n===== Best Train Loss =====")
        print(f"  {self.best_by_metric['train_loss'][0]:.6f} at epoch {self.best_by_metric['train_loss'][1]}")

        if self.best_model_path:
            print(f"\nBest (RMSE summary) model saved at: {self.best_model_path}")

        return {
            'best_rmse': self.best_by_metric['rmse_sum'][0],
            'best_mae':  self.best_by_metric['mae_sum'][0],
            'best_r2':   self.best_by_metric['r2_sum'][0]
        }

    # ------------------ Checkpoint ------------------
    def save_checkpoint(self, epoch: int, val_results_dict: Dict[str, Any], is_best: bool = False):
        ckpt_dir = self.config.get('checkpoint_dir', 'checkpoints')
        os.makedirs(ckpt_dir, exist_ok=True)
        path = os.path.join(ckpt_dir, f'checkpoint_epoch_{epoch}.pt')

        # We save val_results_dict as-is
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_results': val_results_dict,
            'config': self.config
        }, path)

        if is_best:
            best_path = os.path.join(ckpt_dir, 'best_model.pt')
            torch.save(self.model.state_dict(), best_path)
            self.best_model_path = best_path
            print(f"Saved BEST (RMSE summary) model to {best_path}")


# ============================================================
# Optuna Objective
# ============================================================
def objective(trial):
    params = {
        'gnn_hidden_dims': [
            trial.suggest_int('gnn_dim1', 24, 96, step=24),
            trial.suggest_int('gnn_dim2', 12, 48, step=12)
        ],
        'gnn_output_dim': trial.suggest_int('gnn_output_dim', 16, 48, step=16),
        'gnn_aggregation': trial.suggest_categorical('gnn_aggregation', ['mean', 'max', 'add']),
        'gnn_dropout': trial.suggest_float('gnn_dropout', 0.1, 0.4),
        'lstm_hidden_dim': trial.suggest_int('lstm_hidden_dim', 24, 96, step=24),
        'lstm_num_layers': trial.suggest_int('lstm_num_layers', 1, 2),
        'lstm_output_dim': trial.suggest_int('lstm_output_dim', 16, 64, step=16),
        'lstm_dropout': trial.suggest_float('lstm_dropout', 0.1, 0.4),
        'sequence_length': trial.suggest_int('sequence_length', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 5e-3, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
    }

    cfg = {
        'train_end_date': '2019-12-19',
        'val_end_date':   '2020-07-10',
        'forecast_horizon': 24,
        'num_epochs': 11,
        'max_training_samples': 45,
        'patience': 4,
        'checkpoint_dir': f'optuna_trial_{trial.number}',
        'random_seed': 42,
        'default_params': params,
        'target_log1p': False,
        'loss_type': 'mse',
        'huber_delta': 1.0
    }

    try:
        trainer = WildfireTrainer(cfg, val_every=1, quiet=True)
        best_metrics = trainer.train(params)
        best_rmse = best_metrics['best_rmse']
        if best_rmse == float('inf'):
            import shutil
            if os.path.exists(cfg['checkpoint_dir']):
                shutil.rmtree(cfg['checkpoint_dir'])
        return best_rmse
    except Exception as e:
        print(f"Trial {trial.number} error: {e}")
        return float('inf')


def run_bayes(n_trials=20):
    print(f"Optuna: {n_trials} trials")
    study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)
    print("Optimization done")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best RMSE: {study.best_value:.4f}")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    joblib.dump(study, 'optuna_study.pkl')
    return study


# ============================================================
# Main
# ============================================================
def main():
    print("Wildfire Forecasting Training")
    print("=" * 60)
    gpu = optimize_for_gpu()

    config = {
        'forecast_horizon': 24,              # 3/4 of 31
        'num_epochs': 30,                    # 3/4 of 40
        'max_training_samples': 300 if gpu else 90,
        'patience': 8,
        'checkpoint_dir': 'checkpoints',
        'random_seed': 42,
        'default_params': {
            'gnn_hidden_dims': [24, 12],
            'gnn_output_dim': 16,
            'gnn_aggregation': 'mean',
            'gnn_dropout': 0.2,
            'lstm_hidden_dim': 32,
            'lstm_num_layers': 1,
            'lstm_output_dim': 32,
            'lstm_dropout': 0.2,
            'lstm_bidirectional': False,
            'sequence_length': 3,
            'learning_rate': 1e-3,
            'weight_decay': 1e-5
        },
        'target_log1p': True,
        'loss_type': 'huber',
        'huber_delta': 2.0
    }

    trainer = WildfireTrainer(config, val_every=1, quiet=False)
    best_metrics = trainer.train()
    print("\nFinal Best Validation (summary): "
          f"RMSE={best_metrics['best_rmse']:.4f}, "
          f"MAE={best_metrics['best_mae']:.4f}, "
          f"R^2={best_metrics['best_r2']:.4f}")


if __name__ == "__main__":
    main()
