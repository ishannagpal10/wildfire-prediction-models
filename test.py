import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')
sys.path.append(os.getcwd())

try:
    torch.set_float32_matmul_precision('high')
except Exception:
    pass

from utils import (
    WildfireDataLoader,
    GraphConstructor,
    set_random_seeds,
    get_device
)
from models.pipeline import WildfireForecastingPipeline, PipelineConfig
from metrics import (
    evaluate_forecast_sequence,
    EvaluationResults
)

# ------------- helpers -------------
def move_graph_to_device(graph, device):
    if hasattr(graph, 'x') and graph.x is not None:
        graph.x = graph.x.to(device, non_blocking=True)
    if hasattr(graph, 'edge_index') and graph.edge_index is not None:
        graph.edge_index = graph.edge_index.to(device, non_blocking=True)
    if hasattr(graph, 'edge_attr') and graph.edge_attr is not None:
        graph.edge_attr = graph.edge_attr.to(device, non_blocking=True)
    return graph


def compute_daily_metrics_local(daywise_preds: List[np.ndarray],
                                daywise_targets: List[np.ndarray]) -> List[Dict[str, float]]:
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    out = []
    for idx, (p, t) in enumerate(zip(daywise_preds, daywise_targets), start=1):
        if len(p) == 0:
            out.append({'day': idx, 'rmse': np.nan, 'mae': np.nan, 'r2': np.nan})
            continue
        rmse = float(np.sqrt(mean_squared_error(t, p)))
        mae = float(mean_absolute_error(t, p))
        r2 = float(r2_score(t, p)) if np.var(t) > 1e-12 else 0.0
        out.append({'day': idx, 'rmse': rmse, 'mae': mae, 'r2': r2})
    return out


# ------------- dataset -------------
class WildfireTestDataset(Dataset):
    def __init__(self,
                 data_loader: WildfireDataLoader,
                 graph_constructor: GraphConstructor,
                 base_dates: List[datetime],
                 parquet_path: str,
                 forecast_horizon: int = 31,
                 sequence_length: int = 7):
        self.data_loader = data_loader
        self.graph_constructor = graph_constructor
        self.base_dates = base_dates
        self.forecast_horizon = forecast_horizon
        self.sequence_length = sequence_length

        self.df = pd.read_parquet(parquet_path)
        if not pd.api.types.is_datetime64_any_dtype(self.df['date']):
            self.df['date'] = pd.to_datetime(self.df['date'])

        self.feature_cols = [c for c in self.df.columns if c not in ['date', 'FRP', 'latitude', 'longitude']]
        self._graph_cache: Dict[pd.Timestamp, Any] = {}

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
            future_dates = []
            for i in range(1, self.forecast_horizon + 1):
                f_date = base_date + timedelta(days=i)
                if (self.df['date'] == f_date).any():
                    future_dates.append(f_date)
            if len(future_dates) == 0:
                return None

            # historical
            hist_tensors = []
            for i in range(self.sequence_length, 0, -1):
                h_date = base_date - timedelta(days=i)
                hist_df = self.df[self.df['date'] == h_date]
                if len(hist_df) > 0:
                    arr = hist_df.iloc[0][self.feature_cols].values.astype(np.float32)
                    hist_tensors.append(torch.tensor(arr))

            if len(hist_tensors) == 0:
                feature_ct = len(self.feature_cols)
                initial_sequences = torch.zeros(1, self.sequence_length, feature_ct, dtype=torch.float32)
            else:
                while len(hist_tensors) < self.sequence_length:
                    hist_tensors.append(hist_tensors[-1].clone())
                initial_sequences = torch.stack(hist_tensors[-self.sequence_length:]).unsqueeze(0).float()

            future_graph_data, targets = [], []
            for f_date in future_dates:
                day_df = self.df[self.df['date'] == f_date]
                if len(day_df) == 0:
                    continue
                g = self._graph_for_date(day_df, f_date)
                future_graph_data.append(g)

                t = torch.tensor(day_df['FRP'].values, dtype=torch.float32).unsqueeze(1)
                targets.append(t)

            if len(future_graph_data) == 0:
                return None

            return {
                'base_date': base_date,
                'initial_sequences': initial_sequences,
                'future_graph_data': future_graph_data,
                'targets': targets
            }
        except Exception:
            return None


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return batch[0]


# ------------- tester -------------
class WildfireTester:
    def __init__(self,
                 model_path: str,
                 config: Dict[str, Any],
                 test_parquet: str,
                 sequence_length: int):
        self.device = get_device()
        set_random_seeds(config.get('random_seed', 42))

        self.config = config
        self.model_path = model_path
        self.test_parquet = test_parquet
        self.sequence_length = sequence_length

        self.data_loader = WildfireDataLoader()
        self.graph_constructor = GraphConstructor(self.data_loader)

        self._build_dataset_info()
        self.model = self._load_model()

    def _build_dataset_info(self):
        df = pd.read_parquet(self.test_parquet)
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])

        self.test_start = df['date'].min()
        self.test_end = df['date'].max()
        self.test_rows = len(df)
        print(f"Test split rows: {self.test_rows:,}")
        print(f"Test date span: {self.test_start} -> {self.test_end}")

        base_dates = sorted(df['date'].unique())
        if len(base_dates) > 1:
            base_dates = base_dates[:-1]
        self.base_dates = base_dates

        max_samples = self.config.get('max_test_samples', None)
        if max_samples and len(self.base_dates) > max_samples:
            step = max(1, len(self.base_dates) // max_samples)
            self.base_dates = self.base_dates[::step][:max_samples]
            print(f"Using subset of base dates for test: {len(self.base_dates)} (max_test_samples={max_samples})")
        else:
            print(f"Total base dates for test: {len(self.base_dates)}")

    def _load_model(self):
        default_params = self.config.get('default_params', {})
        input_dim = len(self.data_loader.all_features)

        model_conf = PipelineConfig(
            input_dim=input_dim,
            gnn_hidden_dims=default_params.get('gnn_hidden_dims', [24, 12]),
            gnn_output_dim=default_params.get('gnn_output_dim', 16),
            gnn_aggregation=default_params.get('gnn_aggregation', 'mean'),
            gnn_dropout=default_params.get('gnn_dropout', 0.2),
            lstm_hidden_dim=default_params.get('lstm_hidden_dim', 32),
            lstm_num_layers=default_params.get('lstm_num_layers', 1),
            lstm_output_dim=default_params.get('lstm_output_dim', 32),
            lstm_dropout=default_params.get('lstm_dropout', 0.2),
            lstm_bidirectional=default_params.get('lstm_bidirectional', False),
            sequence_length=default_params.get('sequence_length', self.sequence_length),
            forecast_horizon=self.config.get('forecast_horizon', 31),
            learning_rate=default_params.get('learning_rate', 1e-3),
            weight_decay=default_params.get('weight_decay', 1e-5)
        )

        model = WildfireForecastingPipeline(model_conf).to(self.device)
        state_dict = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.eval()
        print(f"Loaded model from {self.model_path}")
        return model

    def _loader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=1,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=False
        )

    def run(self,
            save_preds: Optional[str] = None,
            save_metrics: Optional[str] = None) -> EvaluationResults:

        dataset = WildfireTestDataset(
            data_loader=self.data_loader,
            graph_constructor=self.graph_constructor,
            base_dates=self.base_dates,
            parquet_path=self.test_parquet,
            forecast_horizon=self.config['forecast_horizon'],
            sequence_length=self.sequence_length
        )

        loader = self._loader(dataset, shuffle=False)

        all_preds: List[List[np.ndarray]] = []
        all_tgts:  List[List[np.ndarray]] = []

        if self.device.type == 'cuda':
            autocast = torch.cuda.amp.autocast
        else:
            from contextlib import nullcontext
            autocast = nullcontext

        with torch.no_grad():
            for batch in tqdm(loader, desc="Testing", leave=True):
                if batch is None:
                    continue
                try:
                    future_graph_data = batch['future_graph_data']
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

                    all_preds.append([p.cpu().numpy() for p in preds])
                    all_tgts.append([t.cpu().numpy() for t in targets])

                except RuntimeError as e:
                    if "CUDA out of memory" in str(e):
                        if self.device.type == 'cuda':
                            torch.cuda.empty_cache()
                        continue
                    else:
                        print(f"Error in test batch: {e}")
                        continue
                except Exception as e:
                    print(f"Error in test batch: {e}")
                    continue

        if not all_preds:
            print("No predictions produced. Check your test set or model.")
            return EvaluationResults(
                daily_metrics=[],
                summary_metrics={'mean_rmse': float('inf'), 'mean_mae': float('inf'), 'mean_r2': -float('inf')},
                forecast_horizon=self.config['forecast_horizon'],
                evaluation_date=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            )

        max_day = max(len(p) for p in all_preds)
        daywise_preds, daywise_targets = [], []
        for d in range(max_day):
            P, T = [], []
            for sample_p, sample_t in zip(all_preds, all_tgts):
                if d < len(sample_p):
                    P.extend(sample_p[d].flatten())
                    T.extend(sample_t[d].flatten())
            if P:
                daywise_preds.append(np.array(P))
                daywise_targets.append(np.array(T))

        results = evaluate_forecast_sequence(daywise_preds, daywise_targets)

        if not hasattr(results, 'daily_metrics') or not results.daily_metrics:
            daily_metrics = compute_daily_metrics_local(daywise_preds, daywise_targets)
            results = EvaluationResults(
                daily_metrics=daily_metrics,
                summary_metrics=results.summary_metrics,
                forecast_horizon=results.forecast_horizon,
                evaluation_date=results.evaluation_date
            )
        else:
            daily_metrics = results.daily_metrics

        # Print metrics
        print("\nPer-day test metrics:")
        best_rmse, best_mae, best_r2 = float('inf'), float('inf'), -float('inf')
        for m in daily_metrics:
            print(f"  Day {m['day']:2d}: RMSE={m['rmse']:.4f}  MAE={m['mae']:.4f}  R^2={m['r2']:.4f}")
            best_rmse = min(best_rmse, m['rmse'])
            best_mae  = min(best_mae,  m['mae'])
            best_r2   = max(best_r2,   m['r2'])

        print("\n=== Summary (averaged across days) ===")
        print(f"  RMSE={results.summary_metrics['mean_rmse']:.4f}  "
              f"MAE={results.summary_metrics['mean_mae']:.4f}  "
              f"R^2={results.summary_metrics['mean_r2']:.4f}")

        print("\n=== Best single-day scores ===")
        print(f"  Best RMSE: {best_rmse:.4f}")
        print(f"  Best MAE : {best_mae:.4f}")
        print(f"  Best R^2 : {best_r2:.4f}")

        if save_metrics:
            os.makedirs(os.path.dirname(save_metrics), exist_ok=True)
            pd.DataFrame(daily_metrics).to_csv(save_metrics, index=False)
            print(f"Saved per-day metrics to {save_metrics}")

        if save_preds:
            os.makedirs(os.path.dirname(save_preds), exist_ok=True)
            np.savez_compressed(
                save_preds,
                preds=all_preds,
                targets=all_tgts,
                daywise_preds=daywise_preds,
                daywise_targets=daywise_targets
            )
            print(f"Saved predictions & targets to {save_preds}.npz")

        return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained wildfire model on test set.")
    parser.add_argument('--model', default='checkpoints/best_model.pt', help='Path to best model weights (.pt)')
    parser.add_argument('--test_parquet', default='data/processed/test.parquet', help='Path to processed test parquet')
    parser.add_argument('--horizon', type=int, default=None, help='Override forecast horizon from config')
    parser.add_argument('--sequence_length', type=int, default=None, help='Override sequence length from config')
    parser.add_argument('--max_test_samples', type=int, default=None, help='Subsample test base dates')
    parser.add_argument('--save_metrics', default='results/test_metrics.csv', help='CSV path for metrics')
    parser.add_argument('--save_preds', default='', help='npz path prefix for preds/targets (omit to skip)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    print("Wildfire Forecasting - TEST MODE")
    print("=" * 60)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")

    config = {
        'forecast_horizon': 31,
        'random_seed': args.seed,
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
        'max_test_samples': args.max_test_samples
    }

    if args.horizon:
        config['forecast_horizon'] = args.horizon
    seq_len = args.sequence_length or config['default_params']['sequence_length']

    tester = WildfireTester(
        model_path=args.model,
        config=config,
        test_parquet=args.test_parquet,
        sequence_length=seq_len
    )

    tester.run(
        save_preds=args.save_preds if args.save_preds else None,
        save_metrics=args.save_metrics if args.save_metrics else None
    )

    print("\nDone.")


if __name__ == '__main__':
    main()
