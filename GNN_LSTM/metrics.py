from dataclasses import dataclass
from typing import List, Dict, Any
from datetime import datetime
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


@dataclass
class EvaluationResults:
    daily_metrics: List[Dict[str, float]]
    summary_metrics: Dict[str, float]
    forecast_horizon: int
    evaluation_date: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            'daily_metrics': self.daily_metrics,
            'summary_metrics': self.summary_metrics,
            'forecast_horizon': self.forecast_horizon,
            'evaluation_date': self.evaluation_date
        }


def time_aware_split(df,
                     train_end_date: str,
                     val_end_date: str):
        train_mask = df['date'] <= np.datetime64(train_end_date)
        val_mask = (df['date'] > np.datetime64(train_end_date)) & (df['date'] <= np.datetime64(val_end_date))
        test_mask = df['date'] > np.datetime64(val_end_date)
        return df[train_mask].copy(), df[val_mask].copy(), df[test_mask].copy()


def compute_daily_metrics(preds_list: List[np.ndarray],
                          tgts_list: List[np.ndarray]) -> List[Dict[str, float]]:
    out = []
    for i, (preds, trues) in enumerate(zip(preds_list, tgts_list), start=1):
        if len(preds) == 0:
            out.append({'day': i, 'rmse': np.nan, 'mae': np.nan, 'r2': np.nan})
            continue
        rmse = np.sqrt(mean_squared_error(trues, preds))
        mae = mean_absolute_error(trues, preds)
        if np.var(trues) < 1e-8:
            r2 = np.nan
        else:
            try:
                r2 = r2_score(trues, preds)
            except Exception:
                r2 = np.nan
        out.append({'day': i, 'rmse': rmse, 'mae': mae, 'r2': r2})
    return out


def evaluate_forecast_sequence(daywise_preds: List[np.ndarray],
                               daywise_targets: List[np.ndarray]) -> EvaluationResults:
    daily = compute_daily_metrics(daywise_preds, daywise_targets)
    rmse_list = [d['rmse'] for d in daily if not np.isnan(d['rmse'])]
    mae_list  = [d['mae'] for d in daily if not np.isnan(d['mae'])]
    r2_list   = [d['r2']  for d in daily if not np.isnan(d['r2'])]

    summary = {
        'mean_rmse': float(np.mean(rmse_list)) if rmse_list else float('inf'),
        'mean_mae':  float(np.mean(mae_list))  if mae_list  else float('inf'),
        'mean_r2':   float(np.mean(r2_list))   if r2_list   else float('-inf'),
    }

    return EvaluationResults(
        daily_metrics=daily,
        summary_metrics=summary,
        forecast_horizon=len(daily),
        evaluation_date=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    )
