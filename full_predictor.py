import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import optuna
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import torch

# Load data and preprocessing (unchanged)
reader = pd.read_csv('C:/Users/rt2sa/Downloads/stratified_20160110_20201231_3_days_excl_v3_2016_all_neighbors.csv', chunksize=10000, low_memory=False)
df = pd.concat(reader, ignore_index=True)
df = df.sample(frac=0.1, random_state=42)
df = df.loc[:, ~df.columns.duplicated()]

df.columns = df.columns.str.strip()

df = df.drop_duplicates()

zeros = ['V','V_1_days_ago','V_2_days_ago','U','U_1_days_ago','U_2_days_ago','T','T_1_days_ago','T_2_days_ago',
               'LAT','LON','VHI_AVE','VHI_AVE_1_days_ago','VHI_AVE_2_days_ago']

for col in df.columns:
    if not(col in zeros):
        df = df[df[col] >= 0]

if 'FRP' in df.columns:
    cols = [col for col in df.columns if col != 'FRP'] + ['FRP']
    df = df[cols]

if 'Land Use' in df.columns:
    df['Land Use'] = df['Land Use'].astype(str)  # Ensure string type
    df = pd.get_dummies(df, columns=['Land Use'], prefix='Land Use')



numeric_cols = df.select_dtypes(include=['number']).columns
cols_to_scale = [col for col in numeric_cols if col not in ['date', 'FRP']]

# Separate features and target
X = df.drop(columns=['FRP'])
y = df['FRP'].values.reshape(-1, 1)

# Use two separate scalers
scaler_all = StandardScaler()
scaler_frp = StandardScaler()

if cols_to_scale:
    X_scaled = scaler_all.fit_transform(X)
    y_scaled = scaler_frp.fit_transform(y)
else:
    X_scaled = X.values
    y_scaled = y

col_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, random_state=42)
X_train, X_gpr, y_train, y_gpr = train_test_split(X_train, y_train, random_state=42)

y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
y_valid = y_valid.reshape(-1, 1)
y_gpr = y_gpr.reshape(-1, 1)

# Scoring functions (unchanged)
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))
def r2(y_true, y_pred):
    return r2_score(y_true, y_pred)
def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

import optuna.visualization.matplotlib as optuna_mpl
import matplotlib.pyplot as plt

def bayes_optim(scorer, direction):
    def objective(trial):
        global X_train, y_train, X_test, y_test
        n_d = trial.suggest_int('n_d', 32, 128, step=4)
        n_a = trial.suggest_int('n_a', 32, 128, step=4)
        n_steps = trial.suggest_int('n_steps', 2, 5)
        gamma = trial.suggest_float('gamma', 0.01, 2.0)
        lambda_sparse = trial.suggest_float('lambda_sparse', 0.0, 1.0)
        lr= trial.suggest_float('lr', 1e-4, 1e-1, log=True)

        model = TabNetRegressor(
            n_d=n_d,
            n_a=n_a,
            n_steps=n_steps,
            gamma=gamma,
            lambda_sparse=lambda_sparse,
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=lr),
            seed=42,
            verbose=1,
        )

        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], patience=5, max_epochs=10, batch_size=1024, virtual_batch_size=128, num_workers=6)

        y_pred = model.predict(X_valid)

        return scorer(y_valid, y_pred)

    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=30)
    optuna_mpl.plot_optimization_history(study)
    optuna_mpl.plot_param_importances(study)
    plt.show()

    best_params_ = study.best_params
    best_params = best_params_
    best_model = TabNetRegressor(n_d=best_params['n_d'], n_a=best_params['n_a'], n_steps=best_params['n_steps'], gamma=best_params['gamma'], lambda_sparse=best_params['lambda_sparse'], optimizer_fn=torch.optim.Adam, optimizer_params=dict(lr=best_params['lr']), seed=42, verbose=1)
    best_model.fit(X_train, y_train, patience=10, max_epochs=30, batch_size=1024, virtual_batch_size=128, num_workers=6)
    y_pred = best_model.predict(X_test)
    scores = {
        'rmse': rmse(y_test, y_pred),
        'mae': mae(y_test, y_pred),
        'r2': r2(y_test, y_pred)
    }
    return best_model,best_params,scores

best_model, best_params, scores = bayes_optim(r2, 'maximize')
print("Optimized TabNet Results")
print(scores)


from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RationalQuadratic
import numpy as np

# Update lagged features (unchanged)
def update_X_base(previous_X, predicted_frp, col_names):
    updated_X = previous_X.copy()
    col_index = {name: idx for idx, name in enumerate(col_names)}

    # Update FRP lags
    frp_lags = ['FRP_1_days_ago', 'FRP_2_days_ago']
    updated_X[0, col_index[frp_lags[1]]] = updated_X[0, col_index[frp_lags[0]]]
    updated_X[0, col_index[frp_lags[0]]] = predicted_frp

    # Update other variable lags
    lag_vars = ['V', 'U', 'RH', 'T', 'RAIN', 'VHI_AVE']
    for var in lag_vars:
        lag0 = var
        lag1 = f"{var}_1_days_ago"
        lag2 = f"{var}_2_days_ago"
        if all(k in col_index for k in [lag0, lag1, lag2]):
            updated_X[0, col_index[lag2]] = updated_X[0, col_index[lag1]]
            updated_X[0, col_index[lag1]] = updated_X[0, col_index[lag0]]

    return updated_X

from sklearn.gaussian_process.kernels import Matern, RationalQuadratic

# Initialize training and forecasting sets
X_curr = X_train.copy()
y_curr = y_train.copy()
X_base = X_test[-1].reshape(1, -1)

def gper(X_gpr, y_gpr, model):
    y_pred_gpr = model.predict(X_gpr).ravel()
    y_true_gpr = y_gpr.ravel()
    residuals = y_true_gpr - y_pred_gpr

    kernel = Matern(length_scale=0.01,
                    nu=0.01,
                    length_scale_bounds=(1e-3, 10)) + RationalQuadratic(length_scale=0.01,
                                                                         length_scale_bounds=(1e-3, 10),
                                                                         alpha=0.1,
                                                                         alpha_bounds=(1e-2, 10))
    
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=1, normalize_y=True)
    gpr.fit(X_gpr, residuals)

    

    return gpr

def adjuster(model, gpr, X_test):
    residual_preds = gpr.predict(X_test).ravel()
    y_pred_test = model.predict(X_test).ravel()
    adj_preds = y_pred_test + 0.8*residual_preds
    adj_preds=adj_preds.ravel()
    
    return adj_preds

X_base_scaled = scaler_all.transform(X_base).astype(np.float32)
X_gpr_scaled = scaler_all.transform(X_gpr).astype(np.float32)
y_gpr_scaled = scaler_frp.transform(y_gpr).ravel()

# Sample a small subset for residual fitting
sample_fraction = 0.05 
sample_idx = np.random.choice(X_gpr_scaled.shape[0], 
                              int(sample_fraction * X_gpr_scaled.shape[0]), 
                              replace=False)
X_gpr_sampled = X_gpr_scaled[sample_idx]
y_gpr_sampled = y_gpr_scaled[sample_idx]
y_base = y_test[-1].ravel()


gpr = gper(X_gpr_sampled, y_gpr_sampled, best_model)
adj_preds = adjuster(best_model, gpr, X_base)

print("R2 = ", r2(y_base, adj_preds))
print("RMSE = ", rmse(y_base, adj_preds))
print("MAE = ", mae(y_base, adj_preds))

forecasted_frp = []

for day in range(31):
    # Step 1: Scale X
    X_base_scaled = scaler_all.transform(X_base).astype(np.float32)
    X_gpr_scaled = scaler_all.transform(X_gpr).astype(np.float32)
    y_gpr_scaled = scaler_frp.transform(y_gpr).ravel()

    # Step 2: Predict in scaled space
    adj_preds_scaled = adj_preds = adjuster(best_model, gpr, X_base_scaled)

    # Step 3: Save unscaled prediction (for plot only)
    adj_preds_orig = scaler_frp.inverse_transform(adj_preds_scaled.reshape(-1, 1)).ravel().item()
    forecasted_frp.append(adj_preds_orig)

    # Step 4: Append new scaled values to training data
    X_curr = np.vstack([X_curr, X_base])
    y_curr = np.vstack([y_curr, adj_preds_scaled.reshape(-1, 1)])

    # Step 5: Retrain
    best_model.fit(
        X_curr, y_curr,
        max_epochs=5,
        batch_size=1024,
        virtual_batch_size=128,
        compute_importance=False
    )

    # Step 6: Update GPR window
    X_gpr = X_curr[-len(X_gpr):]
    y_gpr = y_curr[-len(y_gpr):]

    # Step 7: Lag-based feature update
    X_base = update_X_base(X_base, adj_preds_orig, col_names)

