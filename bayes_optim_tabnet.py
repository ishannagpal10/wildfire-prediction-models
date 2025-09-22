import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import optuna
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import torch

reader = pd.read_csv('https://huggingface.co/datasets/SanatRao/cleaned_scaled_fire_data.csv/resolve/main/cleaned_fire_data.csv', chunksize=100000, low_memory=False)
df = pd.concat(reader, ignore_index=True)

y=df['FRP'].values
X = df.drop(columns=['FRP']).values
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, random_state=42)

y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
y_valid = y_valid.reshape(-1, 1)

# Scoring functions
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))
def r2(y_true, y_pred):
    return r2_score(y_true, y_pred)
def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred

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
            optimizer_params=dict(lr=lr),
            verbose=1,
        )

        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], patience=10, max_epochs=50, batch_size=1024, virtual_batch_size=128, num_workers=6)

        y_pred = model.predict(X_valid)
        
        return scorer(y_valid, y_pred)

    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=50)
    optuna_mpl.plot_optimization_history(study)
    optuna_mpl.plot_param_importances(study)
    plt.show()


    best_params_ = study.best_params
    best_params = best_params_
    best_model = TabNetRegressor(n_d=best_params['n_d'], n_a=best_params['n_a'], n_steps=best_params['n_steps'], gamma=best_params['gamma'], lambda_sparse=best_params['lambda_sparse'], optimizer_fn=torch.optim.Adam, optimizer_params=dict(lr=best_params['lr']), seed=42, verbose=1)
    best_model.fit(X_train, y_train, patience=0, max_epochs=0, batch_size=1024, virtual_batch_size=128, num_workers=6)
    y_pred = best_model.predict(X_test)
    scores = {
        'rmse': rmse(y_test, y_pred),
        'mae': mae(y_test, y_pred),
        'r2': r2(y_test, y_pred)
    }
    return best_model,best_params,scores


best_model, best_params, scores = bayes_optim(r2, 'maximize')
all_results = {
    'best_model': best_model,
    'best_params': best_params,
    'scores': scores
}


print(f"\nBest overall model optimized for R2:")
print("Scores:", scores)
print("Params:", best_params)
print(all_results)
