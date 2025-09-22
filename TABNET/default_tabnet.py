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
df = df.sample(frac=0.05,random_state=42)

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
    return mean_absolute_error(y_true, y_pred)

# Default TabNetRegressor

model = TabNetRegressor()
model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], patience=10, max_epochs=30, batch_size=1024, virtual_batch_size=128, num_workers=6)

y_pred = model.predict(X_test)

print("R2 = ", r2(y_test, y_pred))
print("RMSE = ", rmse(y_test, y_pred))
print("MAE = ", mae(y_test, y_pred))
