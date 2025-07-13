import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
import joblib


DATA_PATH = "dataset/train.csv"
MODEL_PATH = "models/house_price_predictor.joblib"

df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=["SalePrice"])

x = df.drop(columns=["Id", "SalePrice"])
y = df["SalePrice"]


x = pd.get_dummies(x)

x = x.fillna(x.median())

x_train, x_val, y_train, y_val = train_test_split(
    x, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_val_scaled = scaler.transform(x_val)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(x_train_scaled, y_train)

y_pred = model.predict(x_val_scaled)
rmse = sqrt(mean_squared_error(y_val, y_pred))
print(f"Validation RMSE: {rmse:.2f}")


joblib.dump(
    {"model": model, "scaler": scaler, "columns": x.columns.tolist()}, MODEL_PATH
)

print(f"Model saved to {MODEL_PATH}")
