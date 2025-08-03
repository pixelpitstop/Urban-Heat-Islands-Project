import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import os

# =========================
# CONFIG
# =========================
DATA_DIR = "data"
TRAIN_FILE = os.path.join(DATA_DIR, "final_realistic_augmented_dataset.csv")
PRED_FILE = os.path.join(DATA_DIR, "NDVI_LST_Chennai_July2025_with_LST.csv")
OUTPUT_PRED_FILE = os.path.join(DATA_DIR, "utei_predictions.csv")
OUTPUT_UTEI_SORTED = os.path.join(DATA_DIR, "utei_sorted.csv")

# =========================
# 1. LOAD & CLEAN TRAINING DATA
# =========================
df = pd.read_csv(TRAIN_FILE).dropna()
df.columns = df.columns.str.strip().str.replace(" ", "_")

# Rename columns for consistency
rename_map = {
    "LST": "LST_C",
    "Latitude_(°N)": "Latitude",
    "Longitude_(°E)": "Longitude"
}
df.rename(columns=rename_map, inplace=True)

# Ensure required columns exist
required_cols = ["Temperature", "Humidity", "Latitude", "Longitude", "LST_C", "NDVI"]
missing = set(required_cols) - set(df.columns)
if missing:
    raise ValueError(f"Missing columns in training data: {missing}")

# Optional: force humidity constant (if proof-of-concept)
df["Humidity"] = 60

# =========================
# 2. TRAIN LASSO MODEL
# =========================
X = df[["Humidity", "LST_C", "NDVI"]]
y = df["Temperature"]

model = Lasso(alpha=0.0001)
model.fit(X, y)

y_pred_train = model.predict(X)
print(f"Train RMSE: {np.sqrt(mean_squared_error(y, y_pred_train)):.3f}")
print(f"Train MAE: {mean_absolute_error(y, y_pred_train):.3f}")
print(f"Train R²: {r2_score(y, y_pred_train):.3f}")

# =========================
# 3. PREDICT ON NEW DATA
# =========================
pred_df = pd.read_csv(PRED_FILE)
if "Humidity" not in pred_df.columns:
    pred_df["Humidity"] = 60

pred_df["Predicted_Temperature"] = model.predict(pred_df[["Humidity", "LST_C", "NDVI"]])

# =========================
# 4. CALCULATE UTEI
# =========================
scaler = MinMaxScaler()
pred_df[["LST_norm", "NDVI_norm", "Humidity_norm"]] = scaler.fit_transform(
    pred_df[["LST_C", "NDVI", "Humidity"]]
)
# Example formula — modify if you have a different one
pred_df["UTEI"] = pred_df["LST_norm"] + pred_df["Humidity_norm"] - pred_df["NDVI_norm"]

# Save predictions + UTEI
pred_df.to_csv(OUTPUT_PRED_FILE, index=False)
print(f"✅ UTEI predictions saved to {OUTPUT_PRED_FILE}")

# =========================
# 5. SORT BY UTEI
# =========================
df_sorted = pred_df.sort_values(by="UTEI", ascending=False)
df_sorted.to_csv(OUTPUT_UTEI_SORTED, index=False)
print(f"📊 Sorted UTEI file saved to {OUTPUT_UTEI_SORTED}")

# =========================
# 6. VISUALIZATIONS
# =========================
# Histogram of predictions
plt.figure(figsize=(10, 6))
sns.histplot(pred_df["Predicted_Temperature"], bins=30, kde=True, color="skyblue")
plt.title("Predicted Air Temperature Distribution (Lasso Model)")
plt.xlabel("Temperature (°C)")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

# NDVI vs Temperature scatter
plt.figure(figsize=(8, 6))
plt.scatter(pred_df["NDVI"], pred_df["Predicted_Temperature"], alpha=0.4, c="green")
plt.xlabel("NDVI")
plt.ylabel("Predicted Temperature (°C)")
plt.title("NDVI vs Predicted Temperature")
plt.grid(True)
plt.show()

# Heatmap of predicted temperatures
heat_df = pred_df.dropna(subset=["Latitude", "Longitude", "Predicted_Temperature"])
m = folium.Map(location=[heat_df["Latitude"].mean(), heat_df["Longitude"].mean()],
               zoom_start=12, tiles="cartodbpositron")
heat_data = [
    [row["Latitude"], row["Longitude"], row["Predicted_Temperature"]]
    for _, row in heat_df.iterrows()
]
HeatMap(heat_data, radius=12, blur=15, max_zoom=1).add_to(m)
m.save("temperature_heatmap.html")
print("🌍 Heatmap saved as temperature_heatmap.html")
