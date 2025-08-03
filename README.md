# 🌆 Urban Thermal Equity Index (UTEI) - Chennai

**A Novel AI-Driven Framework to Detect and Quantify Urban Heat Inequality**  
> Harnessing **satellite data, geospatial analytics, and machine learning** to revolutionize how cities understand and address climate injustice.

---

## 🧠 The Idea

Most urban heat island (UHI) studies stop at mapping surface temperature — but this project goes further.  
I introduce the **Urban Thermal Equity Index (UTEI)** — a groundbreaking metric that doesn’t just measure heat,  
but quantifies **how unequally** heat is distributed across a city.

By blending:
- **Satellite NDVI (vegetation) data**
- **MODIS Land Surface Temperature (LST)**
- **Local humidity**
- **Geospatial coordinates**

...this creates a city-wide, hyperlocal **thermal equity map** that reveals **where heat mitigation is most urgently needed**.

This is not just a research exercise — it’s a **deployable system** that can run on live IoT feeds and inform **real-world urban planning**.

---

## 🏗 Project Structure
```text
uhi_project/
│
├── data/                                      # Data files
│   ├── final_training_dataset.csv             # Training data (with LST, NDVI, humidity, temperature)
│   ├── NDVI_LST_Chennai_July2025_with_LST.csv # GEE-processed prediction dataset
│   ├── utei_predictions.csv                   # Model output with predicted temperature & UTEI
│   ├── utei_sorted.csv                        # Ranked hotspots & coolest zones
│
├── uhi_model.py                               # Main ML + visualization pipeline
├── requirements.txt                           # Python dependencies
├── README.md                                  # Project documentation
└── .gitignore                                 # Ignore venv, cache, large files
```

## 🔄 Pipeline

1. **Data Acquisition**
   - NDVI from **Sentinel-2** (Google Earth Engine)
   - LST from **MODIS** (Google Earth Engine)
   - Humidity & temperature from local IoT/weather stations

2. **Data Preprocessing**
   - Column standardization
   - Missing value handling
   - Humidity adjustments (constant or measured)
   - Scaling for ML compatibility

3. **Model Training**
   - Lasso regression (baseline model)
   - Features: **Humidity, LST_C, NDVI**
   - Evaluation: RMSE, MAE, R²

4. **Prediction**
   - Predict air temperature across **entire Chennai grid**
   - Apply **UTEI formula** (heat + humidity − vegetation)

5. **Visualization**
   - **Interactive heatmap** (Leaflet/Folium)
   - **NDVI vs temperature scatter**
   - **Temperature distribution histogram**

6. **Outputs**
   - `utei_predictions.csv` → UTEI score for each location
   - `utei_sorted.csv` → Ranked zones (hotspots → coolest zones)
   - `temperature_heatmap.html` → Geo-visualized temperature map

---

## 🌱 Future Vision

While the current system uses a mix of satellite and historical IoT data,  
our **next step is sensor deployment**:

- Deploy **low-cost, solar-powered environmental sensors** across Chennai
- Measure **real-time temperature, humidity, and air quality**
- Stream data to a **central UTEI dashboard**
- Enable **live thermal equity tracking** and **instant urban heat alerts**
- Inform **city planners** and **community organizations** for targeted cooling interventions

---

## 🚀 Why This Matters

- **Novelty:** First known index that combines vegetation, land temperature, and humidity to quantify *thermal equity* — not just heat.
- **Scalability:** Works for any city with satellite coverage and minimal IoT data.
- **Impact:** Can drive equitable climate policy and urban planning, prioritizing vulnerable neighborhoods.

---

## 📊 Example Output

**Top 5 Hottest Zones (High UTEI):**
| Latitude | Longitude | UTEI  |
|----------|-----------|-------|
| 13.05    | 80.27     | 0.912 |
| 13.08    | 80.23     | 0.907 |
| ...      | ...       | ...   |

**Top 5 Coolest Zones (Low UTEI):**
| Latitude | Longitude | UTEI  |
|----------|-----------|-------|
| 12.97    | 80.20     | 0.221 |
| 12.95    | 80.18     | 0.230 |
| ...      | ...       | ...   |

---

## ⚙️ Installation & Usage

```bash
# Clone the repo
git clone https://github.com/<your-username>/UHI_Project.git
cd UHI_Project

# Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Mac/Linux
.venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt

# Run the model
python uhi_model.py


