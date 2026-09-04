from pathlib import Path
import numpy as np
import pandas as pd
import rasterio

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
WORLDCLIM = ROOT / "data" / "worldclim"
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)


def sample_raster(path, coordinates):
    with rasterio.open(path) as src:
        values = np.array([x[0] for x in src.sample(coordinates)], dtype=float)
        if src.nodata is not None:
            values[values == src.nodata] = np.nan
    return values


gps_files = sorted(RAW.glob("CURLEW_VLAANDEREN-gps-*.csv.gz"))
if not gps_files:
    raise FileNotFoundError("No GPS files found. Run 01_download_data.py first.")

frames = []
for file in gps_files:
    print("Reading:", file.name)
    frames.append(pd.read_csv(file, low_memory=False))

data = pd.concat(frames, ignore_index=True)

required = [
    "individual-local-identifier",
    "timestamp",
    "location-long",
    "location-lat",
]
missing = [name for name in required if name not in data.columns]
if missing:
    raise KeyError(f"Missing expected GPS columns: {missing}")

tracks = data[required].copy()
tracks.columns = ["bird_id", "timestamp", "longitude", "latitude"]
tracks["timestamp"] = pd.to_datetime(tracks["timestamp"], utc=True, errors="coerce")
tracks["longitude"] = pd.to_numeric(tracks["longitude"], errors="coerce")
tracks["latitude"] = pd.to_numeric(tracks["latitude"], errors="coerce")
tracks = tracks.dropna().drop_duplicates(["bird_id", "timestamp"])
tracks = tracks.sort_values(["bird_id", "timestamp"]).reset_index(drop=True)

# Keep the first fix in each hour. This keeps the example small and avoids
# treating dense sampling bursts as independent observations.
tracks["hour_bin"] = tracks["timestamp"].dt.floor("h")
tracks = tracks.groupby(["bird_id", "hour_bin"], as_index=False).first()
tracks = tracks.drop(columns="hour_bin").sort_values(["bird_id", "timestamp"]).reset_index(drop=True)

tracks["prev_lon"] = tracks.groupby("bird_id")["longitude"].shift()
tracks["prev_lat"] = tracks.groupby("bird_id")["latitude"].shift()
tracks["prev_time"] = tracks.groupby("bird_id")["timestamp"].shift()

lat1 = np.radians(tracks["prev_lat"].to_numpy())
lat2 = np.radians(tracks["latitude"].to_numpy())
dlat = lat2 - lat1
dlon = np.radians(tracks["longitude"].to_numpy() - tracks["prev_lon"].to_numpy())

a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
tracks["step_km"] = 6371.0088 * 2 * np.arcsin(np.sqrt(a))
tracks["dt_hours"] = (tracks["timestamp"] - tracks["prev_time"]).dt.total_seconds() / 3600
tracks["step_speed_kmh"] = tracks["step_km"] / tracks["dt_hours"]

# Very short time gaps are sensitive to GPS jitter, while very long gaps make a
# single-step speed hard to interpret. These limits are simple QA filters.
tracks = tracks[tracks["dt_hours"].between(5 / 60, 6)].copy()
tracks = tracks[tracks["step_speed_kmh"].between(0, 150)].copy()

# This is an operational label for the ML exercise, not a validated behavioural state.
# The threshold is checked again in the sensitivity output below.
tracks["active_movement"] = (tracks["step_speed_kmh"] >= 0.5).astype(int)

tracks["hour"] = tracks["timestamp"].dt.hour + tracks["timestamp"].dt.minute / 60
tracks["day_of_year"] = tracks["timestamp"].dt.dayofyear
tracks["hour_sin"] = np.sin(2 * np.pi * tracks["hour"] / 24)
tracks["hour_cos"] = np.cos(2 * np.pi * tracks["hour"] / 24)
tracks["doy_sin"] = np.sin(2 * np.pi * tracks["day_of_year"] / 365.25)
tracks["doy_cos"] = np.cos(2 * np.pi * tracks["day_of_year"] / 365.25)

bio1 = WORLDCLIM / "wc2.1_10m_bio_1.tif"
bio12 = WORLDCLIM / "wc2.1_10m_bio_12.tif"
elev = WORLDCLIM / "wc2.1_10m_elev.tif"
for path in [bio1, bio12, elev]:
    if not path.exists():
        raise FileNotFoundError(f"Missing WorldClim file: {path.name}. Run 01_download_data.py first.")

coords = list(zip(tracks["longitude"], tracks["latitude"]))
print("Extracting WorldClim values...")
tracks["bio1"] = sample_raster(bio1, coords)
tracks["bio12"] = sample_raster(bio12, coords)
tracks["elevation_m"] = sample_raster(elev, coords)

tracks = tracks.dropna(subset=["bio1", "bio12", "elevation_m"]).copy()
tracks["row_id"] = np.arange(len(tracks))

keep = [
    "row_id",
    "bird_id",
    "timestamp",
    "longitude",
    "latitude",
    "active_movement",
    "hour_sin",
    "hour_cos",
    "doy_sin",
    "doy_cos",
    "bio1",
    "bio12",
    "elevation_m",
]

out = PROCESSED / "curlew_ml_table.csv"
tracks[keep].to_csv(out, index=False)

sensitivity = []
for threshold in [0.5, 1, 2, 5, 10]:
    sensitivity.append(
        {
            "speed_threshold_kmh": threshold,
            "active_fraction": float((tracks["step_speed_kmh"] >= threshold).mean()),
            "n_rows": len(tracks),
        }
    )
pd.DataFrame(sensitivity).to_csv(PROCESSED / "movement_threshold_sensitivity.csv", index=False)

print("\nSaved:", out)
print("Rows:", len(tracks))
print("Birds:", tracks["bird_id"].nunique())
print("Active fraction:", round(tracks["active_movement"].mean(), 3))
