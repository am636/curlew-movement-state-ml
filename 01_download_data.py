from pathlib import Path
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
WORLDCLIM = ROOT / "data" / "worldclim"

RAW.mkdir(parents=True, exist_ok=True)
WORLDCLIM.mkdir(parents=True, exist_ok=True)

zenodo_base = "https://zenodo.org/records/15696532/files"
tracking_files = [
    "CURLEW_VLAANDEREN-gps-2020.csv.gz",
    "CURLEW_VLAANDEREN-gps-2021.csv.gz",
    "CURLEW_VLAANDEREN-gps-2022.csv.gz",
    "CURLEW_VLAANDEREN-gps-2023.csv.gz",
    "CURLEW_VLAANDEREN-gps-2024.csv.gz",
    "CURLEW_VLAANDEREN-reference-data.csv",
    "datapackage.json",
]

for name in tracking_files:
    target = RAW / name
    if target.exists() and target.stat().st_size > 0:
        print("Already downloaded:", name)
    else:
        print("Downloading:", name)
        urllib.request.urlretrieve(f"{zenodo_base}/{name}?download=1", target)

worldclim_files = {
    "wc2.1_10m_bio.zip": "https://geodata.ucdavis.edu/climate/worldclim/2_1/base/wc2.1_10m_bio.zip",
    "wc2.1_10m_elev.zip": "https://geodata.ucdavis.edu/climate/worldclim/2_1/base/wc2.1_10m_elev.zip",
}

for name, url in worldclim_files.items():
    target = WORLDCLIM / name
    if target.exists() and target.stat().st_size > 0:
        print("Already downloaded:", name)
    else:
        print("Downloading:", name)
        urllib.request.urlretrieve(url, target)

    with zipfile.ZipFile(target) as z:
        z.extractall(WORLDCLIM)

print("\nData ready.")
