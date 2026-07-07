"""
download_dataset.py
One-click dataset setup for MindBridge.
Run: python download_dataset.py

Options:
  1. Uses kaggle CLI if credentials exist (~/.kaggle/kaggle.json)
  2. Opens the Kaggle download page in your browser as fallback

After running, the CSV will be at: data/reddit_depression.csv
"""
import os
import sys
import subprocess
import webbrowser

DATASET_PATH = "data/reddit_depression.csv"
KAGGLE_DATASET = "infamouscoder/depression-reddit-cleaned"
KAGGLE_URL = "https://www.kaggle.com/datasets/infamouscoder/depression-reddit-cleaned"

print("=" * 60)
print("MindBridge — Dataset Setup")
print("=" * 60)

# Check if already exists
if os.path.exists(DATASET_PATH):
    import pandas as pd
    df = pd.read_csv(DATASET_PATH)
    print(f"\nDataset already exists: {DATASET_PATH}")
    print(f"Rows: {len(df)} | Columns: {df.columns.tolist()}")
    print("\nNothing to do. Run: python src/data_loader.py")
    sys.exit(0)

os.makedirs("data", exist_ok=True)

# Try Kaggle CLI
kaggle_json = os.path.join(os.path.expanduser("~"), ".kaggle", "kaggle.json")
if os.path.exists(kaggle_json):
    print("\nKaggle credentials found. Downloading via CLI...")
    result = subprocess.run(
        [sys.executable, "-m", "kaggle", "datasets", "download",
         "-d", KAGGLE_DATASET, "-p", "data/", "--unzip"],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        print(result.stdout)
        # Rename to standard name if needed
        import glob
        csvs = glob.glob("data/*.csv")
        for csv in csvs:
            if "depression" in csv.lower() or "reddit" in csv.lower():
                if csv != DATASET_PATH:
                    os.rename(csv, DATASET_PATH)
                    print(f"Renamed {csv} -> {DATASET_PATH}")
        if os.path.exists(DATASET_PATH):
            print(f"\nDataset ready at: {DATASET_PATH}")
            print("Now run: python src/data_loader.py")
        else:
            print("CSV downloaded but could not find it. Check data/ folder.")
    else:
        print(f"Kaggle CLI error: {result.stderr}")
else:
    print("\nKaggle credentials not found.")
    print(f"kaggle.json not at: {kaggle_json}")

print("\n" + "=" * 60)
print("MANUAL DOWNLOAD STEPS (takes 2 minutes):")
print("=" * 60)
print("1. Opening Kaggle dataset page in your browser...")
print("2. Sign in to Kaggle (free account)")
print("3. Click the DOWNLOAD button on the page")
print("4. Extract the zip file")
print("5. Copy the CSV to: data/reddit_depression.csv")
print("6. Run: python src/data_loader.py")
print("=" * 60)
print()

try:
    webbrowser.open(KAGGLE_URL)
    print(f"Browser opened: {KAGGLE_URL}")
except Exception:
    print(f"Open this URL in your browser:\n{KAGGLE_URL}")
