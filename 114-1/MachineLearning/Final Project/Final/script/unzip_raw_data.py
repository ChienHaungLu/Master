import zipfile
from pathlib import Path

RAW_DIR = Path("Final_Project/raw")
OUTPUT_DIR = Path("Final_Project/original_wsis")

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for zip_file_path in RAW_DIR.glob("*.zip"):
    print(f"Extracting {zip_file_path.name} to {OUTPUT_DIR}...")
    try:
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(OUTPUT_DIR)
        print(f"Finished extracting {zip_file_path.name}")
    except zipfile.BadZipFile:
        print(f"Warning: {zip_file_path.name} is a bad zip file and could not be extracted.")
    except Exception as e:
        print(f"Error extracting {zip_file_path.name}: {e}")

print("All zip files extraction process completed.")