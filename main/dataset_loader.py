import subprocess
import datetime
import os
from dotenv import load_dotenv

load_dotenv()

# set Kaggle API credentials from environment variables
KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME")
KAGGLE_KEY = os.getenv("KAGGLE_KEY")

def download_dataset():

    if not KAGGLE_USERNAME or not KAGGLE_KEY:
        raise ValueError("KAGGLE_USERNAME or KAGGLE_KEY not set in .env file")

    download_folder = "data/raw"
    today = datetime.date.today().isoformat()

    file_path = os.path.join(download_folder, f"Consumer_Complaints_{today}.csv")

    # Load the latest version of Dataset from Kaggle
    try:
        subprocess.run([
        "kaggle", "datasets", "download",
        "-d", "shashwatwork/consume-complaints-dataset-fo-nlp",
        "-p", "data/raw", "--unzip"
        ], check=True)
        print("Dataset downloaded successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while downloading the dataset: {e}")
        
    original_file = os.path.join(download_folder, "complaints_processed.csv")

    if os.path.exists(original_file):
        os.rename(original_file, file_path)
    else:
        raise FileNotFoundError(f"{original_file} not found after download.")