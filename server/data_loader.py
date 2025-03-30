# data_loader.py
import requests
import zipfile
import os
import io

class BLSDataLoader:
    def __init__(self, data_dir="data"):
        self.url = "https://www.bls.gov/oes/special-requests/oesm23nat.zip"
        self.headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        self.extracted_filename = None

    def extract_xlsx(self):
        # If already extracted, skip download and extraction
        if self.extracted_filename and os.path.exists(self.extracted_filename):
            print(f"File already exists: {self.extracted_filename}")
            return self.extracted_filename

        # Check for any existing .xlsx file in the data directory
        for file in os.listdir(self.data_dir):
            if file.endswith(".xlsx"):
                file_path = os.path.join(self.data_dir, file)
                print(f"Found existing file in data directory: {file_path}")
                self.extracted_filename = file_path
                return file_path

        # If not found, download and extract
        print("Downloading ZIP file...")
        response = requests.get(self.url, headers=self.headers)
        if response.status_code != 200:
            raise Exception(f"Failed to download file: {response.status_code}")
        zip_bytes = io.BytesIO(response.content)

        with zipfile.ZipFile(zip_bytes) as z:
            file_names = z.namelist()
            xlsx_file = next((name for name in file_names if name.endswith('.xlsx')), None)
            if not xlsx_file:
                raise FileNotFoundError("No .xlsx file found in the archive.")

            output_path = os.path.join(self.data_dir, os.path.basename(xlsx_file))

            print(f"Extracting {xlsx_file} to {output_path}...")
            with z.open(xlsx_file) as extracted_file, open(output_path, "wb") as out_file:
                out_file.write(extracted_file.read())

            self.extracted_filename = output_path
            return output_path
