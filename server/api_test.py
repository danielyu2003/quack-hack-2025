import re
import requests
import zipfile
import io
import pandas as pd
import fasttext.util
import numpy as np
import os
from numpy.linalg import norm

# Set up the request with a browser-like User-Agent
url = 'https://www.bls.gov/oes/special-requests/oesm23nat.zip'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
}

response = requests.get(url, headers=headers)
zip_bytes = io.BytesIO(response.content)

df = None

with zipfile.ZipFile(zip_bytes) as z:
    file_names = z.namelist()
    xlsx_file = next((name for name in file_names if name.endswith('.xlsx')), None)
    if xlsx_file:
        with z.open(xlsx_file) as f:
            df = pd.read_excel(f).query('O_GROUP == "detailed"')
            df = df.reset_index(drop=True)
            print("Data loaded into DataFrame successfully.")
    else:
        print("No .xlsx file found in the archive.")

# Get the directory of the current Python file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Save the original working directory
original_dir = os.getcwd()

try:
    # Change to the directory where the script is located
    os.chdir(current_dir)

    # Download the model (it will now save in the current script directory)
    fasttext.util.download_model('en', if_exists='ignore')
    ft = fasttext.load_model('cc.en.300.bin')

finally:
    # Change back to the original working directory
    os.chdir(original_dir)

fasttext.util.reduce_model(ft, 100)

def sentence_embedder(row):
    word_embeds = [ft.get_word_vector(w) for w in list(row['LOW_TITLE'])]
    return sum(word_embeds) / len(word_embeds)

def embed_words(words):
    word_embeds = [ft.get_word_vector(w) for w in words]
    return sum(word_embeds) / len(word_embeds)

def cosine_sim(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

def clean_string(row):
    text = row['OCC_TITLE']
    return re.sub(r"[^a-zA-Z\s]", "", text).lower()

df['LOW_TITLE'] = df.apply(clean_string, axis=1)
df['EMBEDS'] = df.apply(sentence_embedder, axis=1)
print(df[['OCC_TITLE', 'EMBEDS']])

test_sentence = "software development".lower()
test_embed = embed_words(test_sentence)

similarities = []

for index, row in df.iterrows():
    diff = cosine_sim(test_embed, row['EMBEDS'])
    similarities.append((diff, row))

# Sort by similarity in descending order
top3 = sorted(similarities, key=lambda x: x[0], reverse=True)[:3]

# Separate top3diff and top3jobs
top3diff = [item[0] for item in top3]
top3jobs = [item[1]['OCC_TITLE'] for item in top3]

print(top3jobs)
