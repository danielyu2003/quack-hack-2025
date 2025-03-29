import requests
import zipfile
import io
import pandas as pd
import fasttext.util
import numpy as np
from numpy.linalg import norm

# Set up the request with a browser-like User-Agent
url = 'https://www.bls.gov/oes/special-requests/oesm23nat.zip'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
}

response = requests.get(url, headers=headers)
zip_bytes = io.BytesIO(response.content)

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


# print(sum(df['TOT_EMP']))
# print(df[['OCC_TITLE', 'TOT_EMP', 'H_MEAN', 'H_MEDIAN', 'A_MEAN', 'A_MEDIAN']][:10])

min_emp = max(df['TOT_EMP'])
print(min_emp)
print(df[df['TOT_EMP'] == min_emp]['OCC_TITLE'])

fasttext.util.download_model('en', if_exists='ignore')
ft = fasttext.load_model('cc.en.300.bin')
fasttext.util.reduce_model(ft, 100)

def sentence_embedder(row):
    word_embeds = [ft.get_word_vector(w) for w in list(row['OCC_TITLE'])]
    return np.mean(word_embeds)

def embed_words(words):
    word_embeds = [ft.get_word_vector(w) for w in words]
    return np.mean(word_embeds)

def cosine_sim(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

df['EMBEDS'] = df.apply(sentence_embedder, axis=1)
print(df[['OCC_TITLE', 'EMBEDS']])

test_sentence = "I like dogs"
test_embed = embed_words(test_sentence)

top3diff = [None, None, None]
top3jobs = [None, None, None]

for row in df:
    diff = cosine_sim(test_embed, row['EMBEDS'])
    if top3diff[0] is None or diff > top3diff[0]:
        top3diff[2] = top3diff[1]
        top3diff[1] = top3diff[0]
        top3diff[0] = diff
        top3jobs[2] = top3jobs[1]
        top3jobs[1] = top3jobs[0]
        top3jobs[0] = row['OCC_TITLE']
        continue
    elif top3diff[1] is None or diff > top3diff[1]:
        top3diff[2] = top3diff[1]
        top3diff[1] = diff
        top3jobs[2] = top3jobs[1]
        top3jobs[1] = row['OCC_TITLE']
        continue
    elif top3diff[2] is None or diff > top3diff[2]:
        top3diff[2] = diff
        top3jobs[2] = row['OCC_TITLE']
        continue
    else:
        continue

print(top3jobs)
