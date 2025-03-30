
from data_loader import BLSDataLoader
from embedding_model import EmbeddingModel
from retrieval import get_top_k

import pandas as pd
import os

from openai import AzureOpenAI
from dotenv import load_dotenv
from preprocessor import Preprocessor

load_dotenv()

# Set up the request with a browser-like User-Agent
url = 'https://www.bls.gov/oes/special-requests/oesm23nat.zip'
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

loader = BLSDataLoader()
extracted_file_path = loader.extract_xlsx()

df = pd.read_excel(extracted_file_path)
df = df.query('O_GROUP == "detailed"').reset_index(drop=True)

embedder = EmbeddingModel()
embedder.load_model()
preprocessor = Preprocessor()

df['CLEAN_TITLE'] = df['OCC_TITLE'].apply(preprocessor.pipeline)
df['EMBEDS'] = df['CLEAN_TITLE'].apply(embedder.get_sentence_embedding)

test_sentence = "I like Software development"
test_embedding = embedder(preprocessor(test_sentence))

top_rows = get_top_k(test_embedding, df, k=3)
top_jobs = [row['OCC_TITLE'] for row in top_rows]
print(top_jobs)

client = AzureOpenAI(
  azure_endpoint = "https://models.inference.ai.azure.com",
  api_key=os.getenv("API_TOKEN"),
  api_version="2024-02-01",
)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"{test_sentence}"},
    ]
)

print(response.choices[0].message.content)
