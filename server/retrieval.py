# similarity.py
import numpy as np
from numpy.linalg import norm

def cosine_sim(a, b):
    return np.dot(a, b) / (norm(a) * norm(b)) if norm(a) > 0 and norm(b) > 0 else 0.0

def get_top_k(query_embed, df, k=3, r=False):
    similarities = []

    for index, row in df.iterrows():
        diff = cosine_sim(query_embed, row['EMBEDS'])
        similarities.append((diff, row))

    sorted_by_sim = sorted(similarities, key=lambda x: x[0], reverse=(not r))
    top_rows = [item[1] for item in sorted_by_sim[:k]]
    return top_rows
