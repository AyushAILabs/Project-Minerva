from tokenization import token1
from gensim.models import Word2Vec

tokens = [token1]

model = Word2Vec(tokens, vector_size = 10, window = 6, min_count=1)

with open("embeddings.txt", "w", encoding="utf-8") as file:

    for word in model.wv.index_to_key:

        vector = model.wv[word]
        file.write(f"{word}: {vector}\n")