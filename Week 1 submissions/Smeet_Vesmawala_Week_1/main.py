import requests
from bs4 import BeautifulSoup
import json
import nltk
from nltk.tokenize import word_tokenize
import chromadb
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer




base_url = "https://quotes.toscrape.com"

current_page = "/"

data = []

while current_page:

    response = requests.get(base_url + current_page)

    soup = BeautifulSoup(response.text, "html.parser")

    quotes = soup.find_all("div", class_="quote")

    for quote in quotes:

        text = quote.find("span", class_="text").get_text()
        author = quote.find("small", class_="author").get_text()

        data.append({
            "text": text,
            "author": author
        })

    next_button = soup.find("li", class_="next")

    if next_button:
        current_page = next_button.find("a")["href"]
    else:
        current_page = None

with open("data/corpus.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4)


df = pd.DataFrame(data)

df.to_csv("data/news_corpus.csv", index=False)

print("\nCSV export complete.")
print("Scraping complete.")


all_tokens = []

for item in data:

    tokens = word_tokenize(item["text"])

    all_tokens.append(tokens)



print(f"\nTotal tokenized quotes: {len(all_tokens)}")



model = SentenceTransformer('all-MiniLM-L6-v2')

texts = [item["text"] for item in data]

embeddings = model.encode(texts)

print("\nEmbedding Shape:\n")
print(embeddings.shape)

print("\nEmbeddings saved successfully.")
np.save("outputs/embeddings.npy", embeddings)



client = chromadb.PersistentClient(path="./vector_db")

collection = client.get_or_create_collection(name="quotes_collection")

for i, item in enumerate(data):

    collection.add(
        ids=[str(i)],
        embeddings=[embeddings[i].tolist()],
        documents=[item["text"]],
        metadatas=[{"author": item["author"]}]
    )
client.persist()
print("\nVector DB storage complete.")
print(f"Total vectors stored: {collection.count()}")
print(client.list_collections())