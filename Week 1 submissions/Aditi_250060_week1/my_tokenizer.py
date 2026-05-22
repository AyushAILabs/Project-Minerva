import nltk

nltk.download('punkt_tab')
nltk.download('punkt')
from nltk.tokenize import word_tokenize
with open("corpus/articles.txt", "r", encoding="utf-8") as f:
    text= f.read()
    tokens = word_tokenize(text)
    print(tokens[:50])
