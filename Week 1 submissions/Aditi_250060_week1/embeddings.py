from sklearn.feature-extraction.text
import TfidVectorizer

sentences = [
    "Artificial Intelligence should be used limitedly",
    "Transformers are so much used in NLP hehe"
]
vectorizer = TfidVectorizer()
embeddings = vectorizer.fit_transform(sentences)
print(embeddings.toarray())
