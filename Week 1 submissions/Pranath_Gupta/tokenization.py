import spacy

nlp = spacy.blank("en")

with open("text.txt", 'r', encoding="utf-8") as file:
    text = file.read()

token1 = [token.text for token in nlp(text) if token.is_alpha]