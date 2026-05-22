import requests
from bs4 import BeautifulSoup as bs
from srsly.ruamel_yaml.compat import utf8

URL = "https://www.bbc.com/news/articles/c1k2lmmjvzxo"
page = requests.get(URL)

soup =  bs(page.text, "html.parser")
paragraphs = soup.find_all("p")

article_text = ""

for p in paragraphs:
    article_text += p.get_text() + "\n"

with open("text.txt", "w", encoding="utf-8") as file:
    file.write(article_text)

