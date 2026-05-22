import requests
from bs4 import BeautifulSoup

url = "https://www.bbc.com/news"

response = requests.get(url)

soup = BeautifulSoup(response.text, "html.parser")

headlines = soup.find_all("h2")

articles = []

for h in headlines:
    text = h.get_text(strip=True)

    if text:
        articles.append(text)

with open("corpus/articles.txt", "w", encoding="utf-8") as f:
    for article in articles:
        f.write(article + "\n")

print("Scraping completed!")