import requests
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer

# Download the punctuation rules NLTK needs to split sentences
nltk.download('punkt')

def scrape_article(url):
    # 1. Fetch the webpage using a User-Agent header to avoid being blocked
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    
    # 2. Parse the HTML content
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # 3. Extract text from all paragraph (<p>) tags
    paragraphs = soup.find_all('p')
    raw_text = " ".join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])
    texts = []
    for p in paragraphs:
      t = p.get_text(strip=True)
      if t:
         texts.append(t)
    raw_text = " ".join(texts)
    
    return raw_text

def tokenize(raw_text):
    # Split the giant block of text into a clean list of individual sentences
    sentences = sent_tokenize(raw_text)
    return sentences

def gen_emb(sentences):

    # Load the lightweight, highly efficient pre-trained language model
    model=SentenceTransformer('all-MiniLM-L6-v2')
    
    # Convert our list of text sentences into dense mathematical vectors
    embeddings = model.encode(sentences)
    return embeddings

#MAIN
if __name__ == "__main__":
    
    target_url = "https://en.wikipedia.org/wiki/Quantum_computing"  
    
    #1.Scraping data from the web page
    text_data = scrape_article(target_url)
    
    #2. Tokenizing the raw text into sentences
    sentence_list = tokenize(text_data)
    
    #3. Transforming sentences into vector embeddings
    math_embeddings = gen_emb(sentence_list)
    
    #Execution
    print(f"Successfully processed {len(sentence_list)} sentences into numerical matrices.")
    print(f"Vector Dimensions per sentence: {math_embeddings[0].shape[0]}")
    #Sample preview.
    print("\nSample Preview (Sentence 1 Text):")
    print(sentence_list[0])
    print("\nSample Preview (Sentence 1 Embedding Vector - First 5 Dimensions):")
    print(math_embeddings[0][:5])

# Open a new file in write mode ('w')
with open("output.txt", "w", encoding="utf-8") as file:
    # Loop through both the sentences and their matching math vectors at the same time
    for i, (sentence, embedding) in enumerate(zip(sentence_list, math_embeddings)):
        file.write(f"--- SENTENCE {i+1} ---\n")
        file.write(f"Text: {sentence}\n")
        
        # Convert the mathematical array into a standard Python list so it can be saved as text
        file.write(f"Embedding: {embedding.tolist()}\n\n")

