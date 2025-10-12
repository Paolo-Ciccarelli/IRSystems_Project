import nltk
nltk.download('reuters')
nltk.download('averaged_perceptron_tagger_eng')
print("DEBUG: reuters corpus downloaded")
print("DEBUG: averaged_perceptron_tagger_eng downloaded")

#Each document begins with <Reuters...
from bs4 import BeautifulSoup
with open(r"C:\Users\P_CICCAR\Downloads\reuters21578\reut2-020.sgm", "r", encoding="latin-1") as f:
    soup = BeautifulSoup(f, "html.parser")
    print("reut2-020 contains:", len(soup.find_all("reuters")), "documents.")

with open(r"C:\Users\P_CICCAR\Downloads\reuters21578\reut2-021.sgm", "r", encoding="latin-1") as f:
    soup = BeautifulSoup(f, "html.parser")
    print("reut2-021 contains:", len(soup.find_all("reuters")), "documents.")

class Article:
    newID = 0
    text = ""
    def __init__(self, newID, text):
        self.newID = newID
        self.text = text
    def __str__(self):
        return f"Article ID: {self.newID}, Text: {self.text}"

articles = []
reuters_tags = soup.find_all("reuters")
for tag in reuters_tags:
    newID = tag.get('newid')
    text = tag.text.strip()
    article = Article(newID, text)
    articles.append(article)

for article in articles:
    print(article)