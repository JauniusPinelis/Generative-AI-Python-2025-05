import requests
from bs4 import BeautifulSoup
 
response = requests.get("https://lt.wikipedia.org/wiki/Bir%C5%BEai")
soup = BeautifulSoup(response.text, 'html.parser')
page_text = soup.get_text(strip=True)   # Extract visible text from the page
 
print(page_text)