# python -m pip install requests
# python -m pip install beautifulsoup4

import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse


URL = "https://www.youtube.com/"
page = requests.get(URL)
permitted_domains = [URL]
upper_bound = 25

soup = BeautifulSoup(page.content, 'html.parser')
results = soup.find_all("a", href=True)

print("Starting...")
i = 0
for link in results:
    href = link.get("href")
    if i < upper_bound:
        print(f"Found URL: {link['href']}")
        i += 1
    else:
        break


# python_jobs = results.find_all("h2",string=lambda text: "python" in text.lower())
# # From inside outwards:
# # (i) <h2></h2>
# # (ii) <div class="media-content">
# # (iii) <div class="media">
# # (iv) <div class="card-content">
# python_job_cards = [h2_element.parent.parent.parent for h2_element in python_jobs]
#
# for job_card in python_job_cards:
#     title_element = job_card.find("h2", class_="title")
#     company_element = job_card.find("h3", class_="company")
#     location_element = job_card.find("p", class_="location")
#     print(title_element.text.strip())
#     print(company_element.text.strip())
#     print(location_element.text.strip())
#     link_url = job_card.find_all("a")[1]["href"]
#     print(f"Apply here: {link_url}\n")


