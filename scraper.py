# python -m pip install requests
# python -m pip install beautifulsoup4

import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
from urllib.robotparser import RobotFileParser

URL = "https://store.steampowered.com/"
permitted_domains = ["store.steampowered.com", "google.com", "youtube.com"]
upper_bound = 500

# Check robots.txt
rp = RobotFileParser()
rp.set_url(urljoin(URL,"/robots.txt")) #locates robots.txt
rp.read()

page = requests.get(URL) #downloads the page content
soup = BeautifulSoup(page.content, 'html.parser')
results = soup.find_all("a", href=True) #extracts all URLs

print("Starting program...")
count = 0    #monitors number of valid URLs found
seen = set() #monitors already-visited URLs to avoid duplicates

for link in results:
    # Terminates if the upper bound is reached
    if count >= upper_bound:
        break
    href = link.get("href")
    full_url = urljoin(URL, href) #normalize into functional URL

    # Skips duplicate URLs
    if full_url in seen:
        continue
    seen.add(full_url) #marks the URL as seen
    domain = urlparse(full_url).netloc #extracts domain name

    # Skips if domain is not permitted
    if domain not in permitted_domains:
        continue

    # Skips if the standard for robot exclusion is violated
    if not rp.can_fetch("*", full_url):
        print("NOTICE: URL skipped, Robot Exclusion Standard violated.")
        continue

    print(f"Found URL: {full_url}")
    count += 1

