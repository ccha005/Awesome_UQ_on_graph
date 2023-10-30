import requests
from bs4 import BeautifulSoup
import re

# URL of the search page
URL = "https://arxiv.org/search/?query=Uncertainty+Quantification+machine+learning&searchtype=title&abstracts=show&order=-announced_date_first&size=200"
# change URL for different search

# Make a request to the website and get its content
response = requests.get(URL)
soup = BeautifulSoup(response.content, 'html.parser')

# Find all the articles
articles = soup.find_all('li', class_='arxiv-result')

# Store the formatted information
results = []

for article in articles:
    title = article.find('p', class_='title is-5 mathjax').text.strip()
    date = article.find('p', class_='is-size-7').text.strip().split(';')[0].strip()
    link = article.find('p', class_='list-title is-inline-block').find('a')['href']
    
    # Extract date components using regex
    match = re.search(r"(\d{1,2}) ([A-Za-z]+), (\d{4})", date)
    day, month, year = match.groups()
    month_to_num = {
        'January': '01', 'February': '02', 'March': '03', 'April': '04', 'May': '05', 'June': '06',
        'July': '07', 'August': '08', 'September': '09', 'October': '10', 'November': '11', 'December': '12'
    }

    formatted_date = f"{year}.{month_to_num[month]}.{day.zfill(2)}"
    
    formatted_info = f"- {title}. [*{formatted_date}*] [[Arxiv]({link})]"
    results.append(formatted_info)

# Save results to a markdown file
with open('./arxiv_papers.md', 'w') as f:
    for result in results:
        f.write(result + "\n")

print("Data saved to arxiv_papers.md")
