# Import libraries
import requests
import pandas as pd
from bs4 import BeautifulSoup

# Provide the url from where we take the data 
url = 'https://dotesports.com/dota-2/news/here-are-the-most-frequently-picked-heroes-throughout-the-history-of-the-international'

# create a request method from the url provided above
r   = requests.get(url, headers={'user-agent': 'Mozilla/5.0'})

# Variable which represents BeautifulSoup library 
soup = BeautifulSoup(r.content, 'html5lib')

# Variable which contains the data from website (inside HTML table element)
table = soup("table")

# Variable which contains data passed after reading from website
stats = pd.read_html(str(table))[0]

# Variable in the form of DataFrame to contain passed data 
dataFrame = pd.DataFrame(data = stats)

# Convert process from DataFrame into CSV file.
dataFrame.to_csv('data.csv')


