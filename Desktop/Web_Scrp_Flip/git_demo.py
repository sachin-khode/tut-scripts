import requests
from bs4 import BeautifulSoup

scrapeLink = 'https://www.flipkart.com/search?q=trousers&sid=clo%2Cvua%2Cmle%2Clhk&as=on&as-show=on&otracker=AS_QueryStore_OrganicAutoSuggest_3_8_na_na_na&otracker1=AS_QueryStore_OrganicAutoSuggest_3_8_na_na_na&as-pos=3&as-type=RECENT&suggestionId=trousers%7CMen%27s+Trousers&requestId=bcf3bdc7-28b2-4be9-b2e2-2a5c698eb6ab&as-searchtext=trousers'
page = requests.get(scrapeLink)

soup = BeautifulSoup(page.content, 'html.parser')


