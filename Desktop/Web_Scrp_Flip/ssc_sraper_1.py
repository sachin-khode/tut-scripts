from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
# from pymongo import MongoClient
# db22 = MongoClient().mydb22

# ssc_ppr =db22['ssc_ppr']

chrome_options = Options()
ssc = {}
driver = webdriver.Chrome(executable_path='/home/sk-ji/Desktop/chromedriver_linux64/chromedriver',chrome_options= chrome_options)

driver.maximize_window()
time.sleep(2)

driver.get('https://www.flipkart.com/search?q=trousers&sid=clo%2Cvua%2Cmle%2Clhk&as=on&as-show=on&otracker=AS_QueryStore_OrganicAutoSuggest_3_8_na_na_na&otracker1=AS_QueryStore_OrganicAutoSuggest_3_8_na_na_na&as-pos=3&as-type=RECENT&suggestionId=trousers%7CMen%27s+Trousers&requestId=bcf3bdc7-28b2-4be9-b2e2-2a5c698eb6ab&as-searchtext=trousers')

time.sleep(2)

for i in range(1,3):
    listing = driver.find_elements_by_xpath("//*[@id='container']/div/div[3]/div[2]/div[1]/div[2]/div[12]/div/div/nav/a[2]")
#     print(listing)