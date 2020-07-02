from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
from pymongo import MongoClient


db=MongoClient().scraping
detail_scr=db["detail_scr"]


chrome_options = Options()
chrome_options.add_argument('disable-notification')
rag={}
driver = webdriver.Chrome(executable_path='/home/sk-ji/Desktop/chromedriver_linux64/chromedriver',chrome_options= chrome_options)
driver.maximize_window()
driver.get('https://internshala.com/internships')
time.sleep(4)
#driver.find_element_by_class_name("view_detail_button")
acc=driver.find_elements_by_class_name('view_detail_button')
print(len(acc))
for i in range(1, len(acc)):

    res=driver.find_elements_by_class_name('view_detail_button')[i].get_attribute('href')
    print(res,'\n')
    driver.get(res)
    time.sleep(2)
    name=driver.find_element_by_class_name('profile_on_detail_page')
    rag["name"] = name.text

    time.sleep(2)
    location = driver.find_element_by_class_name('link_display_like_text')
    rag["location"] = location.text
    time.sleep(2)

    about = driver.find_element_by_xpath('//*[@id="internship_list_container"]/div[2]/h5[1]')
    rag["about"] = about.text
    time.sleep(2)

    detail = driver.find_element_by_xpath( '//*[@id="internship_list_container"]/div[2]/div[1]')
    rag["detail"] = detail.text
    time.sleep(2)

    about2 = driver.find_element_by_xpath('//*[@id="internship_list_container"]/div[2]/h5[2]')
    rag[" about2"] = about2.text


    detail1 = driver.find_element_by_xpath('//*[@id="internship_list_container"]/div[2]/div[2]')
    rag["detail1"] = detail1.text
    time.sleep(2)

    about3 = driver.find_element_by_xpath('// *[ @ id = "whoCanApplyHeading"]')
    rag["about3"] = about3.text

    detail2 = driver.find_element_by_xpath('// *[ @ id = "internship_list_container"] / div[2] / div[5]')
    rag[" detail2"] = detail2.text


    detail_scr.insert_many(rag)


    driver.back()
