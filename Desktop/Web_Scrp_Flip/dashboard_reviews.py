#!/usr/bin/python
# -*- coding: utf-8 -*-

"""signup.py: Automated Functionality Testing for Vendor Signup in Joobool"""
from pip._internal.utils import logging

__author__ = "Anurag Singh Kushwah, Chetan Gujrathi, Pragati Gupta"
__copyright__ = "Copyright 2019, Techrefic Technologies Pvt Ltd"
__credits__ = ["Abhishek Sikarwar", "Anurag Singh Kushwah", "Chetan Gujrathi", "Faizan Ali", "Pragati Gupta"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Chetan Gujrathi"
__email__ = "chetan.gujrathi@techrefic.com"
__status__ = "Staging"
# from automated_tests.config import *
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

import logging, datetime, time, inspect
logfile = "C:\\Users\\Professional Comp 1\\Desktop\\joobool_automation.log"
logformat = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'


logging.basicConfig(filename=logfile)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter(logformat)
handler.setFormatter(formatter)


class local_login():
    def __init__(self):
        """ Class to test Local Login Functionalities """
        self.logger = logging.getLogger("vendor local login")
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(handler)
        self.class_name = self.__class__.__name__

    def login(self, driver):
        try:
            frame = inspect.currentframe()
            fun_name = inspect.getframeinfo(frame).function
            print("fun_name : ",fun_name)
            driver.get("http://beta.joobl.tk/")
            time.sleep(2)
            driver.find_element_by_id("open_SignIn").click()
            time.sleep(4)
            driver.find_element_by_id("Login_vendor").click()

            time.sleep(4)
            input1 = driver.find_element_by_xpath('//*[@id="input-v4"]')
            input1.clear()
            self.logger.info("Entering email")
            input1.send_keys("demo19@gmail.com")
            time.sleep(2)
            input2 = driver.find_element_by_xpath('//*[@id="input-v7"]')
            input2.clear()
            self.logger.info("Entering Password")
            input2.send_keys("De12345678")
            time.sleep(2)
            driver.find_element_by_xpath('//*[@id="VENDOR_LOG_IN_BTN"]').click()
            time.sleep(2)



        except Exception as e:
            self.logger.fatal("Exception in Login : "+str(e))
            pass

class Reviews():
    def __init__(self):
        """ Class to test vendor home  Functionalities """
        self.logger = logging.getLogger("Reviews")
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(handler)
        self.class_name = self.__class__.__name__

    def review_hyperlink(self, driver):
        frame = inspect.currentframe()
        fun_name = inspect.getframeinfo(frame).function
        print("fun_name : ", fun_name)
        try:
            driver.find_element_by_xpath('//*[@id="chld-nv-wrapper"]/div/div/div[2]/div[6]/a').click()
        except Exception as e:
            self.logger.fatal('clicking on reviews' + str(e))
            pass
        try:
            time.sleep(2)
            self.logger.info("clicking on option")
            driver.find_element_by_id('icon-1').click()
            time.sleep(2)
            self.logger.info("Deselecting on option")
            driver.find_element_by_id('icon-1').click()
            time.sleep(3)
        except Exception as e:
            self.logger.fatal('Exception in clicking reviews:' + str(e))
            pass

        try:
            for i in range(1, 12):
                try:
                    time.sleep(5)
                    self.logger.info("clickin on sub options")
                    driver.find_element_by_id('nav_txt_' + str(i) + '').click()
                    time.sleep(2)
                except Exception as e:
                    self.logger.fatal('Exception not ckicking on suboptions:' + str(e))
                    pass
        except Exception as e:
            self.logger.fatal('Exception in clicking All:' + str(e))
            pass

    def All(self, driver):
        frame = inspect.currentframe()
        fun_name = inspect.getframeinfo(frame).function
        print("fun_name : ", fun_name)
        driver.get("http://beta.joobl.tk/vendor/reviews/all")
        try:
            check1 = driver.find_element_by_xpath('//*[@id="nav_txt_1"]/span').text
            print(check1)
            self.logger.info("reading count of All")
            try:
                if int(check1) >= 1:

                    for i in range(2,10):

                            try:
                                time.sleep(3)
                                driver.find_element_by_xpath('//*[@id="lft-cntnt-1"]/div['+str(i)+']/div/div/div/div[3]/a').click()
                                self.logger.info("clicked on button")
                                time.sleep(5)
                            except Exception as e:
                                self.logger.fatal("Exception in clicking " + str(e))
                                pass

                            try:
                                elem1 = driver.find_element_by_xpath('//*[@id="lft-cntnt-1"]/div['+str(i)+']/div/div/div/div[3]/a').text
                                print(elem1)
                                time.sleep(2)
                                if elem1 == "Approve":
                                    try:
                                        driver.find_element_by_id("deleteAll").click()
                                        self.logger.info("Approved")
                                        time.sleep(3)
                                        elem1 = driver.find_element_by_xpath('//*[@id="lft-cntnt-1"]/div[' + str(i) + ']/div/div/div/div[3]/a').text
                                        if
                                    except Exception as e:
                                        self.logger.fatal("Exception in clicking Approve:"+str(e))
                                        pass

                                elif elem1 == "Reply":
                                    time.sleep(2)
                                    try:
                                        driver.find_element_by_name("write_a_reply").send_keys(
                                            "Reviews are the better option to select an element")
                                        time.sleep(2)
                                        self.logger.info("writing a reply done")
                                    except Exception as e:
                                        self.logger.fatal("Exception in writing reply:"+str(e))
                                        pass
                                    try:
                                        time.sleep(5)
                                        driver.find_element_by_xpath('//*[@id="replyForm"]/div[2]/div[1]/button').click()
                                        self.logger.info("clicked on submit button of reply")
                                    except Exception as e:
                                        driver.get("http://beta.joobl.tk/vendor/reviews/all")
                                        self.logger.fatal("clicking on submit button failed:"+str(e))
                                        pass
                                    try:
                                        time.sleep(5)
                                        driver.find_element_by_css_selector('#Thank-popup > div > div > div > button').click()
                                        self.logger.info("clicked on cross option for exit ")
                                    except Exception as e:
                                        self.logger.fatal("Exception in clicking on cross button"+str(e))
                                        pass
                                if elem1 == "Replied":
                                    try:
                                        time.sleep(5)
                                        driver.find_element_by_class_name("needpopup_remover").click()
                                        time.sleep(5)

                                    except Exception as e:
                                        self.logger.fatal("Exception in clicking in replied ppup:"+str(e))
                                        pass



                            except Exception as e:
                                self.logger.fatal("Exception in cliking left content:" + str(e))
                                pass
                                self.logger.info("All option hyperlink check button clicked")
                    try:
                        time.sleep(3)
                        driver.find_element_by_class_name('report-but').click()
                        self.logger.info("clicked on report")
                    except Exception as e:
                        self.logger.fatal("Exception in clicking report"+str(e))
                        pass
                    try:
                        time.sleep(5)
                        driver.find_element_by_xpath('//*[@id="reportForm"]/ul/li[4]/label').click()
                        self.logger.info("selected report option")
                    except Exception as e:
                        self.logger.fatal("Not selected the report option"+str(e))
                        pass
                    try:
                        time.sleep(3)
                        driver.find_element_by_xpath('//*[@id="reportForm"]/button').click()
                        self.logger.info("clicked on submit button of report")
                        driver.get("")
                    except Exception as e:
                        self.logger.fatal("Exception in clicking on submit button of report:"+str(e))
                        pass
                    try:
                        time.sleep(3)
                        driver.find_element_by_class_name('report-but archiveReview').click()
                        self.logger.info("clicked on Archive")
                    except Exception as e:
                        self.logger.fatal("Exception in clicking Archive"+str(e))
                        pass
                    try:
                        time.sleep(3)
                        driver.find_element_by_id("archiveReview").click()
                        self.logger.info("clicked on popup Archive")
                    except Exception as e:
                        self.logger.fatal("Exception in clicking on popup of Archive:"+str(e))
                        pass


            except Exception as e:
                self.logger.fatal('Exception in clicking buttons:' + str(e))
                pass

        except Exception as e:
            self.logger.fatal("Exception in working on All:" + str(e))
            pass


if __name__ == '__main__':
    chrome_options = webdriver.ChromeOptions()  # chrome options
    chrome_options.add_argument("--disable-infobars")
    # driver = webdriver.Chrome( chrome_options=chrome_options)
    # driver = webdriver.Chrome(executable_path=chromepath,chrome_options=chrome_options)
    driver = webdriver.Chrome(executable_path="C:\\Users\\Professional Comp 1\\Desktop\\chromedriver", chrome_options=chrome_options)
    driver.maximize_window()
    local_login().login(driver)
    # Reviews().review_hyperlink(driver)
    Reviews().All(driver)
