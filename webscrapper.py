# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 12:29:42 2021

@author: 14198
"""



from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
from bs4 import BeautifulSoup
import pandas as pd



from selenium import webdriver
url= "https://link.springer.com/article/10.1007/s40192-020-00195-z"

driver= webdriver.Chrome(executable_path="D:\chrome_driver\chromedriver.exe")
driver.get(url)
html = driver.page_source 

soup = BeautifulSoup(html, "html.parser") 

images=soup.find_all("a", class_='c-article-section__figure-link')
captions= soup.find_all("div",class_="c-article-section__figure-description")

image_info = []
caption_info=[]


for cap in captions:
    caption=cap.find("p").get_text()
    caption_info.append(caption)
for a in images:
    image_tag = a.findChildren("img")
    image_info.append((image_tag[0]["src"]))
    

caps= pd.DataFrame(caption_info)
img= pd.DataFrame(image_info)
final= pd.concat([caps,img], axis=1)


