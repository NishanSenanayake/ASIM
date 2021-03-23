# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 12:29:42 2021

@author: 14198
"""

from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd
from selenium.webdriver.chrome.options import Options
import time


springer_link='https://link.springer.com'
article_list=[]
image_info = []
caption_info=[]
article_info=[]

options = Options()
options.headless = True
driver = webdriver.Chrome(executable_path="D:\chrome_driver\chromedriver.exe", chrome_options=options)

number_of_pages = 2


for x in range(1, (number_of_pages+1)):
    home_page= "https://link.springer.com/search/page/"+str(x)+"?facet-discipline=%22Materials+Science%22&showAll=false"
    driver.get(home_page)
    html_home = driver.page_source 
    soup_home = BeautifulSoup(html_home, "html.parser") 
    full_list=soup_home.find_all("li")
    for block in full_list:
        caption_content=block.find('p',class_='content-type')
        if caption_content is not None:
            article_caption= caption_content.get_text(strip=True)
            if (article_caption=='Article'):
                article_link_content= block.find("a", class_='title', href=True)
                article_link= article_link_content['href']
                article_name=article_link_content.text
                print(article_name)
                full_article_link=springer_link+article_link
                driver.get(full_article_link)
                article_html = driver.page_source
                soup_article = BeautifulSoup(article_html, "html.parser") 
                images=soup_article.find_all("a", class_='c-article-section__figure-link')
                img_captions= soup_article.find_all("div",class_="c-article-section__figure-description")
                for cap in img_captions:
                    caption=cap.find("p").get_text()
                    caption_info.append(caption)
                    article_info.append(article_name)
                for a in images:
                    image_tag = a.findChildren("img")
                    image_info.append((image_tag[0]["src"]))
                print('number of images = '+str(len(images)))          
          
    
caps= pd.DataFrame(caption_info)
img= pd.DataFrame(image_info)
art=pd.DataFrame(article_info)
final= pd.concat([art,caps,img], axis=1)  
final.columns =['Article', 'Caption', 'Image_link'] 
    
    
    
    
    
    
    
    
    
    
    





for link in soup_home.find_all("a", class_='title', href=True):
    
    article_list.append(article_link)






https://link.springer.com/


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
    




