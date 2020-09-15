from selenium import webdriver
from selenium.webdriver.common.keys import Keys 
from bs4 import BeautifulSoup
import pandas as pd
import pyautogui
 
from datetime import datetime  
import time 
import re 


def search():
    keyword = input("키워드를 입력하세요 : ")

    path = "./chromedriver.exe"
    driver = webdriver.Chrome(path)

    driver.implicitly_wait(3)

    url = "https://m.search.naver.com/search.naver?where=m_news&sm=mtb_jum&query="


    driver.get(url)

    search_box = driver.find_element_by_id("nx_query")
    search_box.send_keys(keyword)
    pyautogui.press('enter')

    titles = driver.find_element_by_class_name('list_news') #전체 페이지
    title = titles.find_elements_by_class_name('news_wrap') #뉴스 별로
    return title 


def parsing(title ):
    import requests
    lst = [] #title
     
    for tit in title: 
        txt = tit.text.split('\n') 
        txt = txt[2] #제목만 남기기 
        lst.append(txt) 
        #기사 클릭 
        news = tit.find_element_by_class_name('news_tit').click()   
        

        #화면 전체를 가져와서 크기정보로 스크롤을 하기 위해서 body 태그 객체를 가져옴ㄴ
        #body= box.find_element_by_tag_name('body')
        #scroll_screen(5,body,driver)
        #print()
    print(lst) #제목만 든 리스트
 

def scroll_screen(cnt,body,driver):
    #화면 스크롤 작업
    scroll_pause_time = 0.5

    #반복적으로 진행
    i = 1
    #화면 길이 알아옴 .documentElement.scrollHeight 자바스크립트
    last_height = driver.execute_script("return document.documentElement.scrollHeight")
    print(last_height)

    while True:

        #스크롤
        body.send_keys(Keys.PAGE_DOWN)
        time.sleep(scroll_pause_time)
        #keys = 키보드 값  - 약속된 상수 값 , enter 키는 return 
        i += 1
        if i >= cnt:
            break
 
 
title = search()
parsing(title)