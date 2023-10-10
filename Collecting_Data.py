# -*- coding: utf-8 -*-
"""Collecting Data

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1TzqdDCofs5CJYCeAz4v5KzILYzQ3DqXH

## Timeline
### 1. Import library
### 2. Crawling News Data
### 3. Collecting Numeric Data

### 1. Import library
"""

# 모듈 설치
# !pip install konlpy
# !pip install -U finance-datareader

# 데이터 처리 모듈
import pandas as pd
import copy
import re
import math
import json
from tqdm import tqdm
import datetime
import time
import os
# 텍스트 관련 모듈
from konlpy.tag import Okt
okt = Okt()
from konlpy.tag import *
import nltk
# 데이터 수집 모듈
import requests
import urllib.request
from bs4 import BeautifulSoup
# 구글 드라이브 마운트
from google.colab import drive
drive.mount("/content/drive")

"""### 2. Crawling News Data

- 2.1 Generate date list
"""

def generate_date_list(startdate, enddate):
  # 시작일과 종료일을 datetime 형식으로 변환
  start = datetime.strptime(startdate, "%Y%m")
  end = datetime.strptime(enddate, "%Y%m")

  # 시작일부터 종료일까지의 날짜 리스트를 생성
  date_list = []
  current = start
  while current <= end:
    # 현재 날짜를 "%Y%m%d" 형식의 문자열로 변환하여 리스트에 추가
    date_list.append(current.strftime("%Y%m%d"))

    # 현재 날짜를 하루씩 증가
    current += relativedelta(days=1)

    # 다음 날짜의 월이 현재 날짜와 다른 경우, 다음 달의 첫 날로 변경
    if current.month != (current + relativedelta(days=1)).month:
      current = current.replace(day=1) + relativedelta(months=1) # relativedelta : 1달 더함

  return date_list

"""- 2.2 Generate news title, news url list"""

# generate news title, news url LIST function

def generate_news_list(date_list):
  news_date_li = []
    # 웹 페이지 크롤링 할 때 사용할 페이지 클래스 타입
  page_class_type = ['type06_headline', 'type06']

    # 날짜 리스트인 date_list를 반복하면서 뉴스 수집
  for date in tqdm(date_list):

    # 날짜별 뉴스 목록을 수집할 URL 생성
    url = f'https://news.naver.com/main/list.naver?mode=LS2D&sid2=259&mid=shm&sid1=101&date={date}&page=999'

    # HTTP 요청 헤더 설정
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36"}
    # HTTP GET 요청을 보내서 웹 페이지의 내용 가져옴
    raw = requests.get(url, headers=headers)

    # BeautifulSoup으르 사용해서 웹 페이지의 HTML 파싱
    soup = BeautifulSoup(raw.text, "html.parser")

    # 해당 날짜에 해당하는 뉴스 목록의 최대 페이지 수
    max_page = int(soup.select("#main_content > div.paging > strong")[0].text)

    # 각 페이지에서 뉴스 정보 수집
    news_page_li = []
    for page in tqdm(range(1, max_page + 1)):
      for ct in page_class_type:
        for news in range(1, 11): # 한 페이지당 10개의 기사

          # 페이지별로 뉴스 목록을 가져오는 URL 생성
          url = f'https://news.naver.com/main/list.naver?mode=LS2D&sid2=259&mid=shm&sid1=101&date={date}&page={page}'
          # HTTP GET 요청을 보내서 해당 페이지의 뉴스 목록을 가져옴
          raw = requests.get(url, headers=headers)
          soup = BeautifulSoup(raw.text, "html.parser")

          # HTML에서 뉴스 제목과 URL 추출
          news_all = soup.select(f"#main_content > div.list_body.newsflash_body > ul.{ct} > li:nth-child({news}) > dl > dt:nth-child(2) > a")
          news_dic = {}

          # 뉴스 정보가 있는 경우 news_dic 딕셔너리에 key값과 value 매핑
          if news_all:
              news_dic['title'] = news_all[0].text.strip()
              news_dic['url'] = news_all[0]['href']
              news_dic['pubdate'] = date

          # 뉴스 정보가 없는 경우 None값을 value로 설정
          else:
              news_dic['title'] = None
              news_dic['url'] = None
              news_dic['pubdate'] = None

          # 존재하는 뉴스 정보인 경우 뉴스 리스트에 추가
          if news_dic['title'] is not None and news_dic['url'] is not None:
              news_page_li.append(news_dic)

    # 해당 날짜의 모든 뉴스 정보를 저장한 리스트를 전체 뉴스 리스트에 추가
    news_date_li.extend(news_page_li)

  return news_date_li

"""- 2.3 Generate news dataframe"""

# generate news title, url, content, pundate DataFrame Function

def generate_news_content_df(news_date_li):
    news_page_li = []

    # 뉴스의 URL을 사용하여 뉴스 페이지에 접근하고 내용 스크랩
    for news in tqdm(news_date_li):
        url = news['url']

        # HTTP 요청 헤더 설정
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36"}

        # HTTP GET 요청을 보내서 뉴스 페이지의 내용을 가져옴
        raw = requests.get(url, headers=headers)
        soup = BeautifulSoup(raw.text, "html.parser")

        # HTML에서 뉴스 내용을 추출
        content = soup.select('#dic_area')

        # 뉴스 내용을 딕셔너리에 저장
        if content:
            news['content'] = content[0].text.strip()
        # 뉴스 내용 없으면 None값 저장
        else:
            news['content'] = None

        # 스크랩한 뉴스 정보를 리스트에 추가
        news_page_li.append(news)

    # 뉴스 정보를 데이터프레임으로 변환
    news_df = pd.DataFrame(news_page_li)
    return news_df

"""- Crawling Month News Data"""

# 2020년1월~2023년6월 데이터 수집
for startdate in range(202001,202306):
  enddate = str(startdate + 1)
  startdate = str(startdate)

  # Generate Date List
  date_list = generate_date_list(startdate, enddate)

  # Generate News title, URL
  news_date_li = generate_news_list(date_list)

  # Generate News Content
  news_df = generate_news_content_df(news_date_li)

  # Save News Data
  file_path =  f"/content/drive/MyDrive/산업 AI 캡스톤/DATA/Original_News_Data/경제면_금융섹터_기사({startdate}).csv"
  news_df.to_csv(file_path,encoding='utf-8-sig', index=False)

"""### 3. Collecting Numeric Data

- FinanceDataReader : kospi지수 및 미국주가지수 데이터 수집
"""

kospi = fdr.DataReader('KS11', '2020','2023.06.30').reset_index()
dji = fdr.DataReader('DJI', '2020','2023.06.30').reset_index()
kosdaq = fdr.DataReader('KQ11', '2020','2023.06.30').reset_index()
us500 = fdr.DataReader('US500', '2020','2023.06.30').reset_index()
ex_AM = fdr.DataReader('USD/KRW', '2020','2023.06.30').reset_index() #달러당 원화
ex_JP = fdr.DataReader('JPY/KRW', '2020','2023.06.30').reset_index() #엔화 원화
KOSPI_df= pd.DataFrame(columns=["Date"])
KOSDAQ_df = pd.DataFrame(columns=["Date"])
USI_df=pd.DataFrame(columns=["Date"])
EX_df = pd.DataFrame(columns=["Date"])

# 코스피 지수 데이터 데이터프레임
KOSPI_df["Date"] = kospi["Date"]
KOSPI_df["Kospi_open"] = kospi["Open"]
KOSPI_df["Kospi_high"] = kospi["High"]
KOSPI_df["Kospi_low"] = kospi["Low"]
KOSPI_df["Kospi_close"] = kospi["Close"]
KOSPI_df["Kospi_vol"] = kospi["Volume"]

# 코스닥 지수 데이터 데이터프레임
KOSDAQ_df["Date"] = kosdaq["Date"]
KOSDAQ_df["kosdaq_open"] = kosdaq["Open"]
KOSDAQ_df["kosdaq_high"] = kosdaq["High"]
KOSDAQ_df["kosdaq_low"] = kosdaq["Low"]
KOSDAQ_df["kosdaq_close"] = kosdaq["Close"]
KOSDAQ_df["kosdaq_vol"] = kosdaq["Volume"]

# 미국 주가지수 데이터 데이터프레임
USI_df["Date"] = dji["Date"]
USI_df["dji_open"] = dji["Open"]
USI_df["dji_high"] = dji["High"]
USI_df["dji_low"] = dji["Low"]
USI_df["dji_close"] = dji["Close"]
USI_df["dji_vol"] = dji["Volume"]
USI_df["us500_open"] = us500["Open"]
USI_df["us500_high"] = us500["High"]
USI_df["us500_low"] = us500["Low"]
USI_df["us500_close"] = us500["Close"]
USI_df["us500_vol"] = us500["Volume"]

# 환율 데이터 데이터프레임
EX_df["Date"] = ex_AM["Date"]
EX_df["ex_AM_open"] = ex_AM["Open"]
EX_df["ex_AM_high"] = ex_AM["High"]
EX_df["ex_AM_low"] = ex_AM["Low"]
EX_df["ex_AM_close"] = ex_AM["Close"]
EX_df["ex_JP_open"] = ex_JP["Open"]
EX_df["ex_JP_high"] = ex_JP["High"]
EX_df["ex_JP_low"] = ex_JP["Low"]
EX_df["ex_JP_close"] = ex_JP["Close"]

"""- Merge Data"""

kospi_middle_df = pd.merge(KOSPI_df,USI_df,on="Date")
kosdaq_middle_df = pd.merge(KOSDAQ_df,USI_df,on="Date")
kospi_total_df = pd.merge(kospi_middle_df,EX_df,on="Date")
kosdaq_total_df = pd.merge(kosdaq_middle_df,EX_df,on="Date")

"""- 한국은행 API : 이자율, 종합소비자물가지수, 종합부동산지수 데이터 수집"""

# Crollecting ECOS Data Method
private_api_key = "GVFDCZ2JSQ3FWKKD4HD8"

def EcosDownload(Statcode, Freq, Begdate, Enddate, Subcode1, Subcode2, Subcode3):
    url = 'http://ecos.bok.or.kr/api/StatisticSearch/%s/xml/kr/1/100000/%s/%s/%s/%s/%s/%s/%s/'%(private_api_key, Statcode, Freq, Begdate, Enddate, Subcode1, Subcode2, Subcode3)
    raw = requests.get(url)
    xml = BeautifulSoup(raw.text,'xml')

    # Pandas 데이터프레임으로 전환합니다.
    raw_data = xml.find_all("row")
    date_list = []
    value_list = []

    for item in raw_data:
        value = item.find('DATA_VALUE').text.encode('utf-8')
        date_str = item.find('TIME').text
        value = float(value)
        date_list.append(datetime.datetime.strptime(date_str,'%Y%m'))
        value_list.append(value)

    df = pd.DataFrame(index = date_list)
    df['value'] = value_list

    return df

"""- Collecting Data"""

interest = EcosDownload('722Y001', 'M', '202001', '202307', '0101000','','')
consumer = EcosDownload('901Y009', 'M', '202001', '202307', '0', '', '')
real_estate = EcosDownload('901Y064', 'M', '202001', '202307', 'P65B','','')

# reset index
interest = interest.reset_index()
consumer = consumer.reset_index()
real_estate = real_estate.reset_index()

# Preprocess data
data_m= pd.DataFrame(columns=["Date"])
data_m["Date"] = interest["index"]
data_m["ko_interest"] = interest["value"]
data_m["ko_consumer"] = consumer["value"]
data_m["ko_real_estate"] = real_estate["value"]
data_m.set_index('Date', inplace=True)
data_m = data_m.resample('D').ffill()
data_m.drop(data_m.index[-1], inplace=True)
data_m = data_m.reset_index()

"""- Merge Data"""

total_kospi = pd.merge(kospi_total_df,data_m,how='left', left_on='Date', right_on='Date')
total_kosdaq = pd.merge(kosdaq_total_df,data_m,how='left', left_on='Date', right_on='Date')