### 1. Import library
# 데이터 처리 모듈
# !pip install konlpy
import pandas as pd
import copy
import re
import math
import json
import requests
import urllib.request
from tqdm import tqdm
import datetime
import time
import os
# 텍스트 관련 모듈
from konlpy.tag import Okt
t = Okt()
okt = Okt()
from konlpy.tag import *
import nltk
# 구글 드라이브 마운트
from google.colab import drive
drive.mount("/content/drive")


### 2. Load Data
class LoadGoogleDriveData():
  def __init__(self, data = None):
    self.data = data

  def loadData(self, file_path: str, file_name_extension,
               columnTF: bool, unicode: str) -> pd.DataFrame():
    self.data = pd.read_csv(os.path.join(file_path + file_name_extension),
                            index_col = columnTF,
                            sep = ",",
                            na_values = "NaN",
                            encoding = unicode)
    return self.data

  def loadTxTData(self, file_path: str, file_name_extension,
               columnTF: bool, unicode: str) -> pd.DataFrame():
    self.data = pd.read_csv(os.path.join(file_path + file_name_extension),
                            index_col = columnTF,
                            sep = "|",
                            na_values = "NaN",
                            encoding = unicode)
    return self.data

  def loadExcelData(self, file_path: str, file_name_extension,
               columnTF: bool) -> pd.DataFrame():
    self.data = pd.read_excel(os.path.join(file_path + file_name_extension),
                              index_col = columnTF)
    return self.data

  # 용량이 큰 csv 파일 읽어오기(fopen - fread와 유사한 방식)
  def loadDataWithChunking(self, file_path: str, file_name_extension,
                           chunking_row_num: int, columnTF: bool, unicode: str) -> pd.DataFrame():
    chunkdata = pd.read_csv(os.path.join(file_path + file_name_extension),
                            chunksize = chunking_row_num,
                            index_col = columnTF,
                            sep = ",",
                            na_values = "NaN",
                            encoding = unicode)
    self.data = list(chunkdata)
    self.data = pd.concat(self.data)

    return self.data

mountInstance = LoadGoogleDriveData()

## load stopwords data
with open('/content/drive/MyDrive/산업 AI 캡스톤/DATA/stopwords.txt', 'r', encoding='utf-8') as f:
    stopwords = f.readlines() # 파일을 읽어서 각 줄을 리스트의 요소로 저장(줄바꿈 문자로 저장)
stopwords = [x.replace('\n','') for x in stopwords] # stopword 파일의 줄바꿈 문자 제거

## load numeric data
total_kospi = mountInstance.loadData(
            file_path = '/content/drive/MyDrive/산업 AI 캡스톤/DATA/Stock Index Data/',
            file_name_extension = "total_kospi.csv",
            unicode = 'utf-8-sig', columnTF = False)
total_kosdaq = mountInstance.loadData(
            file_path = '/content/drive/MyDrive/산업 AI 캡스톤/DATA/Stock Index Data/',
            file_name_extension = "total_kosdaq.csv",
            unicode = 'utf-8-sig', columnTF = False)

### 3. Filtering News Data
def filtering_news_data(start_date, end_date, keyword):
    for date in pd.date_range(start_date, end_date, freq='M'):
        # Read the CSV file
        news_data = mountInstance.loadData(
            file_path = '/content/drive/MyDrive/산업 AI 캡스톤/DATA/Original_News_Data/',
            file_name_extension = f"경제면_금융섹터_기사({date.strftime('%Y%m')}).csv",
            unicode = 'utf-8-sig', columnTF = False)

        # Drop null values
        news_data = news_data.dropna()

        # Filtering news : 코스피,코스피지수,KOSPI,kospi,Kospi,코스피200,Kospi지수,KOSPI지수,KOSPI200,kospi200 행 필터링
        news_data = news_data[news_data['content'].str.contains('코스피|코스피지수|KOSPI|kospi|Kospi|코스피200|Kospi지수|KOSPI지수|KOSPI200|kospi200')]

        # Generate the file name
        file_name = f"/content/drive/MyDrive/산업 AI 캡스톤/DATA/Filtering_News_Data/Kospi_Filtering_News_Data/news_{date.strftime('%Y%m')}_{keyword}.csv"

        # Save the filtered data to a new CSV file
        news_data.to_csv(file_name, encoding='utf-8-sig', index=False)

        # Display the first few rows of the filtered data
        print(f"First few rows of {file_name}, " "Rows number : ", len(news_data) )
        print()


### Filtering : Kospi
start_date = '2022-01'
end_date = '2022-03'
keyword = 'Kospi'
filtering_news_data(start_date, end_date, keyword)

### Filtering : Kosdaq
start_date = '2022-01'
end_date = '2022-03'
keyword = 'Kosdaq'
filtering_news_data(start_date, end_date, keyword)


### 4. Preprocessing News Data
def preprocess_news_data(start_date, end_date, keyword):
    for date in pd.date_range(start_date, end_date, freq='M'):
        # Read the CSV file
        file_path = f'/content/drive/MyDrive/산업 AI 캡스톤/DATA/Filtering_News_Data/{keyword}_Filtering_News_Data/'
        file_name = f"news_{date.strftime('%Y%m')}_{keyword}.csv"
        news_data = mountInstance.loadData(file_path=file_path, file_name_extension=file_name, unicode='utf-8-sig', columnTF=False)

        # Cleaning
        regex = r'[^\w\s]'
        news_data['clean_content'] = news_data['content'].apply(lambda x: re.sub(regex, '', str(x)))

        # Tokenization & Pos Tagging
        pos_tag = []
        for _, row in tqdm(news_data.iterrows()):
            news_text = row['clean_content']
            tokens_ko = t.pos(news_text)
            pos_tag.append(tokens_ko)

        # Normalization
        normalization_li = []
        for pos in pos_tag:
            in_li = []
            for ele in pos:
                if ele[1] in ['Josa', 'Suffix']:
                    continue
                in_li.append(ele[0])
            normalization_li.append(in_li)

        # Stopword Removal
        tokens = normalization_li
        token_stop = []
        for token in tokens:
            in_li = []
            for tok in token:
                if tok not in stopwords:
                    in_li.append(tok)
            token_stop.append(in_li)

        # Data save
        df_li = []
        for tokens in token_stop:
            token = ' '.join(tokens)
            df_li.append(token)

        df = pd.DataFrame(df_li).rename(columns={0: 'preprocess_context'})
        news_data = pd.concat([news_data, df], axis=1)

        # Generate the file name
        file_name = f"/content/drive/MyDrive/산업 AI 캡스톤/DATA/Preprocessing_News_Data/{keyword}_Preprocessing_News_Data/news_preprocess_{date.strftime('%Y%m')}_kospi.csv"

        # Save the preprocessed data to a new CSV file
        news_data.to_csv(file_name, encoding='utf-8-sig', index=False)

## 4.1 Preprocessing : Kospi
start_date = '2020-01'
end_date = '2023-07'
keyword = 'Kospi'
preprocess_news_data(start_date, end_date, keyword)

## 4.2 Preprocessing : Kosdaq
start_date = '2020-01'
end_date = '2023-07'
keyword = "Kosdaq"
preprocess_news_data(start_date, end_date, keyword)

### 5. Preprocessing Numeric Data

