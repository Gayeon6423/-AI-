## 금융 특화 감정분석 모델과 딥러닝 시계열 예측 모델을 활용한 코스피 지수 예측
### Kospi Index Prediction using a Financial-specific Sentiment Analysis and Deep Learning-based Time Series Prediction Model

---

## 1. 초록 및 키워드

---

- Abstract : This paper presents a methodology for predicting the KOSPI index using a news data-based sentiment analysis model and a deep learning-based time series prediction model. The closing price of the KOSPI index was used as a target variable, and macroeconomic indicators such as the gold price and market sentiment indicators such as sentiment scores were used as independent variables. We collected and preprocessed the KOSPI-related news data and used them in calculating the sentiment score by using the title or the summarized article. Subsequently, the KLUE-BERT model-based sentiment score by date and the KoFinBERT model-based sentiment score by date were extracted. LSTM, GRU, CNN-LSTM, and CNN-GRU were used as time series prediction models. As a result of conducting an experiment by combination of variables and models, the best performance was achieved when KLUE-BERT is applied on the summarized article and the CNN-GRU model were used

- keyword : Deep Learning, BERT, Sentiment Analysis, LSTM, Kospi Index Prediction

## 2. 논문 투고 관련 정보

---

- Published : Journal of the Korean Institute of Industrial Engineers - Vol.50, No.4, pp.240-250(11 pages)
- ISSN: 1225-0988 (Print) 2234-6457 (Online)
- Print publication date 15 Aug 2024
- Received 12 Jan 2024 Accepted 05 Feb 2024


## 3. 모델 구조

---

![KOSPI](https://github.com/user-attachments/assets/a226200f-0ce4-486a-ab3f-5cf87b2ad574)

