
# Netflix Stock Price Prediction with News Topic & Sentiment
(프로젝트 진행기간을 입력하세요. ####.##.## ~ ####.##.##)
### Team
송규헌, 권도영, 이태경, 김서윤, 한진솔

## Table of Contents
- [INTRODUCTION](#section_1)
- [DATA PREPROCESSING](#section_2)
- [MODELING](#section_3)
- [CONCLUSIONS AND LIMITATIONS](#section_4)
<br>
<a name='section_1'></a>

## INTRODUCTION

#### 프로젝트 배경

- 뉴스가 주가 변동에 미치는 영향을 탐구하고 뉴스 감성분석 및 토픽 모델링 결과를 사용하여 주가 예측 가능성을 연구 기존의 단기 예측을 넘어 장기적인 추세를 고려한 주가 예측 모델 구현

#### 프로젝트 목표

- 뉴스 데이터를 활용한 주가 예측 모델의 최적화 및 효율적인 파라미터 선정 LSTM GRU Transformer 모델 비교 및 튜닝

#### 데이터 수집

- FinanceDataReader와 Stock News API를 사용하여 2018년 1월 2일부터 2023년 12월 29일까지의 NETFLIX 핀터레스트 메타플렛폼스 스포티파이 주가 데이터 및 뉴스 데이터 수집

<a name='section_2'></a>

## DATA PREPROCESSING

#### 파생 변수 생성

- 변화율 이동 평균 등의 파생 변수 생성 2018년 데이터를 추가로 수집해 파생변수 생성에 활용

#### 지표 추가

- Technical Analysis Library를 사용하여 Bollinger Bands Keltner Channel 등 37개의 금융 지표 추가

#### 유사 주식 종가 추가

- 핀터레스트 메타플렛폼스 스포티파이의 해당 기간 종가를 feature로 추가

#### 다중공선성 제거

- 주가 변화와 관련된 기본 지표를 유지하고 상관관계가 높은 보조 지표를 제거

#### 뉴스 토픽 및 감성 파생 변수 생성

- 뉴스 토픽과 감성 데이터를 Label Encoding과 OneHot Encoding으로 변환하여 추가

#### 데이터셋 구성

- 주식 데이터셋(stockOnly-df)과 뉴스 포함 데이터셋(total-df) 구성

<a name='section_3'></a>

## MODELING

#### 시계열 분석

- 고정된 Window Size로 연속된 데이터 입력 및 예측

#### 모델 정의

- LSTM GRU Transformer 모델 정의 LSTM은 기존 주가 예측에 주로 사용됨 GRU는 LSTM의 복잡성을 단순화 Transformer는 NLP에서 좋은 성능을 보여 주가 예측에 적용

#### 모델 비교

- 예측 대상 변수(Close vs ID-ROC) 뉴스 포함 여부 모델 종류(LSTM vs GRU vs Transformer) 비교

<a name='section_4'></a>

## CONCLUSIONS AND LIMITATIONS

#### 평가 기준

- 예측 평가 지표로 RMSE 사용 Close 종가 예측보다 ROC 예측이 추세 반영이 잘됨

#### 모델 결과

- LSTM GRU Transformer 각각의 모델 결과 비교 LSTM이 평균 손실값이 가장 작고 Transformer가 가장 큼

#### 최적 파라미터

- LSTM (stock-only seq 30 batch 64 avg error 1.85%) GRU (total seq 30 batch 128 avg error 1.66%) Transformer (total seq 30 batch 64 avg error 1.62%)

#### 한계점

- 예측 성능 평가의 어려움 예상치 못한 사건의 발생으로 인한 예측의 어려움 뉴스 데이터의 정확도 문제 등

