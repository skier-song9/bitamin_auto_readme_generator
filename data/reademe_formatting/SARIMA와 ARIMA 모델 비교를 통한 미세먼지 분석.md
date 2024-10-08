
# SARIMA와 ARIMA 모델 비교를 통한 미세먼지 분석
(프로젝트 진행기간을 입력하세요. ####.##.## ~ ####.##.##)
### Team
김채연, 김아진, 문승민, 서윤, 허범현

## Table of Contents
- [프로젝트 배경 소개](#section_1)
- [시계열 모델 스터디](#section_2)
- [ARIMA & SARIMA](#section_3)
- [사용 데이터셋](#section_4)
- [시계열 데이터 분석의 전처리](#section_5)
- [ARIMA SARIMA 적용과 비교](#section_6)
- [분석결과](#section_7)
- [프로젝트 인사이트 공유](#section_8)
<br>
<a name='section_1'></a>

## 프로젝트 배경 소개

#### 프로젝트 배경

- WHO에 따르면, 대기오염으로 인한 조기사망자는 2010년 36명에서 2060년에는 107명으로 증가할 전망이며, 이를 분석하기 위해 시계열 분석 프로젝트를 계획함.

<a name='section_2'></a>

## 시계열 모델 스터디

<a name='section_3'></a>

## ARIMA & SARIMA

#### ARIMA 모델

- AR, MA, 차분을 통해 비정상성을 정상성으로 변환하는 시계열 모델.

#### SARIMA 모델

- 계절성을 포함한 ARIMA 모델.

<a name='section_4'></a>

## 사용 데이터셋

#### 사용 데이터셋

- 대기환경지수, 미세먼지, 미산화탄소 등의 데이터를 사용.

<a name='section_5'></a>

## 시계열 데이터 분석의 전처리

#### 데이터 전처리

- 불필요한 칼럼 제거, '년'과 '월' 추출, 데이터 필터링, 그룹화, 이동 평균 및 이동 표준편차 계산, 정상성 검정, 시각화.

#### 시각화 결과

- 원본 데이터의 월별 변화, 장기적인 추세, 계절성, 잔차 분석.

#### 차분을 통한 정상성 확보

- 정상성 확보를 위해 비정상적인 시계열 데이터를 차분 처리.

<a name='section_6'></a>

## ARIMA SARIMA 적용과 비교

#### ARIMA

- 최적 파라미터 탐색 및 미래값 예측.

#### SARIMA

- 계절성을 추가한 모델 적용 및 미래값 예측.

<a name='section_7'></a>

## 분석결과

#### 파라미터 유의성

- 대부분의 파라미터는 통계적으로 유의미하지만, 잔차의 정규성 및 이분산성으로 인해 불확실성이 존재.

<a name='section_8'></a>

## 프로젝트 인사이트 공유

#### 장기 추세

- 서울시의 미세먼지 농도는 감소 추세를 보이며, 이는 대기 질 개선 정책의 효과로 예상됨.

#### 계절성

- 미세먼지 농도는 겨울철과 봄철에 주기적으로 높아지는 경향이 있음.

#### 미래 예측

- 2024년의 예측 결과는 현재의 감소 추세를 유지할 것으로 보이며, 단기 변동성은 유지될 것임.

#### 정책적 시사점

- 계절적 요인에 맞춘 대기 오염 저감 대책을 강화할 필요가 있음.

