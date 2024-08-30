# 🍊 Auto ReadMe Generator
🗓️(프로젝트 진행기간 : 2024.06.26 ~ 2024.08.30)
### 🚀 Team
권도영, 김지원, 송규헌

## 🗒️Table of Contents
- [Environment](#section_0)
- [Launch Web](#section_1)
- [주제선정배경](#section_2)
- [서비스 플로우](#section_3)
- [INTRODUCTION](#section_4)
- [PDF PROCESSING](#section_5)
- [IMAGE CLASSIFIER](#section_6)
- [TEXT SUMMARIZATION](#section_7)
- [IMPLEMENTATION](#section_8)
- [CONCLUSION](#section_9)
<br>

<a name='section_0'></a>

## 🏠Enviornment
- Python 3.10+
- PyTorch 2.0+
- Java (openjdk11+)
- Flask
- install depedencies via requirments.txt
```pip install -r requirements.txt```
- place the openai_api_key.json in `/code/assets` directory. 
```json
{
    "OPENAI_API_KEY":"YOUR-OPENAI-API-KEY"
}
```

<a name='section_1'></a>

## 🛠️Launch Web
1. `cd /bitamin_auto_readme_generator/code`
2. `python server.py`
- service demonstration : [youtube/BITAMin 12th Conference - ProjectMaker (Auto README Generator)](https://youtu.be/auJGB8_11ww?si=kXZRxjeNRZkaTG9B)

<a name='section_1'></a>

## 🔷 주제선정배경

#### 프로젝트 필요성

- 발표용 PPT를 활용하여 README 파일을 자동으로 생성해 주는 프로그램의 필요성을 강조. 사용자는 PPT에 주요 내용이 포함되어 있어, 이를 통해 파일 작성 시간을 절약할 수 있음.
![04_3.jpg](/images_readme/04_3.jpg)

<br>

<a name='section_2'></a>

## 🔷 서비스 플로우

#### 서비스 흐름도

![flowchart](/images_readme/projectmaker_flowchart.png)

<br>
<a name='section_4'></a>

## 🔷 INTRODUCTION

#### 개요

- 파일 repository의 내용을 읽고 요약하여 README 파일을 생성하는 프로그램의 기능 및 구조를 설명. 이를 통해 유사 서비스와의 비교 및 분석을 수행.

<br>
<a name='section_5'></a>

## 🔷 PDF PROCESSING

#### 처리 개요

- PDF 파일에서 텍스트와 이미지를 추출하여 서버에 저장하고, OCR을 통해 슬라이드의 모든 문자 정보를 추출하는 과정 설명. 하지만 텍스트가 정돈되지 않는 한계가 있음.
![07_3.jpg](/images_readme/07_3.jpg)


<br>
<a name='section_6'></a>

## 🔷 IMAGE CLASSIFIER

#### 이미지 분류 목표

- 4개의 클래스로 이미지를 분류하여 프로젝트 관련 이미지만 웹에 전송하기 위한 모델을 구축. 다양한 이미지 유형을 전처리하여 성능을 평가.
![12_4.jpg](/images_readme/12_4.jpg)
![12_7.jpg](/images_readme/12_7.jpg)

<br>
<a name='section_7'></a>

## 🔷 TEXT SUMMARIZATION

#### 요약 방법

- LLM을 활용하여 청크 수준의 요약을 생성하고, 이를 병합하여 계층적 요약을 수행. 텍스트 누락을 최소화하고, 다양한 텍스트 분할 방법을 적용하여 요약의 신뢰성을 높임.

<br>
<a name='section_8'></a>

## 🔷 IMPLEMENTATION

#### 서비스 아키텍처

- AWS 서비스 기반으로 한 시스템 아키텍처의 개요와 시연을 제공.
![service-architecture](/images_readme/service_architecture.png)

<br>
<a name='section_9'></a>

## 🔷 CONCLUSION

#### 모델 한계 및 개선점

- OCR 모델의 인식 오류 및 데이터의 다양성 부족으로 인한 한계를 언급하며, 향후 개선 방향에 대한 논의. 기본 틀을 마련하여 지속적인 서비스 개선 가능성을 강조.


## 🔗Acknowledgements
#### Datasets
- Our project borrowed lots of datasets from [Chart-Classification-Using-CNN----Keras](https://github.com/devsonni/Chart-Classification-Using-CNN----Keras), [DQA-NET](https://arxiv.org/abs/1603.07396), [TNCR](https://arxiv.org/abs/2106.15322), [Chart2Text](https://github.com/JasonObeid/Chart2Text), [PubTabNet](https://arxiv.org/abs/1911.10683), and other 🍊BITAMin projects.