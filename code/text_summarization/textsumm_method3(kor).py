import json
import os
import re
from langchain_openai import ChatOpenAI

class TextSummarizer:
    def __init__(self, api_key_path):
        self.api_key = self.load_api_key(api_key_path)
        self.llm_3 = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=self.api_key)
        self.llm_4 = ChatOpenAI(model="gpt-4", openai_api_key=self.api_key)

    @staticmethod
    def load_api_key(path):
        with open(path, 'r') as f:
            api_key = json.load(f)['OPENAI_API_KEY']
        os.environ['OPENAI_API_KEY'] = api_key
        return api_key
    
    @staticmethod
    def extract_5pages(text):
        end_marker = text.find('<p.6>')
        if end_marker != -1:
            return text[:end_marker]
        else:
            return text

    def extract_info(self, text):
        prompt = f"""
        주어진 텍스트에서 다음 정보를 추출하십시오:
        1. 이름 (팀 이름은 제외하고 나열된 모든 사람을 포함하십시오)
        2. 제목 (텍스트에 나타나는 전체 제목을 포함하십시오)
        3. 주요 주제 (목차에서 주요 주제만 포함하고, 하위 주제는 제외하십시오)

        주요 주제는 'TABLE OF CONTENTS', '목차 소개'와 같은 헤더 다음 섹션에서 추출해야 합니다.
        팀원의 이름이 여러 줄로 나뉘지 않도록 하십시오. 여러 명의 이름이 있을 경우 쉼표로 연결하십시오.
        제목이 잘리지 않도록 하고, 전체를 추출하십시오.
        주요 주제가 여러 줄로 나뉘지 않도록 하십시오. 여러 개의 주제가 있을 경우 쉼표로 연결하십시오.
        하위 주제나 부수적인 정보는 제외하고 주요 주제만 추출하십시오.
        다음 페이지 마커 '<p.'에서 멈추십시오.

        출력이 **절대로** <xml>로 시작하지 않도록 하십시오.
        출력에 추가적인 메모나 설명을 포함하지 마십시오.

        텍스트: {text}

        추출된 정보를 다음 형식으로 포맷하십시오:
        <subject>제목</subject>
        <team>팀원</team>
        <index>주요 주제</index>

        """

        response = self.llm_4.invoke(prompt)
        extracted_info = response.content.strip()

        # Extract main topics
        main_topics_match = re.search(r'<index>(.*?)</index>', extracted_info, re.DOTALL)
        if (main_topics_match):
            main_topics = main_topics_match.group(1).strip()
            main_topics_list = [topic.strip() for topic in main_topics.split(',')]
        else:
            main_topics_list = []

        return extracted_info, main_topics_list

    def remove_page(self, text):
        cleaned_text = re.sub(r'<p\.\d+>', '', text)
        return cleaned_text

    def divide_text(self, extracted_info, main_topics_list, text):
        prompt = f"""
        제공된 텍스트는 주요 주제에 따라 나누기 전에 특정 섹션을 제외해야 합니다.
        텍스트에서 다음 섹션을 제외하십시오:
        {extracted_info}

        제외한 후, 제공된 텍스트를 주요 주제에 따라 나누십시오.

        텍스트는 정확히 {len(main_topics_list)}개의 섹션으로 나누어야 하며, 각 섹션은 "==================================="로 구분되어야 합니다.

        아래 제공된 주요 주제를 분할의 기준으로 사용하십시오:
        {main_topics_list}

        텍스트는 수정하지 않고 유지되어야 합니다.
        텍스트는 주요 주제의 개수에 따라 나누기만 하십시오.

        텍스트: {text}
        """

        response = self.llm_4.invoke(prompt)
        divided_text = response.content.strip()
        return divided_text

    def summarize(self, text):
        prompt = f"""
        제공된 텍스트는 "==================================="로 구분된 여러 파트로 나뉘어져 있습니다.
        각 파트를 별도로 요약하십시오.
        원본 입력 텍스트의 언어를 변경하지 마십시오.
        입력 텍스트와 동일한 언어로 요약하십시오.
        요약의 목적은 GitHub 프로젝트를 위한 포괄적인 README를 작성하는 것입니다.

        각 요약이 해당 파트의 핵심 포인트와 중요한 정보를 정확하게 담도록 하십시오.
        각 파트에서 주요 주제가 처음 언급된 후 반복되는 언급은 생략하고, 간결하고 정보가 풍부한 방식으로 요약하십시오.
        프로젝트와 관련된 핵심 세부 사항과 결과에 초점을 맞추십시오.

        "===================================" 구분자를 제거하거나 수정하지 마십시오.
        요약을 위해 다음 주요 주제를 사용하십시오:
        {main_topics_list}
    
        요약은 다음 형식을 따라야 합니다:
        ===================================
        대주제
        - 하위 주제: 상세 내용
        - 하위 주제: 상세 내용
        ===================================
        대주제
        - 하위 주제: 상세 내용
        - 하위 주제: 상세 내용
        ===================================

        다음은 잘 구조화된 요약의 예입니다:

        ===================================
        프로젝트 소개
        - 프로젝트 배경: 정제된 '대량의 데이터'를 사용하여 모델 선택 범위를 넓히고자 함.
        - 프로젝트 목표: 사용자 데이터, 웹툰 데이터를 이용해 개인화된 추천 시스템 개발. 사용자와 아이템의 상호작용 데이터에 맞는 알고리즘 탐색 및 적용.
        ===================================
        데이터 수집 및 전처리
        - 데이터 소스 설명: 다양한 데이터 소스 사용. 데이터 전처리 과정 설명.
        - 데이터 처리: 피봇 테이블 형식의 데이터 생성, 데이터 정제 및 전처리.
        ===================================

        텍스트: {text}
        """

        response = self.llm_4.invoke(prompt)
        result = response.content.strip()
        return result

    def tag_text(self, text):
        prompt = f"""
        텍스트의 각 부분을 다음 규칙에 따라 태그하십시오:
        - "-" 이전의 첫 번째 줄은 <main> 태그로 표시하십시오.
        - "-"로 시작하는 각 줄은 ":" 이전 부분을 <sub> 태그로, 이후 부분을 <content> 태그로 표시하십시오.
        - 텍스트에서 "-", "=", ":" 문자를 제거하십시오.
        - 태그가 올바르게 닫히고 형식이 올바른지 확인하십시오.
        원본 입력 텍스트의 언어를 변경하지 마십시오.
        출력이 **절대로** <xml>로 시작하지 않도록 하십시오.

        태그된 텍스트의 예시는 다음과 같습니다:

        <main>대한민국의 계절</main>
        <sub>여름</sub> <content>요즘 한국의 여름은 매우 덥고 습한 편이다.</content>
        <sub>겨울</sub> <content>12월부터 온도가 급격히 떨어지며 건조해진다.</content>

        <main>대한민국의 휴일</main>
        <sub>추석</sub> <content>추석은 비교적 휴일 기간이 긴 편이다.</content>
        <sub>설날</sub> <content>설날에는 떡국을 먹으며, 가족들과 인사를 나눈다.</content> 
 
        텍스트:
        {text}
        """

        response = self.llm_4.invoke(prompt)
        return response.content.strip()

if __name__ == "__main__":
    api_key_path = "C:\\Users\\PC\\Desktop\\DoYoung\\DS\\비타민NLP_240701\\text_summarization\\openai_api_key.json"
    text_file_path = "C:\\Users\\PC\\Desktop\\DoYoung\\DS\\비타민NLP_240701\\text_summarization\\sample_data\\sample2.txt"
    
    with open(text_file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    
    summarizer = TextSummarizer(api_key_path)
    text5 = summarizer.extract_5pages(text)
    extracted_info, main_topics_list = summarizer.extract_info(text5)
    cleaned_text = summarizer.remove_page(text)
    divided_text = summarizer.divide_text(extracted_info, main_topics_list, cleaned_text)
    summarized_text = summarizer.summarize(divided_text)
    tagged_text = summarizer.tag_text(summarized_text)
    
    print(tagged_text)
