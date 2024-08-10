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
        Extract the following information from the provided text:
        1. Name (excluding the team name, include all people listed)
        2. Title (include the entire title as it appears in the text)
        3. Main Topics (only main topics from the Table of Contents, excluding subtopics)

        The Main Topics should be extracted from the section that follows headers such as 'TABLE OF CONTENTS', '목차 소개', or any similar variation.
        Ensure that team member's name are not split into multiple lines. Name should be connected by commas if there are more than one.
        Ensure that title is not cut off and is extracted in their entirety.
        Ensure that main topics are not split into multiple lines. Name should be connected by commas if there are more than one.
        Exclude any subtopics or secondary information while extracting main topics.
        Stop at the next page marker '<p.'.

        Ensure that the output **never** starts with <xml>
        Do not include any additional notes or explanations in the output.

        Text: {text}

        Format the extracted information as follows:
        <subject>title</subject>
        <team>team members</team>
        <index>main topics</index>

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
        The provided text contains specific sections that need to be excluded before dividing it according to the main topics.
        Exclude the following sections from the text:
        {extracted_info}

        After exclusion, divide the provided text according to the main topics.

        The text should be divided into exactly {len(main_topics_list)} sections, separated by "===================================".

        Use the main topics provided below as the indices for division:
        {main_topics_list}

        The text should remain unmodified.
        The text should only be partitioned according to the number of main topics.

        Text: {text}
        """

        response = self.llm_4.invoke(prompt)
        divided_text = response.content.strip()
        return divided_text

    def summarize(self, text):
        prompt = f"""
        The provided text is divided into several partitions separated by "===================================".
        Summarize each partition separately. 
        Do not change the language of the original input text. 
        Summarize in the same language as the input text.
        The purpose of the summaries is to create a comprehensive README for a GitHub project.

        Ensure that each summary accurately captures the key points and essential information of its respective partition.
        Summarize in a way that is concise and informative, omitting repetitive mentions of the main topic after the first mention in each partition.
        Focus on the key details and results relevant to the project.

        Do not remove or modify the "===================================" separators.
        Use the following main topics to guide the summaries:
        {main_topics_list}
    
        The summaries should follow this format:
        ===================================
        Main topic
        - subtopic: detailed contents
        - subtopic: detailed contents
        ===================================
        Main topic
        - subtopic: detailed contents
        - subtopic: detailed contents
        ===================================

        Here is an example of a well-structured summary:

        ===================================
        프로젝트 소개
        - 프로젝트 배경: 정제된 '대량의 데이터'를 사용하여 모델 선택 범위를 넓히고자 함.
        - 프로젝트 목표: 사용자 데이터, 웹툰 데이터를 이용해 개인화된 추천 시스템 개발. 사용자와 아이템의 interaction 데이터에 맞는 알고리즘 탐색 및 적용.
        ===================================
        데이터 수집 및 전처리
        - 데이터 소스 설명: 다양한 데이터 소스 사용. 데이터 전처리 과정 설명.
        - 데이터 처리: 피봇 테이블 형식의 데이터 생성, 데이터 정제 및 전처리.
        ===================================

        Text: {text}
        """

        response = self.llm_4.invoke(prompt)
        result = response.content.strip()
        return result

    def tag_text(self, text):
        prompt = f"""
        Tag each part of the text according to the following rules:
        - The first line before any "-" should be tagged as <main>.
        - Each line starting with "-" should have the part before ":" tagged as <sub> and the part after ":" tagged as <content>.
        - Remove "-", "=", and ":" characters from the text.
        - Ensure the tags are correctly closed and formatted.
        Do not change the language of the original input text.
        Ensure that the output **never** starts with <xml>

        Here is an example of how the tagged text should look:

        <main>대한민국의 계절</main>
        <sub>여름</sub> <content>요즘 한국의 여름은 매우 덥고 습한 편이다.</content>
        <sub>겨울</sub> <content>12월부터 온도가 급격히 떨어지며 건조해진다.</content>

        <main>대한민국의 휴일</main>
        <sub>추석</sub> <content>추석은 비교적 휴일 기간이 긴 편이다.</content>
        <sub>설날</sub> <content>설날에는 떡국을 먹으며, 가족들과 인사를 나눈다.</content> 
 
        Text:
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
