import json
import os
import re
from langchain_openai import ChatOpenAI

class TextSummarizer:
    def __init__(self, api_key_path):
        self.api_key = self.load_api_key(api_key_path)
        self.llm_3 = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=self.api_key)
        self.llm_4 = ChatOpenAI(model="gpt-4o-mini", openai_api_key=self.api_key)


    @staticmethod
    def load_api_key(path):
        with open(path, 'r') as f:
            api_key = json.load(f)['OPENAI_API_KEY']
        os.environ['OPENAI_API_KEY'] = api_key
        return api_key
    

    @staticmethod
    # STEP1: 첫 5페이지에서 팀원, 제목, 목차 추출
    def extract_5pages(text):
        end_marker = text.find('<p.06>')
        if end_marker != -1:
            text5 = text[:end_marker]
        else:
            text5 = text
        return text5


    def extract_info(self, text):
        prompt = f"""
        Extract the following information from the provided text:
        1. Name (excluding the team name, include all people listed)
        2. Title (include the entire title as it appears in the text)
        3. Main Topics (only main topics from the Table of Contents, excluding subtopics)

        The Main Topics should be extracted from the section that follows headers such as 'TABLE OF CONTENTS', '목차 소개', or any similar variation. 
        Only extract the topics from the page where the 'TABLE OF CONTENTS' or equivalent header is located, and stop when you encounter the next page marker (e.g., '<p.1>', '<p.2>').
        If the '|' character is present in the text, treat it as equivalent to a comma and replace it with a comma when outputting the main topics.
        
        Ensure that team member's name are not split into multiple lines. Name should be connected by commas if there are more than one.
        Ensure that title is not cut off and is extracted in their entirety.
        Ensure that main topics are not split into multiple lines. Main topics should be connected by commas if there are more than one.
        Exclude any subtopics or secondary information while extracting main topics.

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
        return extracted_info

    
    def extract_main_topics(self, extracted_info):
        main_topics_match = re.search(r'<index>(.*?)</index>', extracted_info, re.DOTALL)
        if main_topics_match:
            main_topics = main_topics_match.group(1).strip()
            main_topics_list = [topic.strip() for topic in main_topics.split(',')]
        else:
            main_topics_list = []

        return main_topics_list

    # STEP2: 전체 목차를 4등분한 후 각각의 변수에 저장
    def split_topics(self, main_topics_list):
        num_topics = len(main_topics_list)
        
        base_size = num_topics // 4
        remainder = num_topics % 4
        
        topics_list1 = []
        topics_list2 = []
        topics_list3 = []
        topics_list4 = []
        
        start_idx = 0
        
        sizes = [base_size + (1 if i < remainder else 0) for i in range(4)]
        
        topics_list1 = main_topics_list[start_idx:start_idx + sizes[0]]
        start_idx += sizes[0]
        
        topics_list2 = main_topics_list[start_idx:start_idx + sizes[1]]
        start_idx += sizes[1]
        
        topics_list3 = main_topics_list[start_idx:start_idx + sizes[2]]
        start_idx += sizes[2]
        
        topics_list4 = main_topics_list[start_idx:start_idx + sizes[3]]
        
        return topics_list1, topics_list2, topics_list3, topics_list4
    
    # STEP3: 전체 텍스트에서 각 파트에 따른 텍스트 추출
    def extract_info_by_topic(self, text, topics_list):
        """
        Extract text from the provided text that corresponds to the specified topics in the list.
        """
        topics_str = ', '.join(topics_list)

        prompt = f"""
        Extract the following sections from the provided text:
        Topics: {topics_str}

        Only include the text that corresponds to the specified topics. Do not include any unrelated sections or subtopics.
        Ensure that all text related to these topics is included without any omissions.
        Make sure to **include page markers** such as <p.01>, <p.02>, etc., in the extracted text without fail.
        Do not omit any content, especially page markers, or any part of the text directly related to the topics.


        Text: {text}
        """

        response = self.llm_4.invoke(prompt)
        extracted_info = response.content.strip()
        extracted_info = extracted_info.replace("```", "").strip("'")

        return extracted_info


    def main(self, text, topics_list1, topics_list2, topics_list3, topics_list4):
        topic_text1 = self.extract_info_by_topic(text, topics_list1) if topics_list1 else "no text"
        topic_text2 = self.extract_info_by_topic(text, topics_list2) if topics_list2 else "no text"
        topic_text3 = self.extract_info_by_topic(text, topics_list3) if topics_list3 else "no text"
        topic_text4 = self.extract_info_by_topic(text, topics_list4) if topics_list4 else "no text"
        
        return topic_text1, topic_text2, topic_text3, topic_text4
    
    #STEP4: 각 파트별 요약 진행
    def summarize(self, text, topics_list):
        prompt = f"""
        Do not change the language of the original input text. 
        Summarize in the same language as the input text.
        The purpose of the summary is to create a comprehensive README for a GitHub project.
        If multiple subtopics within the same main topic have the same name, combine their content into a single entry with all relevant details, and merge the page numbers into a single list. 
        Ensure that subtopics from different main topics are not merged.

        Ensure that the summary accurately captures the key points and essential information of each topic.
        Summarize in a way that is concise and informative, omitting repetitive mentions of the main topic after the first mention.
        Focus on the key details and results relevant to the project, with particular emphasis on the final model outcomes. 
        Ensure that the final model results are described with a high level of accuracy and detail, including specific metrics, performance evaluations, and any significant findings.
        For each subtopic, aim to provide a summary that is 1 to 2 sentences long. Focus on delivering a concise yet comprehensive overview, capturing the main points and essential details without unnecessary elaboration.
        When summarizing, only include the **first page** where each subtopic begins in the "pages used" list. For example, if a subtopic spans from page 7 to page 10, only include page 7 in the summary.

        Use the following main topic to guide the summary:
        {topics_list}
        
        The summary should be in the given format without requiring any specific markdown style or additional formatting.
        The summary should follow this format:
        Main topic
        - subtopic1: detailed contents (pages used: first page only)
        - subtopic2: detailed contents (pages used: first page only)
            
        Here is an example of a well-structured summary:
        프로젝트 소개
        - 프로젝트 배경: 정제된 '대량의 데이터'를 사용하여 모델 선택 범위를 넓히고자 함. (3)
        - 프로젝트 목표: 사용자 데이터, 웹툰 데이터를 이용해 개인화된 추천 시스템 개발. 사용자와 아이템의 interaction 데이터에 맞는 알고리즘 탐색 및 적용. (7)
        
        데이터 수집 및 전처리
        - 데이터 소스 설명: 다양한 데이터 소스 사용. 데이터 전처리 과정 설명. (11)
        - 데이터 처리: 피봇 테이블 형식의 데이터 생성, 데이터 정제 및 전처리. (13)

        Text: {text}
        """

        response = self.llm_4.invoke(prompt)
        result = response.content.strip()
        return result
    
    def generate_summaries(self, topic_text1, topic_text2, topic_text3, topic_text4, topics_list1, topics_list2, topics_list3, topics_list4):

        topic1_summ = self.summarize(topic_text1, topics_list1) if topic_text1 != "no text" else ""
        topic2_summ = self.summarize(topic_text2, topics_list2) if topic_text2 != "no text" else ""
        topic3_summ = self.summarize(topic_text3, topics_list3) if topic_text3 != "no text" else ""
        topic4_summ = self.summarize(topic_text4, topics_list4) if topic_text4 != "no text" else ""

        return topic1_summ, topic2_summ, topic3_summ, topic4_summ
    
    #STEP5: 각 파트별 요약본 merge
    def concat_summ(self, *args):
        return '\n'.join(args)
    
    #STEP6: 마크다운 형태로 변환
    def tag_text(self, text):
        prompt = f"""
        Tag each part of the text according to the following rules:
        - The first line before any "-" should be tagged as <main>.
        - Each line starting with "-" should have the part before ":" tagged as <sub> and the part after ":" tagged as <content>.
        - Remove "-", "=", and ":" characters from the text.
        - After each <content> tag, add a <page> tag that includes the relevant page numbers extracted from the text (e.g., <p.1>, <p.2>).
        - Ensure the tags are correctly closed and formatted.
        Ensure that the output **never** starts with <xml> or markdown

        Here is an example of how the tagged text should look:

        <main>대한민국의 계절</main>
        <sub>여름</sub> <content>요즘 한국의 여름은 매우 덥고 습한 편이다.</content> <page>1</page>
        <sub>겨울</sub> <content>12월부터 온도가 급격히 떨어지며 건조해진다.</content> <page>4</page>

        <main>대한민국의 휴일</main>
        <sub>추석</sub> <content>추석은 비교적 휴일 기간이 긴 편이다.</content> <page>6</page>
        <sub>설날</sub> <content>설날에는 떡국을 먹으며, 가족들과 인사를 나눈다.</content> <page>7</page>

        Text:
        {text}
        """

        response = self.llm_4.invoke(prompt)
        result = response.content.strip()
        return result
    

if __name__ == "__main__":
    api_key_path = "C:/Users/PC/Desktop/DoYoung/DS/github/bitamin_auto_readme_generator/code/assets/openai_api_key.json"
    input_directory = 'C:\\Users\\PC\\Desktop\\DoYoung\\DS\\github\\bitamin_auto_readme_generator\\data\\object_detection\\output\\ocr_samples_txt'
    output_directory = 'C:\\Users\\PC\\Desktop\\DoYoung\\DS\\github\\bitamin_auto_readme_generator\\data\\text_summarization\\output\\method4'

    file_list = os.listdir(input_directory)

    for file_name in file_list:
        if file_name.endswith('_text.txt'):  
            file_path = os.path.join(input_directory, file_name)
            
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()

            summarizer = TextSummarizer(api_key_path)
            text5 = summarizer.extract_5pages(text)
            extracted_info = summarizer.extract_info(text5)
            main_topics_list = summarizer.extract_main_topics(extracted_info)
            topics_list1, topics_list2, topics_list3, topics_list4 = summarizer.split_topics(main_topics_list)
            topic_text1, topic_text2, topic_text3, topic_text4 = summarizer.main(text, topics_list1, topics_list2, topics_list3, topics_list4)
            topic1_summ, topic2_summ, topic3_summ, topic4_summ = summarizer.generate_summaries(topic_text1, topic_text2, topic_text3, topic_text4, topics_list1, topics_list2, topics_list3, topics_list4)
            combined_summ = summarizer.concat_summ(topic1_summ, topic2_summ, topic3_summ, topic4_summ)
            result = summarizer.tag_text(combined_summ)
            final_text = f"{extracted_info}\n\n{result}"

            base_name = file_name.split('_text')[0]

            output_file_path = os.path.join(output_directory, f'{base_name}_text.txt')

            with open(output_file_path, 'w', encoding='utf-8') as output_file:
                output_file.write(final_text)

            print(f"Summarized text has been saved to '{output_file_path}'.")

