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
    

    # STEP1: 전체 텍스트를 7가지 클래스로 분류
    def extract_info(self, text):
        prompt = f"""
        Categorize each sentence in the provided text into one of the following categories:

        - <subject>: Title of the document or project.
        - <team>: List of team members.
        - <index>: Main topics or sections in the document.
        - <main>: Main topic headings.
        - <sub>: Subtopics under each main topic.
        - <content>: Detailed content related to each subtopic.
        - <page>: For any occurrence of a page marker (e.g., <p.1>, <p.2>), wrap it within <page></page> tags.
        - <nan>: Unnecessary content not fitting into the above categories or repeated main/sub topics.

        Ensure the original text is not modified, summarized, or omitted. 
        Every single sentence must be included in one of the categories. 
        Do not change the order of the original text.
        Do not include any example sentences in the output.

        Example (Do not include the example in the output):

        <nan><p.1>
        2024 BITAmin 겨울 연합프로젝트 시계열 1조</nan>
        <subject>Netflix Stock Price Prediction with News Topic & Sentiment</subject>
        <nan>시계열 1조</nan>
        <nan>12기</nan> <team>송규헌</team>
        <nan>12기</nan> <team>권도영</team>

        <nan><p.2>
        CONTENTS</nan>
        <index>01. INTRODUCTION</index>
        <index>02. DATA PREPROCESSING</index>
        <index>03. MODELING</index>

        <nan><p.3></nan>
        <nan>비타민 11기 겨울 컨퍼런스</nan>
        <main>01. INTRODUCTION</main>

        <nan><p.4></nan>
        <main>01. INTRODUCTION</main>
        <sub>1.1 Background of topic selection</sub>
        <content>1.뉴스가 주가 변동에 미치는 영향 탐구
        주가를 예측하는 데 사용하는 데이터로 뉴스의 감성분석 및 토픽 모델링 결과를 사용하고자 함
        뉴스 감성 분석/토픽 모델링 결과를 활용하여 주가를 예측하는 프로젝트는 많지 않아, 직접 뉴스 데이터를 활용하고자 함</content>

        Text: {text}
        """

        response = self.llm_4.invoke(prompt)
        extracted_info = response.content.strip()
        return extracted_info
    
    
    def remove_nan_tags(self, text):
        cleaned_text = re.sub(r'<nan>[\s\S]*?</nan>', '', text, flags=re.DOTALL)
        return cleaned_text.strip()
    
    
    # STEP2: 요약
    def summarize(self, text):
        prompt = f"""
        After removing those segments, summarize the remaining text in the following format:
        Do **not change** or translate the language of the original input text. 
        Summarize in the same language as the input text.
        Ensure that the output **never** starts with <xml> or markdown
        
        The purpose of the summaries is to create a comprehensive README for a GitHub project.
        If multiple subtopics within the same main topic have the same name, combine their content into a single entry with all relevant details, and merge the page numbers into a single list. 
        Ensure that subtopics from different main topics are not merged.
        If multiple subtopics within the same main topic have the same name, combine their content into a single entry with all relevant details, and merge the page numbers into a single list. 


        Ensure that each summary accurately captures the key points and essential information of its respective partition.
        Summarize in a way that is concise and informative, omitting repetitive mentions of the main topic after the first mention in each partition.
        Focus on the key details and results relevant to the project, with particular emphasis on the final model outcomes. 
        Ensure that the final model results are described with a high level of accuracy and detail, including specific metrics, performance evaluations, and any significant findings.
        For each subtopic, aim to provide a summary that is 1 to 2 sentences long. Focus on delivering a concise yet comprehensive overview, capturing the main points and essential details without unnecessary elaboration.
        When summarizing, only include the **first page** where each subtopic begins in the "pages used" list. For example, if a subtopic spans from page 7 to page 10, only include page 7 in the summary.
        Ensure the summary always begins with <subject>, <team>, and <index> in that order, before listing the main topics and content. 
        Use the example format as a reference only—do not include it in the final output. Ensure that <sub> and <content> tags are separated by a single space, not a line break, and that the output follows this format exactly.

        Here is an example of how the tagged text should look:

        <subject>Korea</subject>
        <team>이하나, 김하니</team>
        <index>Seasons in Korea>
        
        <main>Seasons in Korea</main>
        <sub>봄</sub> <content>한국의 봄은 3월에서 5월까지 지속되며, 온화한 기온과 함께 벚꽃이 만개하는 시기입니다. 이 시기에는 다양한 봄꽃 축제가 열리며, 사람들이 야외 활동을 즐기기에 좋은 날씨입니다.</content> <page>1</page>
        <sub>여름</sub> <content>한국의 여름은 대개 매우 덥고 습하며, 기온이 종종 30°C를 넘습니다. 이 시기는 장마철이기도 해서 특히 7월에 많은 비가 내립니다. 더위에도 불구하고, 이 시기는 휴가철로 인기가 많아 많은 사람들이 해변과 리조트로 향합니다.</content> <page>3</page>
        <sub>가을</sub> <content>가을은 한국에서 가장 아름다운 계절 중 하나로 꼽힙니다. 9월에서 11월 사이, 날씨는 시원하고 하늘은 맑으며, 단풍이 절정을 이루어 산과 들이 붉고 노란 색으로 물듭니다. 이 시기는 또한 수확의 계절로, 각종 축제가 열립니다.</content> <page>5</page>
        <sub>겨울</sub> <content>한국의 겨울은 12월에 시작되며, 대부분의 지역에서 기온이 영하로 떨어집니다. 겨울철은 건조하며, 특히 산악 지역에서는 가끔씩 눈이 내립니다. 이 시기에는 스키나 얼음 낚시와 같은 겨울 스포츠가 인기를 끌고 있습니다.</content> <page>7</page>

        <main>Animals in Korea</main>
        <sub>봄</sub> <content>봄철에는 한국의 자연이 다시 깨어나면서 다양한 동물들이 활동을 시작합니다. 산과 숲에서는 새들이 짝을 찾기 위해 지저귀고, 개구리와 같은 양서류들이 물가에서 활동을 재개합니다. 또한, 겨울잠에서 깨어난 동물들이 활발히 먹이를 찾는 모습을 볼 수 있습니다.</content> <page>9</page>
        <sub>여름</sub> <content>여름철에는 다양한 야생 동물들이 활발히 활동합니다. 특히, 한국의 산과 숲에서 멧돼지, 사슴, 다양한 종류의 새들이 많이 보입니다. 여름은 또한 번식기가 겹쳐 동물들이 더욱 활발히 움직이는 시기입니다.</content> <page>10</page>
        <sub>가을</sub> <content>가을에는 동물들이 겨울을 준비하는 모습을 볼 수 있습니다. 많은 동물들이 겨울잠을 준비하기 위해 지방을 축적하고, 새들은 따뜻한 지역으로 이동하기 시작합니다. 이 시기는 또한 수확기가 겹쳐, 농작물에 접근하는 동물들이 많아집니다.</content> <page>11</page>
        <sub>겨울</sub> <content>겨울철에는 많은 동물들이 추위를 피하기 위해 겨울잠에 들어갑니다. 겨울잠을 자지 않는 동물들은 추위를 이기기 위해 두꺼운 털을 기르거나 활동을 줄입니다. 특히, 한국의 산악지대에서는 고라니와 같은 동물들이 눈 속에서 식량을 찾아다니는 모습을 볼 수 있습니다.</content> <page>12</page>


        Text: {text}
        """

        response = self.llm_4.invoke(prompt)
        summarized_text = response.content.strip()
        return summarized_text


if __name__ == "__main__":
    api_key_path = "C:\\Users\\PC\\Desktop\\DoYoung\\DS\\비타민NLP_240701\\text_summarization\\openai_api_key.json"
    input_directory = 'C:\\Users\\PC\\Desktop\\DoYoung\\DS\\github\\bitamin_auto_readme_generator\\data\\object_detection\\output\\ocr_samples_txt'
    output_directory = 'C:\\Users\\PC\\Desktop\\DoYoung\\DS\\github\\bitamin_auto_readme_generator\\data\\text_summarization\\output\\method2'

    file_list = os.listdir(input_directory)

    for file_name in file_list:
        if file_name.endswith('_text.txt'):  
            file_path = os.path.join(input_directory, file_name)
            
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()

            summarizer = TextSummarizer(api_key_path)
            extracted_info = summarizer.extract_info(text)  
            extracted_info = summarizer.remove_nan_tags(extracted_info)
            final_text = summarizer.summarize(extracted_info)

            base_name = file_name.split('_text')[0]

            output_file_path = os.path.join(output_directory, f'{base_name}_text.txt')

            with open(output_file_path, 'w', encoding='utf-8') as output_file:
                output_file.write(final_text)

            print(f"Summarized text has been saved to '{output_file_path}'.")