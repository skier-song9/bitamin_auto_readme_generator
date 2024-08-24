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
    
    # STEP1: 첫 5페이지의 텍스트에서 팀원, 주제, main topic 추출
    @staticmethod
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
        Ensure that team member's name are not split into multiple lines. Name should be connected by commas if there are more than one.
        Ensure that title is not cut off and is extracted in their entirety.
        Exclude any subtopics or secondary information while extracting main topics.
        Stop at the next page marker '<p.'.

        Do not include any additional notes or explanations in the output.

        Text: {text}


        Format the extracted information as follows:
        <subject>title</subject>
        <team>team members</team>
        <index>main topics</index>

        """

        response = self.llm_4.invoke(prompt)
        extracted_info = response.content.strip()

        main_topics_match = re.search(r'<index>(.*?)</index>', extracted_info, re.DOTALL)
        if main_topics_match:
            main_topics = main_topics_match.group(1).strip()
            main_topics_list = [topic.strip() for topic in main_topics.split(',')]
        else:
            main_topics_list = []
        return extracted_info,  main_topics_list
    

    # STEP2: 전체 텍스트에서 대주제, 소주제, 세부내용 추출
    def extract_details(self, text, main_topics):
        prompt = f"""
        Extract detailed information for each main topic and its subtopics from the provided text.
        Do not change the language of the original input text. 
        Focus on the key details and results relevant to the project, with particular emphasis on the final model outcomes. 
        Ensure that the final model results are described with a high level of accuracy and detail, including specific metrics, performance evaluations, and any significant findings.
        Include the **first page** where each subtopic begins in the "pages used" list. For example, if a subtopic spans from page 7 to page 10, only include page 7 in the result.
        Ensure that detailed contents are extracted comprehensively from all relevant pages, leaving no key information behind. 
        Focus on capturing detailed content across all pages, ensuring that nothing is missed, especially in relation to the provided main topics.

        Format the extracted information as follows:
        <main>main topic</main>
        <sub>subtopic</sub> <content>detailed contents</content> <page>3</page>
        <sub>subtopic</sub> <content>detailed contents</content> <page>5</page>

        <main>main topic</main>
        <sub>subtopic</sub> <content>detailed contents</content> <page>10</page>
        <sub>subtopic</sub> <content>detailed contents</content> <page>11</page>

        Use the main topics provided below:
        {main_topics}
        
        The text is divided into pages using the format <p.number>. For example, page 2 is marked as <p.02>. 
        Make sure to extract text from all pages except for the first five pages to ensure no information is missed.
        If any part of the text seems to be related to the main topics but is not included in the main topics list(main_topics), include it as a subtopic under the appropriate main topic.
        Do not include any additional notes or explanations in the output.

        Text: {text}
        """

        response = self.llm_4.invoke(prompt)
        extracted_details = response.content.strip()
        return extracted_details

    # STEP3: 요약
    def summarize(self, content):
        if not content.strip():
            return content

        prompt = f"""
        Do not change the language of the original input text. 
        Summarize in the same language as the input text.
        The purpose of the summaries is to create a comprehensive README for a GitHub project.
        If multiple subtopics within the same main topic have the same name, combine their content into a single entry with all relevant details, and merge the page numbers into a single list. 
        Ensure that no duplicate subtopics appear within the same main topic. 
        All content related to the same subtopic should be combined into one entry, regardless of how many times it appears.
        Ensure that subtopics from different main topics are not merged.

        Ensure that each summary accurately captures the key points and essential information of its respective partition.
        Summarize in a way that is concise and informative, omitting repetitive mentions of the main topic after the first mention in each partition.
        Focus on the key details and results relevant to the project, with particular emphasis on the final model outcomes. 
        Ensure that the final model results are described with a high level of accuracy and detail, including specific metrics, performance evaluations, and any significant findings.
        For each subtopic, aim to provide a summary that is 1 to 2 sentences long. 
        Focus on delivering a concise yet comprehensive overview, capturing the main points and essential details without unnecessary elaboration.
        When summarizing, only include the **first page** where each subtopic begins in the "pages used" list. For example, if a subtopic spans from page 7 to page 10, only include page 7 in the summary.
        Use the example format as a reference only—do not include it in the final output. Ensure that <sub> and <content> tags are separated by a single space, not a line break, and that the output follows this format exactly.
        Ensure the summary always begins with <subject>, <team>, and <index> in that order, before listing the main topics and content. 

        Here is an example of how the tagged text should look:

        <subject>Korea</subject>
        <team>이하나, 김하니</team>
        <index>Seasons in Korea, Animals in Korea</index>
        
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

        
        Text: {content}

        Please summarize the above content concisely.
        """

        response = self.llm_4.invoke(prompt)
        summary = response.content.strip()
        return summary
    
if __name__ == "__main__":
    api_key_path = "C:/Users/PC/Desktop/DoYoung/DS/github/bitamin_auto_readme_generator/code/assets/openai_api_key.json"
    input_directory = 'C:\\Users\\PC\\Desktop\\DoYoung\\DS\\github\\bitamin_auto_readme_generator\\data\\object_detection\\output\\ocr_samples_txt'
    output_directory = 'C:\\Users\\PC\\Desktop\\DoYoung\\DS\\github\\bitamin_auto_readme_generator\\data\\text_summarization\\output\\method1'

    file_list = os.listdir(input_directory)

    for file_name in file_list:
        if file_name.endswith('_text.txt'):  
            file_path = os.path.join(input_directory, file_name)
            
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()

            summarizer = TextSummarizer(api_key_path)
            text5 = summarizer.extract_5pages(text)
            extracted_info, main_topics_list = summarizer.extract_info(text5)
            details = summarizer.extract_details(text, main_topics_list)
            final_text = summarizer.summarize(details)

            base_name = file_name.split('_text')[0]

            output_file_path = os.path.join(output_directory, f'{base_name}_text.txt')

            with open(output_file_path, 'w', encoding='utf-8') as output_file:
                output_file.write(final_text)

            print(f"Summarized text has been saved to '{output_file_path}'.")



