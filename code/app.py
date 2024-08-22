from flask import Flask
from flask import request
from flask import render_template #템플릿 파일(.html) 서비스용
# from flask import make_response #응답 객체(Response) 생성용
# from flask import flash, redirect #응답메시지 및 리다이렉트용
from flask import jsonify #JSON형태의 문자열로 응답시
from flask_cors import CORS
import os, re, regex, time
import threading
import pandas as pd
from paddleocr import PaddleOCR

# import custom modules
from object_detection import detection, pdf2img
from image_classification import image_classifier
from text_summarization import textsumm_method3

# 웹 앱 생성
app = Flask(__name__,
            template_folder=os.path.join(os.getcwd(),'webapp'),
            static_folder=os.path.join(os.getcwd(),'webapp'))
# app은 WSGI어플리케이션(Web Server Gateway Interface)를 따르는 웹 어플리케이션이다.

#CORS에러 처리
CORS(app)

#브라우저로 바로 JSON 응답시 한글 처리(16진수로 URL인코딩되는 거 막기)
#Response객체로 응답시는 생략(내부적으로 utf8을 사용)
app.config['JSON_AS_ASCII']=False
#최대 파일 업로드 용량 50MB로 설정
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

# server의 ROOT URL
APP_ROOT = os.getcwd() # bitamin_auto_readme_generator/code/

# 필요 변수들
ASSETS = os.path.join(APP_ROOT,'assets')
WORKSPACE = os.path.join(APP_ROOT,'webapp','workspace')
WORKFILEORIGIN = '' # .pdf 확장자까지 포함하는 원본 파일명
WORKFILECLEAN = '' # 확장자/공백/특수문자를 제거한 파일명 + timestamp
WORKFILEHASH = '' # workfileorigin을 hashing한 값.
WORKIMAGEDIR = '' # images_WORKFILECLEAN : 파일을 처리해서 나온 이미지를 저장할 폴더 경로

# # image detector class 준비
img_det = detection.Image_Detector(os.path.join(ASSETS, 'best.pt'))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
ocr = PaddleOCR(lang='korean', )
# # image classifier class 준비
img_clf = image_classifier.Image_Classifier(os.path.join(ASSETS, 'VGG19clf_acc94_0728.pth'))
# # text summarizer class 준비
txt_summ = textsumm_method3.TextSummarizer(api_key_path=os.path.join(ASSETS, 'openai_api_key.json'))

# 필요 함수들
def clean_filename(text):
    # 1. 확장자 제거 (마지막 점(.) 이후의 모든 문자 제거)
    text = re.sub(r'\.[^.]*$', '', text)
    # 2. 공백 대체 (모든 공백 언더바로 대체)
    text = re.sub(r'\s+', '_', text)
    # 3. 특수문자 제거 (모든 나라의 언어는 제외)
    text = regex.sub(r'[^\p{L}\p{N}0-9]', '', text)
    return text

def process_object_detection():
    '''
    {params}
    pdf_filepath : 처리할 pdf파일의 경로 > 바로 open(pdf_filepath,'r') 해서 사용하면 됨.

    {작업}
    1. 텍스트 추출
        pdf 파일에서 추출한 전체 텍스트를 저장한 "텍스트 파일의 경로" >> WORKSPACE 변수로 지정된 디렉토리에 저장하면 됨.
        >> 텍스트 파일 이름은 WORKFILECLEAN으로 설정하면 됨.
        => 전체 경로 = os.path.join(WORKSPACE,'WORKFILECLEAN'+'.txt')

    2. 이미지 추출
        pdf 파일에서 전체 이미지들을 추출하여 WORKIMAGEDIR 경로에 저장하면 됨.
        >> WORKIMAGEDIR이 폴더 전체 경로니까 os.path.join(WORKIMAGEDIR,이미지파일이름) 으로 바로 저장하면 됨.
    3. 종료.
    '''
    global WORKSPACE
    global WORKFILEORIGIN
    global WORKFILECLEAN
    global WORKIMAGEDIR
    global img_det
    global ocr
    pdf_filepath = os.path.join(WORKSPACE,WORKFILEORIGIN)
    # 페이지별로 이미지로 변환 >> page image가 저장된 디렉토리 이름 반환 > WORKFILECLEAN
    pdf2img_dirname = pdf2img.convert_and_rename_pdf(input_filepath=pdf_filepath,
                                   workfileorigin=WORKFILEORIGIN,
                                   workfileclean=WORKFILECLEAN)
    # image detector 실행 >> textbox image들이 저장된 경로 반환 /data/object_detection/output/WORKFILECLEAN/
    textbox_dirpath = img_det.predict(pdf_name=pdf2img_dirname,  save_image_dir=WORKIMAGEDIR, save=True)

    # OCR
    textbox_filenames = os.listdir(textbox_dirpath)
    textbox_filenames.sort()
    page_texts = {}
    for filename in textbox_filenames:
        parts = filename.split('_')
        if len(parts)>=4:
            page_number = parts[2]
        else:
            continue
        textbox_imagepath = os.path.join(textbox_dirpath, filename)
        result = ocr.ocr(textbox_imagepath, cls=False)[0]
        text = ''
        for r in result:
            text += ' '
            text += r[1][0]
        if page_number not in page_texts:
            page_texts[page_number]=[]
        page_texts[page_number].append(text)
    txt_filepath = os.path.join(WORKSPACE, WORKFILECLEAN+'.txt')
    if os.path.exists(txt_filepath):
        os.remove(txt_filepath)
    with open(txt_filepath, 'w', encoding='utf-8') as f:
        for page_number in sorted(page_texts.keys()):
            f.write(f"<p.{page_number}>\n")
            for text in page_texts[page_number]:
                f.write(text+'\n')

def process_image_classification():
    '''
    {작업}
    1. WORKIMAGEDIR 내에 있는 이미지들을 분류하여 결과를 dataFrame으로 임시 저장
    2. dataFrame에서 class가 'none'으로 분류된 이미지들은 WORKIMAGEDIR에서 제거
    3. 종료.
    '''
    global WORKIMAGEDIR
    global img_clf
    print('start image classification')
    # 추론
    img_clf_result_df, _ = img_clf.predict(WORKIMAGEDIR, save=False) # 결과 df를 csv파일로 저장하지 않음.
    for _, row in img_clf_result_df.iterrows():
        if row['class'] == 'none':
            img_filepath = os.path.join(WORKIMAGEDIR, row['filename'])
            if os.path.exists(img_filepath):
                os.remove(img_filepath)
            else:
                raise OSError(f"{row['filename']} File doesn't exists")

def process_text_summarization():
    '''
    {작업}
    1. txt_filepath 경로에 object_detection의 1번 작업 결과물이 있음. 그거 읽어오기
    2. text summarization 작업 수행
    3. txt to markdown 작업 수행
    4. os.path.join(WORKSPACE,'WORKFILECLEAN'+'.md') 경로로 markdown 파일 저장.
    5. 종료.
    '''
    global WORKSPACE
    global WORKFILECLEAN
    print('start text summarization')
    txt_filepath = os.path.join(WORKSPACE, WORKFILECLEAN+'.txt')
    if not os.path.exists(txt_filepath):
        raise OSError(f"{txt_filepath} File doesn't exists")
    with open(txt_filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    text5 = txt_summ.extract_5pages(text)
    extracted_info, main_topics_list = txt_summ.extract_info(text5)
    cleaned_text = txt_summ.remove_page(text)
    divided_text = txt_summ.divide_text(extracted_info, main_topics_list, cleaned_text)
    summarized_text = txt_summ.summarize(divided_text)
    tagged_text = txt_summ.tag_text(summarized_text)

    print('readme formatting')
    # # readme formatting
    main_re = re.compile(r'<main>(.*?)</main>', re.DOTALL)
    sub_re = re.compile(r'<sub>(.*?)</sub>', re.DOTALL)
    content_re = re.compile(r'<content>(.*?)</content>', re.DOTALL)
    page_re = re.compile(r'<page>(.*?)</page>', re.DOTALL)

    # Find all occurrences of each tag
    subject = tagged_text.split("<subject>")[1].split("</subject>")[0].strip()
    team = tagged_text.split("<team>")[1].split("</team>")[0].strip().split(", ")
    index = tagged_text.split("<index>")[1].split("</index>")[0].strip().split(", ")
    mains = main_re.findall(tagged_text)

    # Convert to Markdown format
    readme_md = f"""
    # {subject}
    (프로젝트 진행기간을 입력하세요. ####.##.## ~ ####.##.##)
    ### Team
    {', '.join(team)}

    ## Table of Contents
    """
    for i, section in enumerate(index, 1):
        readme_md += f"- [{section}](#section_{i})\n"
    readme_md += "<br>\n"
    for idx, main in enumerate(mains):
        main_text = main.strip()
        readme_md += f"<a name='section_{idx + 1}'></a>\n\n## {main_text}\n\n"
        section_content = tagged_text.split(f"<main>{main_text}</main>")[1].split("<main>")[0]

        subs = sub_re.findall(section_content)
        contents = content_re.findall(section_content)
        pages = page_re.findall(section_content)

        # Add the subsections and content
        for sub, content_text, page in zip(subs, contents, pages):
            readme_md += f"#### {sub.strip()}\n\n"
            readme_md += f"- {content_text.strip()}\n\n"
            # readme_md += f"*Page: {page.strip()}*\n\n"

    readme_filepath = os.path.join(WORKSPACE, WORKFILECLEAN+'.md')
    if os.path.exists(readme_filepath):
        os.remove(readme_filepath)
    with open(readme_filepath, 'w', encoding='utf-8') as f:
        f.write(readme_md)

@app.route('/')
def index():
    return render_template('editormd.html')

@app.route('/loadimages.do', methods=['GET'])
def loadImageGallery():
    images_list = os.listdir(WORKIMAGEDIR)
    return images_list

# upload 처리
@app.route('/upload.do', methods=['POST']) # 단일 파일 업로드
def fileUpload():
    global WORKFILEORIGIN
    global WORKFILECLEAN
    global WORKIMAGEDIR
    global WORKFILEHASH
    try:
        # 파일은 request.files['파라미터명']으로 받기
        files = request.files['file']
        # print(files)
        if files.filename.split('.')[-1] != 'pdf':
            return jsonify({'code':400,'msg':'Not PDF file!'})
        # 서버에 파일 저장 : save('서버에 저장할 파일의 전체경로')
        files.save(os.path.join(WORKSPACE,files.filename))  # >> upload 디렉토리 밑에 파일명으로 저장함.
        # 해당 파일명으로 image 폴더 생성
        # 파일명에서 공백 및 특수문자 제거
        WORKFILEORIGIN = files.filename  # 사용자가 업로드한 pdf 파일 이름 원본 (확장자 포함)
        timestamp = str(int(time.time()))
        WORKFILECLEAN = clean_filename(WORKFILEORIGIN)+f"_{timestamp}"  # WORKFILEORIGIN에서 공백/특수문자/확장자를 제거한 clean filename (폴더 생성 시 오류 방지를 위해)
        WORKIMAGEDIR = os.path.join(WORKSPACE, 'images_'+WORKFILECLEAN)  # pdf에 대한 이미지 분류 결과를 저장할 이미지 폴더 경로
        WORKFILEHASH = str(hash(WORKFILEORIGIN))[1:]+f"_{timestamp}"
        try:
            if os.path.exists(WORKIMAGEDIR):
                pass
                #📢📢📢📢📢📢 마지막에 주석 풀어야 함.
                # for image in os.listdir(WORKIMAGEDIR): # 이미지 폴더가 이미 존재하면 해당 경로의 이미지들을 모두 삭제
                #     image_path = os.path.join(WORKIMAGEDIR,image)
                #     if os.path.isfile(image_path):
                #         os.remove(image_path)
            else:
                os.mkdir(WORKIMAGEDIR) # 이미지 폴더 생성

        except Exception as e:
            print("Error when making image dir",e)

        ###############################################################
        # object detection 프로세스를 통해 전체 텍스트와 전체 이미지를 추출한다.
        process_object_detection()
        # Multi-thread : 아래 2가지 프로세스를 병렬적으로 진행한다.
        try:
            # # image classification을 진행한 후 WORKFILEIMAGEDIR에 저장한다.
            thread1 = threading.Thread(target=process_image_classification)
            # # text summarization을 진행한 후, markdown 형태로 변환하여 WORKSPACE에 WORKFILECLEAN.md 파일명으로 저장한다.
            thread2 = threading.Thread(target=process_text_summarization)
            thread1.start()
            thread2.start()
            thread1.join()
            thread2.join()
        except Exception as e:
            print("Threading 중 에러:", e, ',',WORKFILECLEAN, WORKIMAGEDIR)

        return jsonify({'code': 200, 'msg': [WORKFILEORIGIN,WORKFILECLEAN]})
    except Exception as e:
        print('파일 처리 중 에러:',e)
        # print(WORKFILEORIGIN, WORKIMAGEDIR)
        return jsonify({'code':400,'msg': '파일 업로드 실패'})

@app.route('/loadfile.do',methods=['GET'])
def loadFile():
    md_filepath = os.path.join(WORKSPACE,WORKFILECLEAN+'.md')
    if os.path.exists(md_filepath):
        try:
            with open(md_filepath,'r',encoding='utf-8') as f:
                data = f.readlines()
            return jsonify({'code': 200, 'msg': data})
        except Exception as e:
            print(e)
            return jsonify({'code':400,'msg':"Server Error: Cannot read markdown file"})
    else:
        return jsonify({'code': 400, 'msg':"Server Error: File doesn't exists!"})


