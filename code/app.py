import queue
from flask import Flask
from flask import request, session, g, copy_current_request_context
# # session은 각 클라이언트별로 json 형태의 데이터를 독립적으로 관리할 수 있는 객체이다.
# # g 객체는 json 형태의 데이터가 아니더라도 클라이언트별로 독립적으로 관리할 수 있도록 해준다.
from flask import render_template #템플릿 파일(.html) 서비스용
# from flask import make_response #응답 객체(Response) 생성용
# from flask import flash, redirect #응답메시지 및 리다이렉트용
from flask import jsonify #JSON형태의 문자열로 응답시
from flask_cors import CORS
import os, re, regex, time
import json
import threading
import shutil
import pandas as pd

# import custom modules
from object_detection import detection, pdf2img, textocr
from image_classification import image_classifier
from text_summarization import textsumm_method3
from object_pooling.objcet_pooling import ObjectPool

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
#
#최대 파일 업로드 용량 50MB로 설정
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

# server의 ROOT URL
APP_ROOT = os.getcwd() # bitamin_auto_readme_generator/code/

# 필요 변수들
ASSETS = os.path.join(APP_ROOT,'assets')
WORKSPACE = os.path.join(APP_ROOT,'webapp','workspace')
### 아래 변수들은 client별로 관리해야 하기 때문에 session으로 관리한다.
# WORKFILEORIGIN = '' # .pdf 확장자까지 포함하는 원본 파일명
# WORKFILECLEAN = '' # 확장자/공백/특수문자를 제거한 파일명 + timestamp
# WORKFILEHASH = '' # workfileorigin을 hashing한 값.
# WORKIMAGEDIR = '' # images_WORKFILECLEAN : 파일을 처리해서 나온 이미지를 저장할 폴더 경로

# ✅ Object Pool
with app.app_context():
    def create_img_det():
        return detection.Image_Detector(os.path.join(ASSETS, 'best.pt'))
    def create_ocr():
        return textocr.PaddleOcr(lang='korean')
    def create_img_clf():
        return image_classifier.Image_Classifier(os.path.join(ASSETS, 'VGG19clf_acc94_0728.pth'))
    def create_txt_summ():
        return textsumm_method3.TextSummarizer(api_key_path=os.path.join(ASSETS, 'openai_api_key.json'))
    # 각 객체 풀 생성
    img_det_pool = ObjectPool(create_img_det, max_size=3)
    ocr_pool = ObjectPool(create_ocr, max_size=3)
    img_clf_pool = ObjectPool(create_img_clf, max_size=3)
    txt_summ_pool = ObjectPool(create_txt_summ, max_size=3)


# session 사용을 위한 secret_key 설정
# with open(os.path.join(ASSETS,'session_secret_key.json'),'r',encoding='utf-8') as f:
#     app.secret_key = json.load(f)['secret-key']

# # image detector class 준비
# img_det = detection.Image_Detector(os.path.join(ASSETS, 'best.pt'))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" # paddleOCR용 환경변수 설정
# ocr = textocr.PaddleOcr(lang='korean')
# # image classifier class 준비
# img_clf = image_classifier.Image_Classifier(os.path.join(ASSETS, 'VGG19clf_acc94_0728.pth'))
# # text summarizer class 준비
# txt_summ = textsumm_method3.TextSummarizer(api_key_path=os.path.join(ASSETS, 'openai_api_key.json'))

# 필요 함수들
def clean_filename(text):
    # 1. 확장자 제거 (마지막 점(.) 이후의 모든 문자 제거)
    text = re.sub(r'\.[^.]*$', '', text)
    # 2. 공백 대체 (모든 공백 언더바로 대체)
    text = re.sub(r'\s+', '_', text)
    # 3. 특수문자 제거 (모든 나라의 언어는 제외)
    text = regex.sub(r'[^\p{L}\p{N}0-9_]', '', text)
    return text
def get_basename(path):
    base = os.path.basename(path)
    if os.path.isfile(path): # file
        return base[:base.rfind('.')]
    else: # directory
        return base

def process_object_detection(workfileclean, workimagedir, session_json):
    '''
    {params}

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
    # global WORKFILECLEAN
    # global WORKIMAGEDIR
    # global img_det
    # global ocr
    WORKFILECLEAN = workfileclean
    WORKIMAGEDIR = workimagedir
    img_det, ocr = None, None
    try:
        img_det = img_det_pool.get(timeout=120)
        ocr = ocr_pool.get(timeout=120)
        try:
            pdf_filepath = os.path.join(WORKSPACE, WORKFILECLEAN + ".pdf")
            # 페이지별로 이미지로 변환 >> page image가 저장된 디렉토리 이름 반환 > WORKFILECLEAN, /data/object_detection/input/WORKFILECLEAN
            pdf2img_dirname, pdf2img_dirpath = pdf2img.convert_and_rename_pdf(input_filepath=pdf_filepath,
                                                                              workfileorigin=WORKFILECLEAN + ".pdf",
                                                                              workfileclean=WORKFILECLEAN)
            session_json['pdf2img_dir'].append(pdf2img_dirpath)
        except Exception as e:
            raise Exception("process_object_detection,pdf2img")
        try:
            # image detector 실행 >> textbox image들이 저장된 경로 반환 /data/object_detection/output/WORKFILECLEAN/textbox
            textbox_dirpath = img_det.predict(pdf_name=pdf2img_dirname, save_image_dir=WORKIMAGEDIR, save=True)
            session_json['textbox_dir'].append(os.path.dirname(textbox_dirpath)) # /data/object_detection/output/WORKFILECLEAN/
        except Exception as e:
            # print(e)
            raise Exception(e)
            # raise Exception("process_object_detection,textbox")
        try:
            # OCR
            ocr(textbox_dirpath=textbox_dirpath, save_filepath=os.path.join(WORKSPACE, WORKFILECLEAN + '.txt'))
            session_json['text_filepath'].append(os.path.join(WORKSPACE, WORKFILECLEAN + '.txt'))
        except Exception as e:
            raise Exception("process_object_detection,txt")
    except TimeoutError as e:
        raise TimeoutError(e)
    finally:
        if img_det:
            img_det_pool.put(img_det)
        if ocr:
            ocr_pool.put(ocr)
            print(ocr_pool.__len__())
    return session_json

# def process_image_classification(workimagedir):
#     '''
#     {작업}
#     1. WORKIMAGEDIR 내에 있는 이미지들을 분류하여 결과를 dataFrame으로 임시 저장
#     2. dataFrame에서 class가 'none'으로 분류된 이미지들은 WORKIMAGEDIR에서 제거
#     3. 종료.
#     '''
#     # global WORKIMAGEDIR
#     # global img_clf
#     print('start image classification')
#     # 추론
#     WORKIMAGEDIR = workimagedir
#     img_clf = None
#     try:
#         img_clf = img_clf_pool.get(timeout=120)
#         try:
#             img_clf_result_df, _ = img_clf.predict(WORKIMAGEDIR, save=False) # 결과 df를 csv파일로 저장하지 않음.
#             for _, row in img_clf_result_df.iterrows():
#                 if row['class'] == 'none':
#                     img_filepath = os.path.join(WORKIMAGEDIR, row['filename'])
#                     if os.path.exists(img_filepath):
#                         os.remove(img_filepath)
#                     else:
#                         raise OSError(f"{row['filename']} File doesn't exists")
#         except Exception as e:
#             print(e)
#             raise Exception("process_image_classification,img_clf")
#     except TimeoutError as e:
#         raise TimeoutError(e)
#     finally:
#         if img_clf:
#             img_clf_pool.put(img_clf)
#             print(img_clf_pool.__len__())

# def process_text_summarization(workfileclean):
#     '''
#     {작업}
#     1. txt_filepath 경로에 object_detection의 1번 작업 결과물이 있음. 그거 읽어오기
#     2. text summarization 작업 수행
#     3. txt to markdown 작업 수행
#     4. os.path.join(WORKSPACE,'WORKFILECLEAN'+'.md') 경로로 markdown 파일 저장.
#     5. 종료.
#     '''
#     global WORKSPACE
#     # global txt_summ
#     WORKFILECLEAN = workfileclean
#     txt_summ = None
#     try:
#         txt_summ = txt_summ_pool.get(timeout=120)
#         try:
#             print('start text summarization')
#             txt_filepath = os.path.join(WORKSPACE, WORKFILECLEAN+'.txt')
#             if not os.path.exists(txt_filepath):
#                 raise OSError(f"{txt_filepath} File doesn't exists")
#             with open(txt_filepath, 'r', encoding='utf-8') as f:
#                 text = f.read()
#             text5 = txt_summ.extract_5pages(text)
#             extracted_info, main_topics_list = txt_summ.extract_info(text5)
#             divided_text = txt_summ.divide_text(extracted_info, main_topics_list, text)
#             summarized_text = txt_summ.summarize(divided_text, main_topics_list)
#             tagged_text = txt_summ.tag_text(summarized_text)
#             total_text = extracted_info+'\n'+tagged_text
#
#             print('readme formatting')
#             # # readme formatting
#             main_re = re.compile(r'<main>(.*?)</main>', re.DOTALL)
#             sub_re = re.compile(r'<sub>(.*?)</sub>', re.DOTALL)
#             content_re = re.compile(r'<content>(.*?)</content>', re.DOTALL)
#             page_re = re.compile(r'<page>(.*?)</page>', re.DOTALL)
#
#             # Find all occurrences of each tag
#             subject = total_text.split("<subject>")[1].split("</subject>")[0].strip()
#             team = total_text.split("<team>")[1].split("</team>")[0].strip().split(", ")
#             index = total_text.split("<index>")[1].split("</index>")[0].strip().split(", ")
#             mains = main_re.findall(total_text)
#
#             # Convert to Markdown format
#             readme_md = f"""# {subject}
# (프로젝트 진행기간을 입력하세요. ####.##.## ~ ####.##.##)
# ### Team
# {', '.join(team)}
#
# ## Table of Contents
# """
#             for i, section in enumerate(index, 1):
#                 readme_md += f"- [{section}](#section_{i})\n"
#             readme_md += "<br>\n"
#             for idx, main in enumerate(mains):
#                 main_text = main.strip()
#                 readme_md += f"<a name='section_{idx + 1}'></a>\n\n## {main_text}\n\n"
#                 section_content = total_text.split(f"<main>{main_text}</main>")[1].split("<main>")[0]
#
#                 subs = sub_re.findall(section_content)
#                 contents = content_re.findall(section_content)
#                 pages = page_re.findall(section_content)
#
#                 # Add the subsections and content
#                 for sub, content_text, page in zip(subs, contents, pages):
#                     readme_md += f"#### {sub.strip()}\n\n"
#                     readme_md += f"- {content_text.strip()}\n\n"
#                     # readme_md += f"*Page: {page.strip()}*\n\n"
#
#             readme_filepath = os.path.join(WORKSPACE, WORKFILECLEAN+'.md')
#             if os.path.exists(readme_filepath):
#                 os.remove(readme_filepath)
#             with open(readme_filepath, 'w', encoding='utf-8') as f:
#                 f.write(readme_md)
#             session['md_filepath'].append(readme_filepath)
#         except Exception as e:
#             raise Exception("process_text_summarization,txt_summ")
#     except TimeoutError as e:
#         raise TimeoutError(e)
#     finally:
#         if txt_summ:
#             txt_summ_pool.put(txt_summ)
#             print(txt_summ_pool.__len__())

@app.route('/')
def index():
    # set session variables
    # session['workfileorigin'] = []
    # session['workfileclean'] = []
    # session['workimagedir'] = []
    # session['pdf_filepath'] = []
    # session['pdf2img_dir'] = []
    # session['textbox_dir'] = []
    # session['text_filepath'] = []
    # session['md_filepath'] = []
    return render_template('editormd.html')

@app.route('/loadimages.do', methods=['POST'])
def loadImageGallery():
    try:
        request_session = request.get_json()
        WORKIMAGEDIR = request_session['workimagedir'][-1]
        images_list = os.listdir(WORKIMAGEDIR)
        return jsonify({'code':200,'msg':request_session,'dir':WORKIMAGEDIR.split(os.sep)[-1],'filelist':images_list})
    except Exception as e:
        return jsonify({'code':400,'msg':str(e)})

# upload 처리
@app.route('/upload.do', methods=['POST']) # 단일 파일 업로드
def fileUpload():
    @copy_current_request_context
    def process_image_classification(workimagedir):
        '''
        {작업}
        1. WORKIMAGEDIR 내에 있는 이미지들을 분류하여 결과를 dataFrame으로 임시 저장
        2. dataFrame에서 class가 'none'으로 분류된 이미지들은 WORKIMAGEDIR에서 제거
        3. 종료.
        '''
        # global WORKIMAGEDIR
        # global img_clf
        print('start image classification')
        # 추론
        WORKIMAGEDIR = workimagedir
        img_clf = None
        try:
            img_clf = img_clf_pool.get(timeout=120)
            try:
                img_clf_result_df, _ = img_clf.predict(WORKIMAGEDIR, save=False)  # 결과 df를 csv파일로 저장하지 않음.
                for _, row in img_clf_result_df.iterrows():
                    if row['class'] == 'none':
                        img_filepath = os.path.join(WORKIMAGEDIR, row['filename'])
                        if os.path.exists(img_filepath):
                            os.remove(img_filepath)
                        else:
                            raise OSError(f"{row['filename']} File doesn't exists")
            except Exception as e:
                print(e)
                raise Exception("process_image_classification,img_clf")
        except TimeoutError as e:
            raise TimeoutError(e)
        finally:
            if img_clf:
                img_clf_pool.put(img_clf)
                print(img_clf_pool.__len__())
    @copy_current_request_context
    def process_text_summarization(workfileclean):
        '''
        {작업}
        1. txt_filepath 경로에 object_detection의 1번 작업 결과물이 있음. 그거 읽어오기
        2. text summarization 작업 수행
        3. txt to markdown 작업 수행
        4. os.path.join(WORKSPACE,'WORKFILECLEAN'+'.md') 경로로 markdown 파일 저장.
        5. 종료.
        '''
        global WORKSPACE
        # global txt_summ
        WORKFILECLEAN = workfileclean
        txt_summ = None
        try:
            txt_summ = txt_summ_pool.get(timeout=120)
            try:
                print('start text summarization')
                txt_filepath = os.path.join(WORKSPACE, WORKFILECLEAN + '.txt')
                if not os.path.exists(txt_filepath):
                    raise OSError(f"{txt_filepath} File doesn't exists")
                with open(txt_filepath, 'r', encoding='utf-8') as f:
                    text = f.read()
                text5 = txt_summ.extract_5pages(text)
                extracted_info, main_topics_list = txt_summ.extract_info(text5)
                divided_text = txt_summ.divide_text(extracted_info, main_topics_list, text)
                summarized_text = txt_summ.summarize(divided_text, main_topics_list)
                tagged_text = txt_summ.tag_text(summarized_text)
                total_text = extracted_info + '\n' + tagged_text

                print('readme formatting')
                # # readme formatting
                main_re = re.compile(r'<main>(.*?)</main>', re.DOTALL)
                sub_re = re.compile(r'<sub>(.*?)</sub>', re.DOTALL)
                content_re = re.compile(r'<content>(.*?)</content>', re.DOTALL)
                page_re = re.compile(r'<page>(.*?)</page>', re.DOTALL)

                # Find all occurrences of each tag
                subject = total_text.split("<subject>")[1].split("</subject>")[0].strip()
                team = total_text.split("<team>")[1].split("</team>")[0].strip().split(", ")
                index = total_text.split("<index>")[1].split("</index>")[0].strip().split(", ")
                mains = main_re.findall(total_text)

                # Convert to Markdown format
                readme_md = f"""# {subject}
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
                    section_content = total_text.split(f"<main>{main_text}</main>")[1].split("<main>")[0]

                    subs = sub_re.findall(section_content)
                    contents = content_re.findall(section_content)
                    pages = page_re.findall(section_content)

                    # Add the subsections and content
                    for sub, content_text, page in zip(subs, contents, pages):
                        readme_md += f"#### {sub.strip()}\n\n"
                        readme_md += f"- {content_text.strip()}\n\n"
                        # readme_md += f"*Page: {page.strip()}*\n\n"

                readme_filepath = os.path.join(WORKSPACE, WORKFILECLEAN + '.md')
                if os.path.exists(readme_filepath):
                    os.remove(readme_filepath)
                with open(readme_filepath, 'w', encoding='utf-8') as f:
                    f.write(readme_md)
                # session_json['md_filepath'].append(readme_filepath)
            except Exception as e:
                raise Exception("process_text_summarization,txt_summ")
        except TimeoutError as e:
            raise TimeoutError(e)
        finally:
            if txt_summ:
                txt_summ_pool.put(txt_summ)
                print(txt_summ_pool.__len__())
    global WORKSPACE
    # global WORKFILEORIGIN
    # global WORKFILECLEAN
    # global WORKIMAGEDIR
    # global WORKFILEHASH
    session_json = {'workfileclean':[], 'workimagedir':[], 'pdf_filepath':[], 'pdf2img_dir':[], 'textbox_dir':[], 'text_filepath':[], 'md_filepath':[]}
    start_time = time.time()
    # # 이전에 업로드한 파일이 있다면 제거 >> 세션으로 관리하면 불필요
    # try:
    #     if len(session['workfileorigin']>0):
    #         os.remove(os.path.join(WORKSPACE,session['workfileclean']+'.pdf'))
    #         shutil.rmtree(session['workimagedir'])
    #         prev_pdf2img_dirpath = os.path.join(os.path.dirname(APP_ROOT),'data','object_detection','input',session['workfileclean'])
    #         prev_textbox_parent_dirpath = os.path.join(os.path.dirname(APP_ROOT),'data','object_detection','output',session['workfileclean'])
    #         shutil.rmtree(prev_pdf2img_dirpath)
    #         shutil.rmtree(prev_textbox_parent_dirpath)
    #         os.remove(os.path.join(WORKSPACE,session['workfileclean']+".txt"))
    #         os.remove(os.path.join(WORKSPACE,session['workfileclean']+".md"))
    # except Exception as e:
    #     print('Error when removing previously uploaded files')
    #     pass
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
        # WORKFILEHASH = str(hash(WORKFILEORIGIN))[1:]+f"_{timestamp}"
        # session['workfileorigin'].append(WORKFILEORIGIN)
        session_json['workfileclean'].append(WORKFILECLEAN)

        # test code
        # WORKFILECLEAN = '비타민_시계열예측과_강화학습을_활용한_시스템_트레이딩_1724372048'
        # WORKIMAGEDIR = os.path.join(WORKSPACE, 'images_'+WORKFILECLEAN)
        # return jsonify({'code': 200, 'msg': [WORKFILEORIGIN, WORKFILECLEAN]})

        try:
            # 원본 파일명을 WORKFILECLEAN으로 변경함
            os.rename(os.path.join(WORKSPACE,WORKFILEORIGIN), os.path.join(WORKSPACE,WORKFILECLEAN+".pdf"))
            session_json['pdf_filepath'].append(os.path.join(WORKSPACE, WORKFILECLEAN + '.pdf'))
            if os.path.exists(WORKIMAGEDIR): # 이미 존재하면 안에 내용 모두 제거 후 재생성
                shutil.rmtree(WORKIMAGEDIR)
                os.mkdir(WORKIMAGEDIR)
            else:
                os.mkdir(WORKIMAGEDIR) # 이미지 폴더 생성
            session_json['workimagedir'].append(WORKIMAGEDIR)
        except Exception as e:
            print("Error when making image dir",e)

        ###############################################################
        # object detection 프로세스를 통해 전체 텍스트와 전체 이미지를 추출한다.
        session_json = process_object_detection(WORKFILECLEAN, WORKIMAGEDIR, session_json)

        ttime = time.time()
        print(f"{ttime-start_time:0.3f} seconds to process_object_detection")
        # Multi-thread : 아래 2가지 프로세스를 병렬적으로 진행한다.
        # try:
        # # image classification을 진행한 후 WORKFILEIMAGEDIR에 저장한다.
        thread1 = threading.Thread(target=process_image_classification, args=(WORKIMAGEDIR,))
        # # text summarization을 진행한 후, markdown 형태로 변환하여 WORKSPACE에 WORKFILECLEAN.md 파일명으로 저장한다.
        thread2 = threading.Thread(target=process_text_summarization, args=(WORKFILECLEAN,))
        thread1.start()
        thread2.start()
        thread1.join()
        thread2.join()
        session_json['md_filepath'].append(os.path.join(WORKSPACE,WORKFILECLEAN+'.md'))
        ttime2 = time.time()
        print(f"{ttime2 - ttime:0.3f} seconds to img_clf and text_summ")
        # except Exception as e:
        #     print("Threading 중 에러:", e, ',',WORKFILECLEAN, WORKIMAGEDIR)
        print(session_json)
        return jsonify({'code': 200, 'msg': session_json, 'workfileorigin':WORKFILEORIGIN})
    except Exception as e:
        if str(e) == "object_pool_timeout":
            print(e)
            return jsonify({'code':400,'msg':"⚠️죄송합니다. 현재 서버에 접속 중인 사람이 많으므로 조금 후에 이용해주시길 바랍니다."})
        else:
            print('파일 처리 중 에러:',e, session_json, g)
            return jsonify({'code':400,'msg': '파일 업로드 실패'})


@app.route('/loadfile.do',methods=['POST'])
def loadFile():
    global WORKSPACE
    # global WORKFILECLEAN
    # print(session)
    # WORKFILECLEAN = session['workfileclean'][-1]
    request_session = request.get_json()
    WORKFILECLEAN = request_session['workfileclean'][-1]
    md_filepath = os.path.join(WORKSPACE,WORKFILECLEAN+'.md')
    # print(md_filepath)
    if os.path.exists(md_filepath):
        try:
            with open(md_filepath,'r',encoding='utf-8') as f:
                data = f.read()
            # print(data)
            return jsonify({'code': 200, 'msg': data})
        except Exception as e:
            # print(e)
            return jsonify({'code':400,'msg':"Server Error: Cannot read markdown file"})
    else:
        return jsonify({'code': 400, 'msg':"Server Error: File doesn't exists!"})

@app.route('/tutorial.do',methods=['GET'])
def tutorial():
    global APP_ROOT
    md_filepath = os.path.join(APP_ROOT,'webapp','tutorial.md')
    # print(md_filepath)
    if os.path.exists(md_filepath):
        try:
            with open(md_filepath,'r',encoding='utf-8') as f:
                data = f.read()
            # print(data)
            return jsonify({'code': 200, 'msg': data})
        except Exception as e:
            # print(e)
            return jsonify({'code':400,'msg':"Server Error: Cannot read markdown file"})
    else:
        return jsonify({'code': 400, 'msg':"Server Error: File doesn't exists!"})

@app.route('/endsession.do',methods=['POST'])
def endSession():
    print('`✅endSession')
    request_session = request.get_json()
    print(request_session)
    session_keys = ['workimagedir','pdf_filepath','pdf2img_dir','textbox_dir','text_filepath','md_filepath']
    for key in session_keys:
        paths = request_session[key]
        for path in paths:
            if os.path.isfile(path):
                os.remove(path)
            else: # directory
                shutil.rmtree(path)
    return 'Session CLeared.',200