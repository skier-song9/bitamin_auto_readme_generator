from flask import Flask
from flask import request
from flask import render_template#템플릿 파일(.html) 서비스용
from flask import make_response#응답 객체(Response) 생성용

from flask import flash, redirect #응답메시지 및 리다이렉트용
from flask import jsonify#JSON형태의 문자열로 응답시

from flask_cors import CORS
from flask_restful import Api
import os

# 웹 앱 생성
app = Flask(__name__) # app은 WSGI어플리케이션(Web Server Gateway Interface)를 따르는 웹 어플리케이션이다.

#CORS에러 처리
CORS(app)

#브라우저로 바로 JSON 응답시 한글 처리(16진수로 URL인코딩되는 거 막기)
#Response객체로 응답시는 생략(내부적으로 utf8을 사용)
app.config['JSON_AS_ASCII']=False
#업로드용 폴도 설정
UPLOAD_FOLDER = os.getcwd()
app.config['UPLOAD_FOLDER'] = os.path.join(UPLOAD_FOLDER,'uploads')
#최대 파일 업로드 용량 50MB로 설정
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
