# Flask 웹 어플리케이션 서비스 하기
'''
1. 콘솔창에서 python server.py 명령어 실행
  (flaskserver) D:\\CCH\\Workspace\\Python\\FlaskApp>python server.py
'''
# 혹은
'''
2. .exe실행파일 만들어서 실행
 [.py를 .exe로 만들기]
 1) pip install pyinstaller
 2) pyinstaller --help 사용법 확인
  예 : pyinstaller --noconsole --onefile server.py -i="./myicon.ico" --add-data "templates;templates" --add-data "static;static"
  --noconsole옵션은 콘솔창을 띄우지 않기 위한 설정
 -i는 .exe파일의 아이콘 설정
 --onefile server.py는 server.py를 server.exe로 생성

  아이콘은 https://www.iconfinder.com/에서 이미지 다운후
   https://www.icoconverter.xn--com-k94n91q .ico로 컨버트

  서버중지시는 작업관리자에서 server.exe 찾은후 작업끝내기
  ※ unicodeDecodeError : 'utf-8' codec can't decode byte 0xff 와 같은 에러 발생시
    (pyinstaller의 고질적인 문제)
  C:\\Users\kosmo\\anaconda3\\envs\\flaskserver\\Lib\\site-packages\\PyInstaller폴더의
  compat.py의 348라인 수정
  수정 전 :#out = out.decode(encoding)
  수정 후 : out = out.decode(encoding, errors='ignore')
'''

import socket
from waitress import serve
from app import app

host=""
try:
    hostname = socket.gethostname()
    host = socket.gethostbyname(hostname)
except socket.error as e:
    host = "0.0.0.0"

port = "5000"

print(f'➡️서버가 시작되었습니다.\nURL: http://{host}:{port}')

serve(app, host=host, port=port)