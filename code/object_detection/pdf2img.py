import os
from pdf2jpg import pdf2jpg
import time
import re

ROOT = 'bitamin_auto_readme_generator'

def convert_and_rename_pdf(
        input_filepath, workfileorigin, workfileclean
        # pdf_name
        ):
    root_absdir = os.getcwd().split(ROOT)[0]+ROOT
    pdf_name_origin = ''.join(workfileorigin.split('.')[:-1])
    pdf_name = workfileclean
    # input_dir = os.path.join(root_absdir,'data','object_detection','input',f'{pdf_name}.pdf')
    input_dir = input_filepath # pdf의 파일 경로
    output_dir = os.path.join(root_absdir,'data','object_detection','input') # pdf 페이지별로 이미지로 변환한 것을 저장할 디렉토리
    new_output_dir = os.path.join(output_dir, pdf_name) # output_dir의 이름을 변경하기 위함

    pdf2jpg.convert_pdf2jpg(input_dir, output_dir, pages="ALL")
    
    generated_dir = os.path.join(output_dir, pdf_name_origin + '.pdf_dir')

    def sort_key(filename):
        match = re.search(r'\d+', filename)
        return int(match.group()) if match else 0

    if os.path.exists(generated_dir):
        os.rename(generated_dir, new_output_dir)
        for idx, filename in enumerate(sorted(os.listdir(new_output_dir),key=sort_key)):
            old_file = os.path.join(new_output_dir, filename)
            new_filename = f"{idx:02}.jpg"
            new_file = os.path.join(new_output_dir, new_filename)
            os.rename(old_file, new_file)
        print(f"Conversion completed.")
    else:
        print(f"Error: Generated directory '{generated_dir}' not found.")
        pass
    return pdf_name

'''
convert_and_rename_pdf('C:\\Users\\happy\Desktop\\bitamin_auto_readme_generator\\data\\object_detection\\input\\lstm.pdf',
                       'lstm.pdf',
                       'lstm_0822.pdf')
'''
