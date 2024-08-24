from paddleocr import PaddleOCR
import os
import re
import argparse
import stat

def sort_key(filename):
        match = re.search(r'\d+', filename)
        return int(match.group()) if match else 0

class PaddleOcr(object):
    def __init__(self, lang='korean'):
        self.ocr = PaddleOCR(lang=lang, show_log=False)
    
    def __call__(self, textbox_dirpath, save_filepath):
        page_texts = {}
        filenames = os.listdir(textbox_dirpath)
        filenames.sort(key=sort_key)

        for filename in filenames:
            parts = filename.split('_')
            # if len(parts) >= 4:
            #     page_number = parts[2]
            # else:
            #     continue
            page_number = int(parts[0])
            # print('page:',page_number)
            textbox_filepath = os.path.join(textbox_dirpath,filename)
            result = self.ocr.ocr(textbox_filepath, cls=False)[0]
            text = ''
            if result is not None:
                for r in result:
                    text += ' '
                    text += r[1][0]
                if page_number not in page_texts:
                    page_texts[page_number] = []
                page_texts[page_number].append(text)
            else:
                # print(filename,'is None')
                pass

        with open(save_filepath, 'w', encoding='utf-8') as f:
            for page_number in sorted(page_texts.keys()):
                f.write(f'<p.{str(page_number)}>\n')
                for text in page_texts[page_number]:
                    f.write(f"{text}\n")
                f.write('\n')

if __name__=='__main__':
    # 매개변수 받기
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir')
    parser.add_argument('--save')
    # 인자 파싱
    args = parser.parse_args()

    ocr = PaddleOcr(lang='korean')
    ocr(textbox_dirpath=args.dir, save_filepath=args.save)