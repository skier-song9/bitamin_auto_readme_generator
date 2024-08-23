from paddleocr import PaddleOCR
import time
import os
current_dir = os.path.dirname(__file__)
# set up model 
ocr = PaddleOCR(lang='korean')

starttime = time.time()

folder_name = 'lier-detector'
image_dir = os.path.join(current_dir, '..','..', 'data', 'object_detection','output',folder_name,'textbox' )
output_dir = os.path.join(current_dir, '..','..', 'data', 'object_detection','output',folder_name,'ocr_result' )

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

page_texts = {}
def sort_key(filename):
        match = re.search(r'\d+', filename)
        return int(match.group()) if match else 0
# Get a sorted list of filenames
# filenames = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
# filenames.sort(key=lambda x: (int(x.split('_')[0]), int(x.split('_')[1])))
filenames = os.listdir(image_dir)
filenames.sort(key=sort_key)

for filename in filenames:
    parts=filename.split('_')
    if len(parts)>=4:
        page_number=parts[2]
    else:
        continue
    image_path=os.path.join(image_dir,filename)
    result = ocr.ocr(image_path,cls=False)[0]
    text = ''
    for re in result:
        text += ' '
        text += re[1][0]
    if page_number not in page_texts:
        page_texts[page_number]=[]
    page_texts[page_number].append(text)
    print(f"img{filename} processing is done...")

with open(f'{output_dir}/paddle-ocr_{folder_name}_text.txt','w',encoding='utf-8') as file:
    for page_number in sorted(page_texts.keys()):
        file.write(f'p.{page_number}\n')
        for text in page_texts[page_number]:
            file.write(f'{text}\n')


processing_time = time.time() - starttime
print(processing_time)