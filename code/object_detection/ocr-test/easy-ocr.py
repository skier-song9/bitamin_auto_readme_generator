import easyocr
import time
import os
current_dir = os.path.dirname(__file__)

reader = easyocr.Reader(['ko','en'],gpu=False) # this needs to run only once to load the model into memory

starttime = time.time()

folder_name = 'lier-detector'
image_dir = os.path.join(current_dir, '..','..', 'data', 'object_detection','output',folder_name,'textbox' )
output_dir = os.path.join(current_dir, '..','..', 'data', 'object_detection','output',folder_name,'ocr_result' )

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

page_texts = {}

# Get a sorted list of filenames
filenames = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
filenames.sort(key=lambda x: (int(x.split('_')[2]), int(x.split('_')[3].split('.')[0])))

for filename in filenames:
    parts = filename.split('_')
    if len(parts) >= 4:
        page_number = parts[2]
    else:
        continue

    image_path = os.path.join(image_dir,filename)
    result = reader.readtext(image_path,detail=0)
    text = ' '.join(result)

    if page_number not in page_texts:
        page_texts[page_number]=[]
    page_texts[page_number].append(text)
    print(f"img{filename} processing is done...")
      
with open(f'{output_dir}/easy-ocr_{folder_name}_text.txt','w',encoding='utf-8') as file:
    for page_number in sorted(page_texts.keys()):
        file.write(f'p.{page_number}\n')
        for text in page_texts[page_number]:
            file.write(f'{text}\n')

processing_time = time.time() - starttime
print(processing_time)
