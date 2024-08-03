import torch
import os
from ultralytics import YOLO
from utils import sort_by_y_x, save_cropped_image
current_dir = os.path.dirname(__file__)

# Check if CUDA is available and set the device accordingly
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load finetuned model
class_list = ['textbox', 'image']
best_pt_path = os.path.join(current_dir, '..', 'assets', 'best.pt')
model = YOLO(best_pt_path)
model.to(device)

# Set directory
# 원하는 pdf의 이미지가 담겨있는 폴더명으로 바꾸기
folder_name = 'webtoon-rec'
image_dir =  os.path.join(current_dir, '..','..','data','object_detection','input',folder_name)
textbox_save_dir = os.path.join(current_dir, '..','..','data','object_detection','output',folder_name,'textbox')
image_save_dir = os.path.join(current_dir, '..','..','data','object_detection','output',folder_name,'image')


for filename in os.listdir(image_dir):
   if filename.endswith('.jpg'):
    image_path = os.path.join(image_dir,filename)
    result = model.predict(image_path, conf=0.5)[0] #cf level 조정 가능

    boxes = []
    for box in result.boxes:
        class_id = result.names[box.cls[0].item()]
        cords = box.xyxy[0].tolist()
        conf = round(box.conf[0].item(), 2)
        boxes.append((class_id,cords,conf))
    # print(boxes)
    
    sorted_boxes = sort_by_y_x(boxes)
    idx=0
    for class_id,cords,conf in sorted_boxes:
        print("Object type: ", class_id)
        print("Coordinates: ", cords)
        print("Probability: ", conf)
        if class_id == 'textbox':
            save_cropped_image(image_path,textbox_save_dir,class_id,cords,idx)
        elif class_id == 'image':
            save_cropped_image(image_path, image_save_dir,class_id,cords,idx)
        idx+=1

    print(f"{filename} detection done...")

