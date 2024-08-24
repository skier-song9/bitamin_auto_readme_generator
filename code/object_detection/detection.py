import torch
from torch.utils.data import DataLoader, Dataset
import os
import cv2
from ultralytics import YOLO
# from utils import sort_by_y_x, crop_image
import argparse
from collections import defaultdict

ROOT = 'bitamin_auto_readme_generator'

# bounding box 정렬 함수
def sort_by_y_x(boxes):
    groups = defaultdict(list)
    for item in boxes:
        y_min = item[1][1]
        found_group = False
        for key in groups:
            if abs(key - y_min) <= 100:
                groups[key].append(item)
                found_group = True
                break
        if not found_group:
            groups[y_min].append(item)

    for key in groups:
        groups[key].sort(key=lambda x: x[1][0])  # Sort by x_cord

    sorted_data = []
    for key in sorted(groups):  
        sorted_data.extend(groups[key])
    
    return sorted_data

# textbox/image bounding box 크롭 함수
def crop_image(image_path, class_id, cords):
    image = cv2.imread(image_path)
    x1, y1, x2, y2 = map(int, cords)
    cropped_image = image[y1:y2, x1:x2]
    return cropped_image

class ImageDataset(Dataset):
    def __init__(self,img_abspath, img_filenames):
        self.img_abspath = img_abspath
        self.img_filenames = img_filenames

    def __len__(self):
        return len(self.img_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_abspath,self.img_filenames[idx])
        return img_path
    
class Image_Detector:
    def __init__(self, checkpoint_path):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        #assert str(self.device).startswith('cuda'), f'Device is {self.device}, Must need cuda device!'
        self.root_absdir = os.getcwd().split(ROOT)[0]+ROOT
        self.model = YOLO(checkpoint_path)
    
    def predict(self, pdf_name, save_image_dir, save=True):
        img_abspath = os.path.join(self.root_absdir,'data','object_detection','input', pdf_name) # page별 image가 있는 폴더
        img_filenames = os.listdir(img_abspath) # page image의 각 파일명
        dataset = ImageDataset(img_abspath,img_filenames)
        dataloader = DataLoader(dataset, batch_size= len(img_filenames), shuffle=False)

        if save:
            textbox_dir = os.path.join(self.root_absdir, 'data', 'object_detection','output',pdf_name,'textbox')
            # image_dir = os.path.join(self.root_absdir,'data','object_detection','output',pdf_name,'image')
            image_dir = save_image_dir
            os.makedirs(textbox_dir, exist_ok=True)
            os.makedirs(image_dir,exist_ok=True)

        self.model.to(self.device)
        # print("model loaded..")

        for paths in dataloader:
            for path in paths:
                outputs = self.model.predict(path, conf=0.65, verbose=False)[0]
                boxes = []
                for box in outputs.boxes:
                    class_id = outputs.names[box.cls[0].item()]
                    cords = box.xyxy[0].tolist()
                    conf = round(box.conf[0].item(),2)
                    boxes.append((class_id,cords,conf))
                
                # print("boxes detected..")
            
                sorted_boxes = sort_by_y_x(boxes)
                cropped_images = []
                idx=0
                for class_id, cords, conf in sorted_boxes:
                    cropped_image = crop_image(path,class_id,cords)
                    cropped_images.append((class_id,cropped_image))
                    if save:
                        filename = os.path.splitext(os.path.basename(path))[0]
                        if class_id == 'textbox':
                            save_path = os.path.join(textbox_dir, f'{filename}_{idx}.jpg')
                            # print(save_path)
                        elif class_id == 'image':
                            save_path = os.path.join(image_dir, f'{filename}_{idx}.jpg')
                            # print(save_path)
                        cv2.imwrite(save_path, cropped_image)
                    idx += 1
                # print("Image cropped & saved..")
        # print("image detector finished")
        # return cropped_images
        return textbox_dir

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdf')
    parser.add_argument('--ckt')
    args = parser.parse_args()
    img_det = Image_Detector(args.ckt)
    img_det.predict(pdf_name=args.pdf,save=True)

if __name__ == "__main__":
    main()

    # '''
    # example>
    # python detection.py --pdf variation --ckt C:\Users\happy\Desktop\bitamin_auto_readme_generator\code\assets\best.pt
    # '''