import torch
from torch.utils.data import DataLoader, Dataset
import os
import cv2
from ultralytics import YOLO
from utils import sort_by_y_x, crop_image
import argparse

ROOT = 'bitamin_auto_readme_generator'

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
    
    def predict(self, pdf_name, save=True):
        img_abspath = os.path.join(self.root_absdir,'data','object_detection','input', pdf_name)
        img_filenames = os.listdir(img_abspath)
        dataset = ImageDataset(img_abspath,img_filenames)
        dataloader = DataLoader(dataset, batch_size= len(img_filenames), shuffle=False)

        if save:
            textbox_dir = os.path.join(self.root_absdir, 'data', 'object_detection','output',pdf_name,'textbox')
            image_dir = os.path.join(self.root_absdir,'data','object_detection','output',pdf_name,'image')
            os.makedirs(textbox_dir, exist_ok=True)
            os.makedirs(image_dir,exist_ok=True)

        self.model.to(self.device)
        print("model loaded..")

        for paths in dataloader:
            for path in paths:
                outputs = self.model.predict(path, conf=0.65)[0]
                boxes = []
                for box in outputs.boxes:
                    class_id = outputs.names[box.cls[0].item()]
                    cords = box.xyxy[0].tolist()
                    conf = round(box.conf[0].item(),2)
                    boxes.append((class_id,cords,conf))
                
                print("boxes detected..")
            
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
                            print(save_path)
                        elif class_id == 'image':
                            save_path = os.path.join(image_dir, f'{filename}_{idx}.jpg')
                            print(save_path)
                        cv2.imwrite(save_path, cropped_image)
                    idx += 1
                print("Image cropped & saved..")
            
        return cropped_images

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdf')
    parser.add_argument('--ckt')
    args = parser.parse_args()
    img_det = Image_Detector(args.ckt)
    img_det.predict(pdf_name=args.pdf,save=True)

if __name__ == "__main__":
    main()

'''
example>
python detection.py --pdf variation --ckt C:\Users\happy\Desktop\bitamin_auto_readme_generator\code\assets\best.pt
'''