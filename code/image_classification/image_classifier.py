import pandas as pd
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import os, time
from PIL import Image
import timm
import argparse
import stat

ROOT = 'bitamin_auto_readme_generator'
CLASS_NAMES = ['chart','diagram','none','table']

def plot_image_from_path(img_path):
    img = Image.open(img_path)
    img = transforms.Compose([transforms.ToTensor()])(img)
    plt.imshow(img.permute(1,2,0))
    plt.axis('off')
    plt.show()

def softmax(input):
        c = np.max(input)
        exp_input = np.exp(input-c)
        sum_exp_input = np.sum(exp_input)
        y = exp_input / sum_exp_input
        return y

### Custom Transformation class
class ResizeWithPadding:
    def __init__(self, size=(244,244), bg_color=(255,255,255), fill=0):
        self.size = size
        self.bg_color = bg_color
        self.fill = fill

    def __call__(self, image):
        original_ratio = image.width / image.height
        output_ratio = self.size[0] / self.size[1]
        # Determine the new size that fits within the output size while maintaining the aspect ratio
        if original_ratio > output_ratio:
            # Fit to width
            new_width = self.size[0]
            new_height = int(new_width / original_ratio)
        else:
            # Fit to height
            new_height = self.size[1]
            new_width = int(new_height * original_ratio)
        
        # Resize the image
        resized_image = image.resize((new_width, new_height), Image.LANCZOS)

        # Create a new image with the desired size and background color
        new_image = Image.new("RGB", self.size, self.bg_color)
        
        # Paste the resized image onto the center of the new image
        paste_position = ((self.size[0] - new_width) // 2, (self.size[1] - new_height) // 2)
        new_image.paste(resized_image, paste_position)
    
        return new_image
        
class ImageDataset(Dataset):
    def __init__(self,img_abspath, img_filenames, transform=None):
        self.img_abspath = img_abspath
        self.img_filenames = img_filenames
        self.transform = transform

    def __len__(self):
        return len(self.img_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_abspath,self.img_filenames[idx])
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return image, img_path

class Image_Classifier:
    def __init__(self, clf_params_path):
        '''
        clf_params_path : fine-tuning한 vgg19 모델 classifier의 parameters 파일 경로 -> 상대경로, 절대경로 모두 가능(단, 상대경로 시 image_classifier.py 파일을 import하는 파일의 위치가 기준이 됨. e.g. main.py에서 해당 클래스를 객체화했다면 상대경로는 main.py를 현위치로 인식함.)
        '''
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        assert str(self.device).startswith('cuda'), f"Device is {self.device}, MUST NEED CUDA DEVICE!"

        # 프로젝트 root의 절대경로 : ~~~/bitamin_auto_readme_generator/
        self.root_absdir = os.getcwd().split(ROOT)[0]+ROOT
        
        # Load pretrained vgg19
        self.model = timm.create_model('vgg19.tv_in1k',pretrained=True)
        # modify model
        self.model.reset_classifier(num_classes = len(CLASS_NAMES))
        # load classifier parameter weights
        self.model.get_classifier().load_state_dict(torch.load(clf_params_path))

        # set transform for inference
        self.transform_infer = transforms.Compose([
            ResizeWithPadding(size=(244,244)),
            transforms.ToTensor()
        ])
    
    def predict(self, image_dirpath, save = False):
        '''
        # pdf_name : image classification을 진행할 프로젝트의 이름 -> 함수 내부에서 root/data/object_detection/output/프로젝트이름으로 결합하여 사용함.
        image_dirpath : 이미지가 저장되어 있는 Flask 서버의 디렉토리 경로를 전달 >> 바로 사용 가능
        save : True면 classification 결과를 data/image_classification/output에 저장함.

        return : 
            - result_df : 추론 결과를 저장한 데이터프레임
            - len(img_filenames) : 추론한 이미지 개수를 반환
        '''
        # img_abspath = os.path.join(self.root_absdir,'data','object_detection','output',pdf_name,'image')
        img_filenames = os.listdir(image_dirpath) # for loading images

        # Set dataset
        # dataset = ImageDataset(img_abspath, img_filenames, transform = self.transform_infer)
        dataset = ImageDataset(image_dirpath, img_filenames, transform = self.transform_infer)
        dataloader = DataLoader(dataset, batch_size = len(img_filenames), shuffle=False)
        
        self.model.to(self.device).eval()
        results = []

        with torch.no_grad():
            for imgs, paths in dataloader:
                imgs = imgs.to(self.device)
                outputs = self.model(imgs)
                for i in range(len(paths)):
                    result = outputs[i].cpu().numpy()
                    class_name = CLASS_NAMES[np.argmax(result)]
                    probs = softmax(result).tolist()
                    fname = paths[i].split(os.sep)[-1] # .replace('.','_')
                    results.append([fname, class_name]+probs)
        result_df = pd.DataFrame(data=results, columns=['filename','class']+CLASS_NAMES)
        ### [이미지 파일 이름, 예측 클래스, 예측 클래스에 대한 각 확률값] 을 컬럼으로 저장.

        if save:
            output_dir = os.path.join(self.root_absdir,'data','image_classification','output')
            lt = time.localtime()
            save_path = os.path.join(output_dir,f"{image_dirpath.split(os.sep)[-1]}_{lt.tm_mon}{lt.tm_mday}_{lt.tm_hour}{lt.tm_min}.csv")
            
            result_df.to_csv(save_path,index=False)

        return result_df, dataset.__len__()

            
### ↓↓↓ test를 위한 코드 ↓↓↓
if __name__ == '__main__':
    #  pdf_name, clf_params_path
    start_time = time.time()
    # 매개변수 받기
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdf')
    parser.add_argument('--clf')
    # 인자 파싱
    args = parser.parse_args()

    img_clf = Image_Classifier(args.clf)
    _, length=img_clf.predict(pdf_name = args.pdf,save = True)
    print(f"✅ {length} images, finished in {time.time()-start_time:.3f} seconds")
    '''    
    python image_classifier.py --pdf 넷플릭스주가 --clf D:\SKH\Github_Projects\bitamin_auto_readme_generator\code\assets\VGG19clf_acc94_0728.pth

    실험 결과
    - DataLoader를 사용했을 때, 40개 이미지 추론 시간 : 2.614초
    - DataLoader 미사용 시, 40개 이미지 추론 시간 : 4.5초

    22, 35개 이미지도 평균적으로 2.8~3.7초 소요
    '''
        
        