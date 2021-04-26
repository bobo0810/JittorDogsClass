import cv2
import socket
import yaml
import os
from jittor import transform
from jittor.dataset import Dataset
import numpy as np
import jittor as jt
from PIL import Image
cur_path = os.path.abspath(os.path.dirname(__file__))
def default_loader(path):
    try:
        img = cv2.imread(path)
    except:
        with open('read_error.txt', 'a') as fid:
            fid.write(path+'\n')
        return Image.new('RGB', (224,224), 'white')
    return img



class TestDataset(Dataset):
    def __init__(self,img_size=None,pad=True,batch_size=None,dataloader=default_loader,crop_type=None,data_conf='data_conf.yaml'):
        super().__init__()
        assert crop_type in ['head', 'body']
        self.crop_type = 'Head'  if 'head' in  crop_type else 'Body'

        self.dataloader = dataloader
        self.pad = pad
        self.batch_size=batch_size

        data_dict = self.load_params(data_conf,self.crop_type)
        self.imglist = os.listdir(data_dict['Test'])
        self.imgs = []
        self.root_path = data_dict['Test'] + '/'
        for img in self.imglist:
            self.imgs.append(img)

        self.transform = transform.Compose([
            transform.Resize( tuple(img_size)),
            # transform.RandomHorizontalFlip(p=1),
            transform.ToTensor(),
            transform.ImageNormalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            )])

        # 加载参数
        self.set_attrs(
            batch_size=self.batch_size,
            total_len=len(self.imglist),
            shuffle=False,
            num_workers=3,
        )

    def __getitem__(self, index):
        image_name = self.imgs[index]
        image = image_name
        img = self.dataloader(self.root_path + image)
        # 填充为正方形
        if self.pad:
            rows, cols, _ = img.shape
            if cols > rows:
                top = int((cols - rows) / 2)
                bottom = cols - top - rows
                img = cv2.copyMakeBorder(img, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            elif rows > cols:
                left = int((rows - cols) / 2)
                right = rows - left - cols
                img = cv2.copyMakeBorder(img, 0, 0, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = self.transform(img)
        # img = np.asarray(img)
        return img, image_name

    @staticmethod
    def load_params(data_conf,crop_type):
        '''
        读取yaml,加载数据集
        '''
        file = open(cur_path+'/'+data_conf, 'r',encoding="utf-8")
        data_conf = yaml.load(file, Loader=yaml.FullLoader)

        data_dict = {}
        data_dict['Test'] = data_conf['Test'][crop_type]['default']
        return data_dict

# if __name__=='__main__':
#     jt.flags.use_cuda = 1
#     dataloader =TestDataset(img_size=[368,368],batch_size=200,crop_type='head')
#     for batch_idx, [img, image_name] in enumerate(dataloader):
#         print(img.shape)
#         print(image_name)
#         print('--------')