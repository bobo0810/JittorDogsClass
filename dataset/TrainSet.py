import os
import jittor as jt
from jittor import transform
from jittor.dataset import Dataset
import numpy as np
from PIL import Image
import albumentations as A
import socket
import cv2
import random
import yaml
import time
import numpy as np

cur_path = os.path.abspath(os.path.dirname(__file__))
# https://blog.csdn.net/qq_27039891/article/details/100795846
def strong_aug(p=0.5):
    return A.Compose([
        # 亮度、对比度
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=None, always_apply=False,
                                   p=0.5),
        # 旋转
        A.Rotate(limit=30, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=0.5)
    ], p=p)


class TsinghuaDog(Dataset):
    def __init__(self,img_size=None,mode=False,shuffle=None,batch_size=None,num_workers=0,pad=True,crop_type=None,data_conf='data_conf.yaml'):
        super().__init__()
        assert mode in ['train', 'eval']
        assert crop_type in ['head','body']
        self.pad = pad
        self.mode = mode
        self.crop_type = crop_type

        # 参数
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

        self.augmentation = strong_aug(p=0.5)
        data_dict = self.load_params(data_conf)
        file = cur_path + '/' + "all_imgs.txt"
        with open(file, 'r') as fid:
            imglist = fid.readlines()
        random.shuffle(imglist)  # 打乱顺序
        self.imglist = imglist[:int(len(imglist) * 0.8)] if self.mode == 'train' else imglist[int(
                                len(imglist) * 0.8):]  # 划分 训练集和验证集

        self.image_list = []
        self.boxes = []
        self.id_list = []
        self.root_path = data_dict['Train'] + '/'

        if 'body' in self.crop_type:
            img_index=[2,3,4,5]      # 狗身
        elif 'head' in self.crop_type:
            img_index = [6, 7, 8, 9] # 狗头
        else:
            raise ValueError


        for line in self.imglist:
            context = line.strip().split()
            self.image_list.append( context[0])
            self.id_list.append(int(context[1]))
            box = []
            box.append(int(context[img_index[0]]))
            box.append(int(context[img_index[1]]))
            box.append(int(context[img_index[2]]))
            box.append(int(context[img_index[3]]))
            self.boxes.append(box)


        if self.mode == "train":
            self.transform = transform.Compose([
                transform.Resize((int(img_size[0] * 1.3), int(img_size[1] * 1.3))),
                transform.RandomCrop(tuple(img_size)),
                transform.RandomHorizontalFlip(),
                transform.ToTensor(),
                transform.ImageNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                                         )])
        elif self.mode == "eval":
            self.transform = transform.Compose([
                transform.Resize(tuple(img_size)),
                transform.ToTensor(),
                transform.ImageNormalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)
                )])
        else:
            pass

        self.set_attrs(
            batch_size=self.batch_size,
            total_len=len(self.id_list),
            shuffle=self.shuffle,
            num_workers=self.num_workers,  # 线程数
            )
    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        if 'Fila_Braziliero' in image_name:
            image_name=image_name.replace('Fila_Braziliero','Fila Braziliero')

        label = self.id_list[idx]

        image = cv2.imread(self.root_path + image_name)
        # 裁剪出狗头or狗身
        box = self.boxes[idx]
        image = image[box[1]:box[3], box[0]:box[2]]

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 训练集增广
        if self.mode == "train":
            image = self.augmentation(image=image)['image']

        # pad成正方形
        if self.pad:
            rows, cols, _ = image.shape
            if cols > rows:
                top = int((cols - rows) / 2)
                bottom = cols - top - rows
                image = cv2.copyMakeBorder(image, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            elif rows > cols:
                left = int((rows - cols) / 2)
                right = rows - left - cols
                image = cv2.copyMakeBorder(image, 0, 0, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        image = Image.fromarray(image)
        image = self.transform(image)

        return image, label

    @staticmethod
    def load_params(data_conf):
        '''
        读取yaml,加载数据集
        '''
        file = open(cur_path + '/' + data_conf, 'r', encoding="utf-8")
        data_conf = yaml.load(file, Loader=yaml.FullLoader)

        data_dict = {}
        data_dict['Train'] = data_conf['Train']['default']
        return data_dict


class BalancedBatchSampler_TsinghuaDog(TsinghuaDog):
    '''
    batch内平衡采样类别
    '''
    def __init__(self,  n_classes, n_samples,img_size,mode,shuffle,num_workers,crop_type):
        # 初始化父类TsinghuaDog属性

        super().__init__(img_size=img_size,mode=mode,shuffle=shuffle,batch_size=n_classes * n_samples,num_workers=num_workers,crop_type=crop_type)


        self.labels = np.array(self.id_list)
        self.labels_set = list(set(self.labels))
        self.label_to_indices = {label: np.where(self.labels == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples

        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        if self.total_len is None:
            self.total_len = len(self)
        self.count = 0
        while self.count + self.batch_size < len(self.labels):
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            index_list = indices
            if jt.in_mpi:
                world_size = mpi.world_size()
                world_rank = mpi.world_rank()
                index_list = np.int32(index_list)
                mpi.broadcast(index_list, 0)

                assert self.batch_size >= world_size, \
                    f"Batch size({self.batch_size}) is smaller than MPI world_size({world_size})"
                real_batch_size = (self.batch_size - 1) // world_size + 1
                if real_batch_size * world_size != self.batch_size:
                    LOG.w("Batch size is not divisible by MPI world size, "
                          "The distributed version may be different from "
                          "the single-process version.")
                fix_batch = self.total_len // self.batch_size
                last_batch = self.total_len - fix_batch * self.batch_size
                fix_batch_l = index_list[0:fix_batch * self.batch_size] \
                    .reshape(-1, self.batch_size)
                fix_batch_l = fix_batch_l[
                              :, real_batch_size * world_rank:real_batch_size * (world_rank + 1)]
                real_batch_size = fix_batch_l.shape[1]
                fix_batch_l = fix_batch_l.flatten()
                if not self.drop_last and last_batch > 0:
                    last_batch_l = index_list[-last_batch:]
                    real_last_batch = (last_batch - 1) // world_size + 1
                    l = real_last_batch * world_rank
                    r = l + real_last_batch
                    if r > last_batch: r = last_batch
                    if l >= r: l = r - 1
                    index_list = np.concatenate([fix_batch_l, last_batch_l[l:r]])
                else:
                    index_list = fix_batch_l

                self.real_len = len(index_list)
                self.real_batch_size = real_batch_size
                assert self.total_len // self.batch_size == \
                       self.real_len // self.real_batch_size, f"Number of batches({self.total_len // self.batch_size}!={self.real_len // self.real_batch_size}) not match, total_len: {self.total_len}, batch_size: {self.batch_size}, real_len: {self.real_len}, real_batch_size: {self.real_batch_size}"
            else:
                self.real_len = self.total_len
                self.real_batch_size = self.batch_size

            self.batch_len = self.__batch_len__()

            if not hasattr(self, "workers") and self.num_workers:
                self._init_workers()
            self.num_workers = False
            if self.num_workers:
                self._stop_all_workers()
                self.index_list_numpy[:] = index_list
                gid_obj = self.gid.get_obj()
                gid_lock = self.gid.get_lock()
                with gid_lock:
                    gid_obj.value = 0
                    self.gidc.notify_all()
                start = time.time()
                self.batch_time = 0
                for i in range(self.batch_len):
                    # try not get lock first
                    if gid_obj.value <= i:
                        with gid_lock:
                            if gid_obj.value <= i:
                                if mp_log_v:
                                    print("wait")
                                self.gidc.wait()
                    now = time.time()
                    self.wait_time = now - start
                    start = now

                    self.last_id = i
                    worker_id = self.idmap[i]
                    w = self.workers[worker_id]
                    if mp_log_v:
                        print(f"#{worker_id} {os.getpid()} recv buffer", w.buffer)
                    batch = w.buffer.recv()
                    now = time.time()
                    self.recv_time = now - start
                    start = now

                    if mp_log_v:
                        print(f"#{worker_id} {os.getpid()} recv", type(batch).__name__,
                              [type(b).__name__ for b in batch])
                    batch = self.to_jittor(batch)
                    now = time.time()
                    self.to_jittor_time = now - start
                    start = now

                    yield batch

                    now = time.time()
                    self.batch_time = now - start
                    start = now
            else:
                batch_data = []
                for idx in index_list:
                    batch_data.append(self[int(idx)])
                    if len(batch_data) == self.real_batch_size:
                        batch_data = self.collate_batch(batch_data)
                        batch_data = self.to_jittor(batch_data)
                        yield batch_data
                        batch_data = []

                # depend on drop_last
                if not self.drop_last and len(batch_data) > 0:
                    batch_data = self.collate_batch(batch_data)
                    batch_data = self.to_jittor(batch_data)
                    yield batch_data

            self.count += self.n_classes * self.n_samples



# if __name__ == '__main__':
#     img_size = [1024, 1024]
#     mode = "eval"
#     shuffle = True
#     n_classes = 3
#     n_samples = 2
#     num_workers=3
#     train_dataloader = BalancedBatchSampler_TsinghuaDog(n_classes=n_classes, n_samples=n_samples,
#                                                         img_size=img_size, mode=mode, shuffle=shuffle,
#                                                         num_workers=num_workers,crop_type='body')
#
#     index = 0
#     for img, label in train_dataloader:
#         # 打乱BN  eg: [ 25  25 102 102  94  94] -> [102  25  94  25  94 102]
#         batch_size=  list(range(len(label)))
#         random.shuffle(batch_size)
#         img = img[batch_size]
#         label=label[batch_size]



