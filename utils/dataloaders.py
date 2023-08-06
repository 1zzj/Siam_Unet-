import os
import torch.utils.data as data
from PIL import Image
import utils.transforms as tr
import cv2 as cv
import numpy as np
'''
Load all training and validation data paths
'''
def full_path_loader(data_dir):
    # 图片名
    train_data = [i for i in os.listdir(data_dir + 'train/A/') if not   # os.listdir文件路径顺序是混乱的
    i.startswith('.')]
    train_data.sort()

    valid_data = [i for i in os.listdir(data_dir + 'val/A/') if not
    i.startswith('.')]
    valid_data.sort()

    # 标签路径+名
    train_label_paths = []
    val_label_paths = []
    # for img in train_data:
    #     train_label_paths.append(data_dir + 'train/label/' + img)
    # for img in valid_data:
    #     val_label_paths.append(data_dir + 'val/label/' + img)
    for img in train_data:
        train_label_paths.append(data_dir + 'train/label/' + img)
    for img in valid_data:
        val_label_paths.append(data_dir + 'val/label/' + img)


    #图片路径+名
    train_data_path = []
    val_data_path = []

    for img in train_data:
        train_data_path.append([data_dir + 'train/', img])
    for img in valid_data:
        val_data_path.append([data_dir + 'val/', img])

    # 存放dataset的完整路径
    train_dataset = {}
    val_dataset = {}
    for cp in range(len(train_data)):
        train_dataset[cp] = {'image': train_data_path[cp],
                         'label': train_label_paths[cp]}
    for cp in range(len(valid_data)):
        val_dataset[cp] = {'image': val_data_path[cp],
                         'label': val_label_paths[cp]}


    return train_dataset, val_dataset  # 返回字典列表[{'image':,'label':},{},{}]

'''
Load all testing data paths
'''
def full_test_loader(data_dir):

    test_data = [i for i in os.listdir(data_dir + 'test/A/') if not         # os.listdir文件路径顺序是混乱的
                    i.startswith('.')]
    test_data.sort()

    test_label_paths = []
    # for img in test_data:
    #     test_label_paths.append(data_dir + 'test/label/' + img)
    for img in test_data:
        test_label_paths.append(data_dir + 'test/label/' + img)

    test_data_path = []
    for img in test_data:
        test_data_path.append([data_dir + 'test/', img])

    test_dataset = {}
    for cp in range(len(test_data)):
        test_dataset[cp] = {'image': test_data_path[cp],
                           'label': test_label_paths[cp]}

    return test_dataset

def cdd_loader(img_path, label_path, aug):
    dir = img_path[0]
    name = img_path[1]
    # 图片归一化到[0,1]
    img1 = cv.imread(dir + 'A/' + name,1)/255
    # img1 = Image.fromarray(np.uint8(img1))
    # img1 = Image.fromarray(img1)     # PIL.Image 数据是 uinit8 型的
    # img1 = Image.open(dir + 'A/' + name)
    img2 = cv.imread(dir + 'B/' + name,1)/255
    # img2 = Image.fromarray(np.uint8(img2))
    # img2 = Image.fromarray(img2)
    # img2 = Image.open(dir + 'B/' + name)
    # 标签二值化
    label = cv.imread(label_path,0)  # 按灰度读取
    # label = np.where(label > 128, 1, 0)  #
    # label = Image.fromarray(np.uint8(label))     # 转PIL Image类型
    # label = Image.open(label_path).convert('L')

    sample = {'image': (img1, img2), 'label': label}

    if aug:                                    # 数据增强
        sample = tr.train_transforms(sample)
    else:
        sample = tr.test_transforms(sample)

    return sample['image'][0], sample['image'][1], sample['label']


class CDDloader(data.Dataset):

    def __init__(self, full_load, aug=False):

        self.full_load = full_load   # 字典列表[{'image':,'label':},{},{}] 所有图片的路径
        self.loader = cdd_loader     # 函数
        self.aug = aug               # 是否数据增强

    def __getitem__(self, index):

        img_path, label_path = self.full_load[index]['image'], self.full_load[index]['label']

        return self.loader(img_path,
                           label_path,
                           self.aug)

    def __len__(self):
        return len(self.full_load)

if __name__ == '__main__':
    test_path = full_test_loader("G:\\DataSet\\LEVIR-CD\\subset\\")
    for i in range(len(test_path)):
        print(test_path[i]['label'] )
    img = cdd_loader(test_path[1]['image'],test_path[1]['label'],False)
    print(img[2].shape)