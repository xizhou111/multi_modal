import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import cv2

def resize_and_pad(image_path, target_size=448):
    # 读取图像
    image = cv2.imread(image_path)
    # 获取原始图像的尺寸
    h, w = image.shape[:2]
    # 计算缩放比例，使得长边变为 target_size
    if h > w:
        scale = target_size / h
        new_h, new_w = target_size, int(w * scale)
    else:
        scale = target_size / w
        new_h, new_w = int(h * scale), target_size
    # 调整图像大小
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    # 创建一个新的白色背景图像
    padded_image = np.ones((target_size, target_size, 3), dtype=np.uint8) * 255
    # 将调整大小后的图像粘贴到白色背景上
    start_y = (target_size - new_h) // 2
    start_x = (target_size - new_w) // 2
    padded_image[start_y:start_y+new_h, start_x:start_x+new_w] = resized_image
    return padded_image

class MultiModelDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform or transforms.Compose([
            transforms.ToTensor() # dtype=torch.float32
        ])
    
    def resize_and_pad(self, image_path, target_size=None):
        # 读取图像
        image = cv2.imread(image_path)
        # 获取原始图像的尺寸
        h, w = image.shape[:2]
        # 计算缩放比例，使得长边变为 target_size
        if h > w:
            scale = target_size / h
            new_h, new_w = target_size, int(w * scale)
        else:
            scale = target_size / w
            new_h, new_w = int(h * scale), target_size
        # 调整图像大小
        resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        # 创建一个新的白色背景图像
        padded_image = np.ones((target_size, target_size, 3), dtype=np.uint8) * 255
        # 将调整大小后的图像粘贴到白色背景上
        start_y = (target_size - new_h) // 2
        start_x = (target_size - new_w) // 2
        padded_image[start_y:start_y+new_h, start_x:start_x+new_w] = resized_image
        return padded_image

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data[idx]['image']
        text = self.data[idx]['text_input']
        image_ids = self.data[idx]['image_id']

        if image_path != '':
            # 使用 resize_and_pad 函数预处理图像到 448x448 大小
            try:
                image = self.resize_and_pad(image_path, target_size=448)
                # 将 NumPy 数组转换为 PIL 图像，以便可以使用 torchvision.transforms
                image = Image.fromarray(image)
                image = self.transform(image) # dtype=torch.float32
            except:
                image = torch.zeros((3, 448, 448), dtype=torch.float32)
                print('image_path not exists:', image_path)
        else:
            image = torch.zeros((3, 448, 448), dtype=torch.float32)

        if text == '':
            text = ' '

        sample = {
            'image': image,
            'text_input': text,
            'image_id': image_ids
        }

        return sample

if __name__ == '__main__':
    image_path = '/mnt/cfs/CV/xuyuan/data/L3_ocr_data/images/data_1/202312-01-b52b72c42a9e600e91783903c1f59a66.jpeg'

    image = resize_and_pad(image_path, target_size=378)

    cv2.imwrite('resized_image.jpg', image)
