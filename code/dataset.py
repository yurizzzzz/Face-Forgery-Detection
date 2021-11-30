import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import random
import dlib
import torch
import cv2


def get_boundingbox(face, width, height, scale=1.3, minsize=None):
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    size_bb = int(max(x2 - x1, y2 - y1) * scale)
    if minsize:
        if size_bb < minsize:
            size_bb = minsize
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)

    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)

    return x1, y1, size_bb


class LoadData(Dataset):
    def __init__(self, img, label_dict, mode='train'):
        self.img = img
        self.label_dict = label_dict
        self.mode = mode

    def __getitem__(self, item):
        input_img = self.img[item]

        t_list = [transforms.ToTensor()]
        composed_transform = transforms.Compose(t_list)

        if self.mode == 'train':
            face_detect = dlib.get_frontal_face_detector()
            img = cv2.imread('/home/fzw/face/image/train/' + input_img)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_detect(gray, 1)
            if len(faces) != 0:
                face = faces[0]
                x, y, size = get_boundingbox(face, 1024, 1024)
                cropped_face = img[y:y+size, x:x+size]
                cropped_face = cv2.resize(cropped_face, (256, 256))
                cropped_face = composed_transform(cropped_face)

                label = self.label_dict[input_img]
                label = torch.LongTensor([label])
            else:
                cropped_face = cv2.resize(img, (256, 256))
                cropped_face = composed_transform(cropped_face)
                label = self.label_dict[input_img]
                label = torch.LongTensor([label])
        
        if self.mode == 'val':
            face_detect = dlib.get_frontal_face_detector()
            img = cv2.imread('/home/fzw/face/image/val/' + input_img)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_detect(gray, 1)
            if len(faces) != 0:
                face = faces[0]
                x, y, size = get_boundingbox(face, 1024, 1024)
                cropped_face = img[y:y+size, x:x+size]
                cropped_face = cv2.resize(cropped_face, (256, 256))
                cropped_face = composed_transform(cropped_face)

                label = self.label_dict[input_img]
                label = torch.LongTensor([label])
            else:
                cropped_face = cv2.resize(img, (256, 256))
                cropped_face = composed_transform(cropped_face)
                label = self.label_dict[input_img]
                label = torch.LongTensor([label])


        return cropped_face, label

    def __len__(self):
        return len(self.img)


