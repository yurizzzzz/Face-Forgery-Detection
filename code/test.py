import sys
import os
import cv2
import dlib
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torchvision.transforms as transforms
import torch.optim as optim
from torchvision import utils as vutils

from tqdm import tqdm
import csv
import dataset
import model_core
from loss import am_softmax

torch.cuda.set_device(2)

if __name__ == '__main__':
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    model = model_core.Two_Stream_Net()
    model = model.cuda()

    model_state_dict = torch.load('/home/fzw/face-forgery-detection-val/checkpoint/checkpoint_9.tar', map_location='cuda:2')['state_dict']
    model.load_state_dict(model_state_dict)

    test_list = [file for file in os.listdir('/home/fzw/face/image/test/') if file.endswith('.jpg')]
    test_list = tqdm(test_list)

    t_list = [transforms.ToTensor()]
    composed_transform = transforms.Compose(t_list)

    f = open('predict.csv', 'w', encoding='utf-8', newline="")
    csv_write = csv.writer(f)
    csv_write.writerow(["fname", "label"])

    label_dict = dict()

    for filename in test_list:
        model.eval()
        face_detect = dlib.get_frontal_face_detector()
        img = cv2.imread('/home/fzw/face/image/test/' + filename)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detect(gray, 1)
        if len(faces) != 0:
            face = faces[0]
            x, y, size = dataset.get_boundingbox(face, 1024, 1024)
            cropped_face = img[y:y+size, x:x+size]
            cropped_face = cv2.resize(cropped_face, (256, 256))
            cropped_face = composed_transform(cropped_face)

        else:
            cropped_face = cv2.resize(img, (256, 256))
            cropped_face = composed_transform(cropped_face)

        cropped_face = cropped_face.unsqueeze(0)
        cropped_face = cropped_face.to(device)

        with torch.no_grad():
            test_output = model(cropped_face)

        _, test_predict = torch.max(test_output.data, 1)
        # desc = 'Filename: %s, Predict = %d' % (filename, test_predict.cpu().numpy()[0])
        # test_list.set_description(desc)
        # test_list.update()
        label_dict.update({filename: test_predict.cpu().numpy()[0]})

    for i in range(len(test_list)):
        fname = 'test_' + str(i) + '.jpg'
        fname_label = label_dict[fname]
        csv_write.writerow([fname, str(fname_label)])
