import sys
import os
if __name__ == '__main__':
    sys.path.append(os.path.dirname(sys.path[0]))

import cv2
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
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

    csvFile = open("/home/fzw/face/train.labels.csv", "r")
    reader = csv.reader(csvFile)
    label_dict = dict()
    for item in reader:
        key = item[-1][:-2]
        value = item[-1][-1]
        if value != 'l':
            value = int(value)
            label_dict.update({key: value})

    train_list = [file for file in os.listdir('/home/fzw/face/image/train/') if file.endswith('.jpg')]
    val_list = [file for file in os.listdir('/home/fzw/face/image/val/') if file.endswith('.jpg')]
    TrainData = torch.utils.data.DataLoader(dataset.LoadData(train_list, label_dict, mode='train'),
                                            batch_size=16,
                                            shuffle=True,
                                            num_workers=16,
                                            drop_last=False)
    ValData = torch.utils.data.DataLoader(dataset.LoadData(val_list, label_dict, mode='val'),
                                            batch_size=16,
                                            shuffle=True,
                                            num_workers=16,
                                            drop_last=False)
    

    model = model_core.Two_Stream_Net()
    model = model.cuda()
    model_state_dict = torch.load('/home/fzw/face-forgery-detection-val/checkpoint/checkpoint_9.tar', map_location='cuda:2')['state_dict']
    model.load_state_dict(model_state_dict)
    optimizer = optim.Adam(model.parameters(), lr=0.0002, betas=(0.9, 0.999))

    epoch = 0 
    
    while epoch < 10:
        count = 0
        total_loss = 0
        correct = 0
        train_bar = tqdm(TrainData)
        total = 0

        for batch_idx, (input_img, img_label) in enumerate(train_bar):
            count = count + 1

            model.train()
            input_img = input_img.to(device)
            img_label = img_label.to(device)

            outputs= model(input_img)
            optimizer.zero_grad()

            amloss = am_softmax.AMSoftmaxLoss()
            img_label = img_label.squeeze()
            loss = amloss(outputs, img_label)
            loss.backward()
            optimizer.step()

            total_loss = total_loss + loss
            avg_loss = total_loss / count
            _, predict = torch.max(outputs.data, 1)
            print(outputs)
            correct += predict.eq(img_label.data).cpu().sum()
            total = total + img_label.size(0)
            correct_per = 100.0 * correct / total

            desc = 'Training : Epoch %d, AvgLoss = %.4f, AC = %.4f' % (epoch, avg_loss, correct_per)
            train_bar.set_description(desc)
            train_bar.update()
        
        val_correct = 0
        val_total = 0
        val_bar = tqdm(ValData)
        for batch_idx, (val_input, val_label) in enumerate(val_bar):
            model.eval()

            val_input = val_input.to(device)
            val_label = val_label.to(device)
            val_label = val_label.squeeze()

            with torch.no_grad():
                val_output = model(val_input)
            _, val_predict = torch.max(val_output.data, 1)
            val_correct += val_predict.eq(val_label.data).cpu().sum()
            val_total = val_total + val_label.size(0)
            val_ac = 100.0 * val_correct / val_total

            desc = 'Validation  : Epoch %d, AC = %.4f' % (epoch, val_ac)
            val_bar.set_description(desc)
            val_bar.update()
        
        savename = '/home/fzw/face-forgery-detection-val/checkpoint/checkpoint' + '_' + str(epoch) + '.tar'
        torch.save({'epoch': epoch, 'state_dict': model.state_dict()}, savename)
        epoch = epoch + 1

        
