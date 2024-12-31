import os
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import shutil

train_dataset = "./dataset/train"

def im_preprocess(type="v1"):
    for v_name in os.listdir(train_dataset):
        v_path = os.path.join(train_dataset, v_name)
        failed_path = os.path.join(v_path, "failed")
        if type == "v1":
            for i, im_name in enumerate(os.listdir(failed_path)):
                im_path = os.path.join(failed_path, im_name)
                # os.rename(im_path, f"{train_candidate}/{i}.png")
                gray = cv2.cvtColor(cv2.imread(im_path), cv2.COLOR_BGR2GRAY)
                height, width = gray.shape
                gray = cv2.resize(gray, (int(118/height*width), 118))
                height, width = gray.shape
                gray = np.pad(gray, ((0,0), (0, 1167-width)), 'median')
                blurred = cv2.GaussianBlur(gray, (5,5), 0)
                eroded = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY,9,-1)
                cv2.imwrite(f"{failed_path}/{os.path.splitext(os.path.basename(im_path))[0]}_pre.png", eroded)
        
        if type == "v2":
            for i, im_name in enumerate(os.listdir(failed_path)):
                im_path = os.path.join(failed_path, im_name)
                gray = cv2.cvtColor(cv2.imread(im_path), cv2.COLOR_BGR2GRAY)
                height, width = gray.shape
                rate = 0.01
                num_noise = int(gray.size*rate)
                coords = [np.random.randint(0, i-1, num_noise) for i in gray.shape]
                gray[coords[0], coords[1]] = 0

                gray = cv2.resize(gray,(int(118/height*width),118))
                height, width = gray.shape
                gray = np.pad(gray, ((0,0),(0, 1167-width)), 'median')

                blurred = cv2.GaussianBlur(gray, (5,5), 0)
                thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 4)
                kernel = np.ones((1,2), np.uint8)
                eroded = cv2.erode(thresh, kernel, iterations=2)
                cv2.imwrite(f"{failed_path}/{os.path.splitext(os.path.basename(im_path))[0]}_pre.png", eroded)


def dataset_prepocess():
    for v_name in os.listdir(train_dataset):
        v_path = os.path.join(train_dataset, v_name)
        failed_dir = os.path.join(v_path, "failed")
        passed_dir = os.path.join(v_path, "passed")
        json_path = os.path.join(v_path, "labels.json")
        os.makedirs(passed_dir, exist_ok=True)
        with open(json_path, "r") as file:
            data = json.load(file)
        
        for item in data:
            im_name = f"{item}.png"
            im_name_pre = f"{item}_pre.png"
            im_path = os.path.join(failed_dir, im_name)
            im_path_pre = os.path.join(failed_dir, im_name_pre)
            if os.path.exists(im_path):
                shutil.copy(im_path, os.path.join(passed_dir, f"{v_name}_{im_name}"))
                shutil.copy(im_path_pre, os.path.join(passed_dir, f"{v_name}_{im_name_pre}"))
                
def label_preprocess():
    char_list= set()
    for v_name in os.listdir(train_dataset):
        v_path = os.path.join(train_dataset, v_name)
        label_path = os.path.join(v_path, "labels.json")
        with open(label_path, "r") as file:
            train_labels = json.load(file)
                
            for label in train_labels.values():
                char_list.update(set(label))
    char_list=sorted(char_list)
            
    with open("./char_list.json", "w") as file:
        json.dump(char_list, file)
        
    print(char_list)
            
if __name__ == "__main__":
    label_preprocess()