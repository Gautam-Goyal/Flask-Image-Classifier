import numpy as np
import torch
from torch import nn
from torch import tensor
import json
import PIL
from PIL import Image
import argparse

import predict_load_functions

#Command Line Arguments

ap = argparse.ArgumentParser(
    description='predict-file')
ap.add_argument('--input_img', default='/Users/shivamgoyal/Desktop/ImageClassifier/flowers/test/102/image_08004.jpg', action="store", type = str)
ap.add_argument('--checkpoint',default='MASTER_CHECKPOINT.pth',action="store",type = str)
ap.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)
ap.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001)
ap.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
ap.add_argument('--gpu', default="gpu", action="store", dest="gpu")

pa = ap.parse_args()
path_image = pa.input_img
number_of_outputs = pa.top_k
power = pa.gpu
category_val= pa.category_names
learningr = pa.learning_rate
path = pa.checkpoint

print('Loading Checkpoint....')
model=predict_load_functions.load_checkpoint(path,learningr)

print('Predicting.......')
top_probs, top_labels, top_flowers = predict_load_functions.predict(category_val,path_image, model, number_of_outputs, power)
#
predict_load_functions.print_probability(top_flowers, top_probs)

print("Here you are")