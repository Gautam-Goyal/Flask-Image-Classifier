from __future__ import division, print_function

import json
import os

import numpy as np
import torch
from PIL import Image
# Flask utils
from flask import Flask, request, render_template
# Keras
# from keras.applications.imagenet_utils import decode_predictions
from torch import optim
from torchvision import transforms, models
from werkzeug.utils import secure_filename, redirect

import predict

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = '/Users/shivamgoyal/PycharmProjects/FLASK-Python/MASTER_CHECKPOINT.pth'


def load_checkpoint(path):
    if torch.cuda.is_available():
        map_location = lambda storage, loc: storage.cuda()
    else:
        map_location = 'cpu'

    if path == None:
        path = '/Users/shivamgoyal/PycharmProjects/FLASK-Python/MASTER_CHECKPOINT.pth'

    learningr = 0.001
    checkpoint = torch.load(path, map_location=map_location)
    #     with open(path,'rb') as f:
    #         checkpoint = pickle.load(f,map_location=map_location)
    model = models.vgg16(pretrained=True)
    model.arch = checkpoint['arch']
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])

    optimizer = optim.Adam(model.classifier.parameters(), lr=learningr)
    optimizer.load_state_dict(checkpoint['optimizer_dict'])
    #     epoch=checkpoint['epochs']
    for param in model.parameters():
        param.requires_grad = False

    return model


def process_image(image):
    pic = Image.open(image)
    img_transform = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    pic2 = img_transform(pic)

    # np_image = np.array(pic2)

    return pic2


def predict(model,image_path):
    if torch.cuda.is_available():
        model.to('cuda:0')

    # No need for GPU on this part (just causes problems)
    with open('cat_to_name.json', 'r') as json_file:
        ind_to_name = json.load(json_file)
    model.to("cpu")

    # Set model to evaluate
    model.eval();

    # Convert image from numpy to torch
    torch_image = torch.from_numpy(np.expand_dims(process_image(image_path),
                                                  axis=0)).type(torch.FloatTensor).to("cpu")

    log_probs = model.forward(torch_image)

    # Convert to linear scale
    linear_probs = torch.exp(log_probs)

    # Find the top 5 results
    top_probs, top_labels = linear_probs.topk(2)

    # Detatch all of the details
    top_probs = np.array(top_probs.detach())[0]
    top_labels = np.array(top_labels.detach())[0]

    # Convert to classes
    idx_to_class = {val: key for key, val in
                    model.class_to_idx.items()}
    top_labels = [idx_to_class[lab] for lab in top_labels]
    top_flowers = [ind_to_name[lab] for lab in top_labels]

    return top_probs, top_labels, top_flowers


# Load your trained model
model = load_checkpoint(MODEL_PATH)
# model.predict()

print('Model loaded. Check http://127.0.0.1:5000/')


# def model_predict(img_path, model):
#     img = image.load_img(img_path, target_size=(224, 224))
#
#     # Preprocessing the image
#     x = image.img_to_array(img)
#     # x = np.true_divide(x, 255)
#     x = np.expand_dims(x, axis=0)
#
#     # Be careful how your trained model deals with the input
#     # otherwise, it won't make correct prediction!
#     x = preprocess_input(x, mode='caffe')
#
#     preds = model.predict(x)
#     return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, secure_filename(f.filename))
        f.save(file_path)
        prob, class_id, class_name = predict(model,file_path)

        # # Process your result for human
        # # pred_class = preds.argmax(axis=-1)            # Simple argmax
        # pred_class = decode_predictions(name[0], top=1)  # ImageNet Decode
        # result = str(pred_class[0][0][1])  # Convert to string
        for i, j in enumerate(zip(class_name, prob)):
            # print("Rank {}:".format(i + 1),
            #       "Flower: {}, liklihood: {}%".format(j[0], j[1] * 100))
            return j[0]
    return None


if __name__ == '__main__':
    app.run(debug=True)
