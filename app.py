from __future__ import division, print_function
# coding=utf-8

import os
import numpy as np
import cv2
# Keras

from tensorflow.keras.models import load_model
#from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array

# Flask utils
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app = Flask(__name__)
MODEL_PATH = 'Tomato_disease_detection.h5'



model = load_model(MODEL_PATH)
model.make_predict_function()

default_image_size = (256,256)
labels = ["Bacterial Spot" , "Early Blight" , "Late Blight" , "Leaf Mold" , "Healthy"]
def convert_image_to_array(image_dir):
    try:
        image = cv2.imread(image_dir)
        if image is not None :
            image = cv2.resize(image, default_image_size)   
            return img_to_array(image)
        else :
            return np.array([])
    except Exception as e:
        print(f"Error : {e}")
        return None

def model_predict(file_path , model):
    x = convert_image_to_array(file_path)
    x = np.expand_dims(x , axis = 0)
    preds = model.predict(x)
    return preds
    
    

@app.route("/" , methods = ['GET'])
def index():
    return render_template("index.html" , query = "")

@app.route("/" , methods = ['GET','POST'])
def upload():
    if(request.method == 'POST'):
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath , 'uploads' , secure_filename(f.filename))
        f.save(file_path)
        
        preds = model_predict(file_path , model)
        preds = np.argmax(preds)
        result = labels[preds]
        return render_template('index.html', prediction_text= result)
    return None

if __name__ == "__main__":
    app.run(debug = True)
        
        
    