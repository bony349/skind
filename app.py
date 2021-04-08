import PIL
import numpy
from keras.models import load_keras
from skimage import transform
import numpy as np
from keras.preprocessing.image import load_img
from helper import *
from flask import Flask, request, render_template , jsonify

app = Flask(__name__)
from PIL import Image

svm_model = load_model('SVM.sav')
covid =  load_keras('Covid19.h5')
def Load_Images(img):
  pred_img = np.array(img).astype('float32')/255
  pred_img = transform.resize(pred_img,(200,200,3))
  pred_img = np.expand_dims(pred_img,axis=0)
  return pred_img

@app.route('/')
def index():
    new_test = []  # new images

    pil_image = PIL.Image.open("download.jpg").convert('RGB')
    open_cv_image = numpy.array(pil_image)
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    resized_img = resize(open_cv_image, (128, 64))

    fd_img, hog_img = hog(resized_img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True,
                          multichannel=True)
    new_test.append(fd_img)
    ############################################
    svm_model = load_model('SVM.sav')
    prediction = svm_model.predict(new_test)
    
    if (prediction == 0):
        return jsonify({'prediction':'Vitiligo'})
    elif (prediction == 1):
        return jsonify({'prediction':'Psoriasis'})
    elif (prediction == 2):
        return jsonify({'prediction':'Melanoma'})
    

#API Route 
@app.route('/RequestImageWithMetadata', methods=['POST'])
def post():
    new_test = []

    imagefile = request.files.get('imagefile', '')
    pil_image = PIL.Image.open(imagefile).convert('RGB')
    open_cv_image = numpy.array(pil_image)
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    resized_img = resize(open_cv_image,(128,64))

    fd_img, hog_img = hog(resized_img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True,
                          multichannel=True)
    new_test.append(fd_img)


    prediction = svm_model.predict(new_test)
    if (prediction == 0):
        return jsonify({'prediction':'Vitiligo'})
    elif (prediction == 1):
        return jsonify({'prediction':'Psoriasis'})
    elif (prediction == 2):
        return jsonify({'prediction':'Melanoma'})

@app.route('/CovidRequest', methods=['POST'])
def post():
   imagefile = request.files.get('imagefile', '')
   Image_to_pred = load_img(imagefile)
   Image_to_pred = Load_Images(Image_to_pred)
   prediction = np.argmax(covid.predict(Image_to_pred))
   if (prediction == 0):
       return jsonify({'prediction':'Non Informative Data'})
   elif (prediction == 1):
       return jsonify({'prediction':'Negative'})
   elif (prediction == 2):
       return jsonify({'prediction':'Positive'})


if __name__ == '__main__':
    app.run()
