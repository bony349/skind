from keras.models import load_keras
from skimage import transform
import numpy as np
from keras.preprocessing.image import load_img
from flask import Flask, request, render_template , jsonify

app = Flask(__name__)

covid =  load_keras('Covid.h5')
def Load_Images(img):
  pred_img = np.array(img).astype('float32')/255
  pred_img = transform.resize(pred_img,(200,200,3))
  pred_img = np.expand_dims(pred_img,axis=0)
  return pred_img

@app.route('/')
def index():
    print('Working')
    


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
