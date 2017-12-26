from django.http import HttpResponse
from django.shortcuts import render
from keras.models import model_from_json
from scipy.misc import imresize
from PIL import Image
import numpy as np

def index(request):
    return render(request, 'index.html', {'result': "I'll translate you."})

def predict(request):
    sonuc = 'Error!'
    if request.method == 'POST' and request.FILES:
        file_img = Image.open(request.FILES['image'])
        image = 1-np.array(file_img)
        image = imresize(image, (64,64, 1))
        image = image.reshape(1, 64, 64, 1)
        model_file = open('static/model.json', 'r')
        model = model_file.read()
        model_file.close()
        model = model_from_json(model)
        model.load_weights("static/weights.h5")
        Y = model.predict(image)
        result = np.argmax(Y, axis=1)
        result = "<span style='color:#fc4f3f; font-size: 2em;'>{0}</span>".format(result)
    else:
        sonuc = "<span style='color:#fc4f3f;'>Your file couldn't found!</span>"
    return render(request, 'index.html', {'result': result})


