!pip install tensorflow
import numpy as np
import tensorflow as tf
from tensorflow import keras
import numpy as np
import tensorflow as tf
from tensorflow import keras
labels = {
    0:'buildings',
 1:'forest',
 2:'glacier',
 3:'mountain',
 4:'sea',
 5:'street'}
image_size = (299, 299, 3)
model = keras.models.load_model('xception_02_0.903.h5')
!wget https://www.constructionkenya.com/wp-content/uploads/2019/09/Leonardo-South-Africa.jpg -O img.jpg
from tensorflow.keras.preprocessing.image import load_img 
img = load_img('img.jpg', target_size=(image_size))
img

from tensorflow.keras.applications.xception import preprocess_input
x = np.array(img)
X = np.array([x])
X = preprocess_input(X)
pred = model.predict(X)
labels[pred[0].argmax()]
