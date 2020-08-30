import tensorflow as tf
import pandas as pd
import os
from PIL import Image
import warnings

warnings.filterwarnings('ignore')

embedding_dim = 256
units = 256
vocab_size = 4000
max_length = 32
attention_features_shape = 64

image_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
new_input = image_model.input
hidden_layer = image_model.layers[-1].output
image_features_extract_model = tf.keras.Model(new_input, hidden_layer)
tokenizer = pd.read_pickle('tokenizer.pkl')

def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (224, 224))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path