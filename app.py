import gradio as gr
import cv2 as cv
import tensorflow as tf
import PIL.Image as Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
model = tf.keras.models.load_model('main_model.h5')
file = pd.read_csv('ArSL_Data_Labels.csv')
class_names = file['Class'].to_list()
class_set = pd.unique(class_names).tolist()
def preprocess(img):
   img = img.convert("L").resize((64, 64))
   img = np.array(img) / 255.0
   img = np.expand_dims(img, axis=-1)
   img = np.expand_dims(img, axis=0)
   return img

def predict_sign(img):
    processed_img = preprocess(img)
    preds = model.predict(processed_img)
    predicted_index = np.argmax(preds)
    return class_set[predicted_index]
interface = gr.Interface(
    fn=predict_sign,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Sign Language Recognizer",
    description="Upload a hand sign image and the model will predict the corresponding character."
)
interface.launch(share = True)