# Import Necessary packages
import streamlit as st

from fastai.vision.all import *
from fastai.vision import models

import torchvision
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

import warnings
warnings.filterwarnings('ignore')
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath


from fastbook import load_learner
st.set_option('deprecation.showPyplotGlobalUse', False)
map_location = torch.device('cpu') 
cnn_model = load_learner("./resnet34-leukemia.pkl")
unet_model = load_learner("./unet_leukemia.pkl")

clxs = {
    0:'Benign', 
    1:'Early-Precursor', 
    2:'Precursor', 
    3:'Progenitor', 
}

def detect(img, file_info, threshold=0.9):
    image = img
    image = np.array(image)
    result = model.predict(image)
    img = image.copy()
    plt.figure(figsize=(16, 16))
    st.metric(label="Class Label:", value=result[0])


def main():        
    st.title("Blood Leukemia Detection Model")
    uploaded_file = st.file_uploader('Upload Input Image', type=['jpg'])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Input Image", clamp=True, width=200)

    if st.button('Detect', key=1):
        if uploaded_file is not None:
            with st.spinner("Processing..."):
                st.session_state['input'] = image
                detect(image, uploaded_file.name[:-4])
        else:
            st.write('Upload an image first!')
    else:        
        st.info('Browse for the input image and click on Detect Button')

if __name__ == '__main__':
    st.set_page_config(page_title="Blood Leukemia Detection Model", layout="wide")
    main()
