# Import Necessary packages
import streamlit as st

from fastai.vision.all import *
from fastai.vision import models
from skimage.morphology import *

from email.utils import parseaddr
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import smtplib

import warnings
warnings.filterwarnings('ignore')

# Uncomment if you are on Windows
# import pathlib
# temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPath

#===============================   EMAIL CONFIGURATION ========================================#
sender_email = "example@outlook.com"                             # Enter your outlook address
password = "password"                                            # Enter your outlook password
stmp_server = "smtp-mail.outlook.com"
stmp_port = 587
#==============================================================================================#

# Load Models
def get_msk(o): 
    return pathlib.Path(str(o).replace('dataset', 'unet_dataset'))
idx = {'Background': 0, 'Cancerous blood cells': 1}
void_code = idx['Background']

def acc(inp, targ):
    targ = targ.squeeze(1)
    mask = targ != void_code
    return (inp.argmax(dim=1)[mask]==targ[mask]).float().mean()

from fastbook import load_learner
st.set_option('deprecation.showPyplotGlobalUse', False)
cnn_model = load_learner("./resnet34-leukemia.pkl")

# Class Labels
clxs = {
    0:'Benign', 
    1:'Early-Precursor', 
    2:'Precursor', 
    3:'Progenitor', 
}

def detect(img, email):
    image = img
    image = np.array(image)
    orig = image.copy()

    # Predict Class label for the input image
    result = cnn_model.predict(image)
    class_label = clxs[int(result[1])]

    if class_label == "Benign":
        st.success('Healthy Blood Sample', icon="✅")
        st.metric("Class Label", f"{class_label}")
    else:
        st.error(f'Acute Lymphoblastic Leukemia found in Blood Sample. Sending email alert to <{email}>', icon="🚨")
        col1, col2 = st.columns(2)
        col1.metric("Class Label", f"Malignant")
        col2.metric("Classification", f"{class_label}")

        # Get the mask with cancerous cells
        HSV_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(HSV_image, (120, 50, 50), (145, 255, 255))
        hsv_mask = mask.copy()

        mask = mask > 0
        mask = np.array(mask)

        # Morphological Operation on mask
        footprint = square(4) 
        mask = erosion(mask, footprint)
        
        footprint = disk(4) 
        mask = closing(mask, footprint)

        mask = mask.astype(np.uint8)

        # Small object removal
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        min_area = 100  
        filtered_labels = []
        for label in range(1, num_labels):
            if stats[label, cv2.CC_STAT_AREA] >= min_area:
                filtered_labels.append(label)

        # Draw red circles around the cancerous cells
        for label in filtered_labels:
            # Create a binary image containing only the current component 
            component_image = (labels == label).astype('uint8')
            # Find the contours of the component 
            contours, _ = cv2.findContours(component_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            center, size = cv2.minEnclosingCircle(contours[0])
            size = int(size)
            cv2.circle(orig, (int(center[0]), int(center[1])), size+10, (255, 87, 51), thickness=4, lineType=8, shift=0)

        # Show Image with detected cancerous cells
        st.image([hsv_mask, orig], caption=["HSV Thresholding", "Blood Cancer Cell Detection"], width = 300)

        # Send Email alert
        # receiver_email = email
        # message = f"""\
        # Subject: Blood Sample Analysis

        # Your blood sample was detected with Acute Lymphoblastic Leukemia.        
        # Classification: {class_label}
        # """
        # try:
        #     server = smtplib.SMTP(stmp_server, stmp_port)
        #     server.ehlo()
        #     server.starttls()
        #     server.login(sender_email, password)
        #     server.sendmail(sender_email, receiver_email, message)
        #     st.success(f"Email sent successfully to <{email}>!", icon="✅")
        # except Exception as e:
        #     st.error(f"Error occurred while sending email: {e}", icon="❌")
        # finally:
        #     server.quit()

def main():        
    st.title("Blood Leukemia Detection Model")
    uploaded_file = st.file_uploader('Upload Input Image', type=['jpg'])
    email = st.text_input('Enter your email Address (optional)', 'example@email.com')

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Input Image", clamp=True, width=200)

    if st.button('Detect', key=1):
        if uploaded_file is not None:
            with st.spinner("Processing..."):
                st.session_state['input'] = image
                detect(image, email)
        else:
            st.write('Upload an image first!')
    else:        
        st.info('Browse for the input image and click on Detect Button')

if __name__ == '__main__':
    st.set_page_config(page_title="Blood Leukemia Detection Model", layout="wide")
    main()
