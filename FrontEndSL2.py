import streamlit as st
from PIL import Image, ImageEnhance
import os
import torch
import numpy as np
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from preporcess import get_largest_connected_component

# Define the path to the images directory
IMAGE_DIR = 'D:/Liver_Segmentation/images'

# Define the path to the trained model
MODEL_PATH = 'D:/Liver_Segmentation/result/best_metric_model.pth'

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(
    dimensions=3,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm=Norm.BATCH,
).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))


# Define the display_image function
def display_image(image, size):
    try:
        st.image(image, caption='Selected Image', use_column_width=True, width=size)
    except IOError:
        st.error('Error: Unable to open image file.')


# Define the enhance_image function
def enhance_image(image, brightness=1.0, contrast=1.0, sharpness=1.0):
    enhancer = ImageEnhance.Brightness(image).enhance(brightness)
    enhancer = ImageEnhance.Contrast(enhancer).enhance(contrast)
    enhancer = ImageEnhance.Sharpness(enhancer).enhance(sharpness)
    return enhancer


# Define the predict function
def predict(image):
    # Convert the image to a PyTorch tensor
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=0)
    image = torch.from_numpy(image).float().to(device)

    # Make a prediction using the trained model
    model.eval()
    with torch.no_grad():
        pred = model(image)

    # Convert the prediction to a binary mask
    pred = torch.softmax(pred, dim=1)
    pred = pred[:, 1, ...].detach().cpu().numpy()
    pred = np.round(pred)

    # Get the largest connected component in the mask
    pred = get_largest_connected_component(pred)

    # Convert the mask back to an image
    pred = pred.squeeze(axis=0)
    pred = Image.fromarray((pred * 255).astype(np.uint8))

    return pred


# Add some styling to the sidebar
st.sidebar.title('Select an Image')
st.sidebar.markdown('---')

# Add sliders for image enhancements
brightness = st.sidebar.slider('Brightness', 0.0, 2.0, 1.0, 0.1)
contrast = st.sidebar.slider('Contrast', 0.0, 2.0, 1.0, 0.1)
sharpness = st.sidebar.slider('Sharpness', 0.0, 2.0, 1.0, 0.1)

# Add a button to reset the image enhancements
if st.sidebar.button('Reset Enhancements'):
    brightness = 1.0
    contrast = 1.0
    sharpness = 1.0

# Add a checkbox to toggle between the original and the enhanced image
show_original = st.sidebar.checkbox('Show original image')

# Display the image selection dropdown menu
selected_image = st.sidebar.selectbox('Select an image:', image_files)


def process_image(image_file):
    # Open the image file
    image = Image.open(os.path.join(IMAGE_DIR, image_file))

    # Enhance the image
    image = enhance_image(image, brightness, contrast, sharpness)

    # Display the original or enhanced image
    if show_original:
        display_image(image, 400)
    else:
        display_image(image, 400)

    # Generate the segmentation mask
    mask = predict(image)

    # Display the segmentation mask
    display_image(mask, 400)


# Define the image selection dropdown menu
image_files = os.listdir(IMAGE_DIR)
selected_image = st.sidebar.selectbox('Select an image:', image_files)

# Call the process_image function when an image is selected
if selected_image:
    process_image(selected_image)
