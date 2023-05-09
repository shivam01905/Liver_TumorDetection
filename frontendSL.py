import streamlit as st
from PIL import Image, ImageEnhance
import os

# Define the path to the images directory
IMAGE_DIR = 'D:/Liver_Segmentation/images'

# Generate a list of image paths
image_paths = [os.path.join(IMAGE_DIR, img_file) for img_file in os.listdir(IMAGE_DIR)]

# Set the app title and description
st.set_page_config(page_title='Tumor Detection using Liver Segmentation', page_icon=':microscope:', layout='wide')
st.title('Tumor Detection using Liver Segmentation')
st.write('This app uses liver segmentation to detect tumors in medical images. Select an image from the sidebar to get started!')


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
selected_image_path = st.sidebar.selectbox('Select an image', image_paths)

# Display the selected image
if selected_image_path:
    if os.path.isfile(selected_image_path):
        image = Image.open(selected_image_path)
        if not show_original:
            image = enhance_image(image, brightness=brightness, contrast=contrast, sharpness=sharpness)
        size = st.sidebar.slider('Image Size', 100, 1000, 500, 50)
        display_image(image, size)
        # Add a button to save the enhanced image to the local machine
        if st.button('Save Image'):
            image.save('enhanced_image.jpg')
            st.success('Image saved successfully!')
        # Add a button to open a new window with the enhanced image
        if st.button('Open Image'):
            from PIL import ImageGrab
            # Save the enhanced image to a temporary file
            temp_file = 'temp_image.jpg'
            image.save(temp_file)
            # Open the temporary file in a new window
            ImageGrab.grab().show(temp_file)


# Add a footer with some attribution and contact info
st.markdown('---')
st.write('Created by Shivam Tiwari - [GitHub](https://github.com/shivam01905)')
st.write('For questions or comments, email me at shivam.tiwari01905@gmail.com')
