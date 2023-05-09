# Liver_TumorDetection

Tumor Detection using Liver Segmentation
This is a project for detecting tumors in medical images using liver segmentation. The project is split into five code files:

train.py: This file contains the code for training the deep learning model that performs liver segmentation.

preprocess.py: This file contains the code for preprocessing the medical images and their corresponding masks to prepare them for training the model.

model_evaluation.py: This file contains the code for defining utility functions that are used in the main .py file. These functions include loading the images, preprocessing the images, generating a binary mask for the liver, and resizing the images to a standard size

testing.py: This file contains the code for using the trained model to perform liver segmentation on a new medical image.

frontend.py: This file contains the code for a Streamlit app that allows users to select a medical image and visualize the results of liver segmentation. Users can also adjust the brightness, contrast, and sharpness of the image, save the enhanced image to their local machine, and open the enhanced image in a new window.

Usage
To use the project, follow these steps:

Install the required packages listed in requirements.txt.

Run train.py to train and save the deep learning model for liver segmentation.

Run preprocessing.py to preprocess the medical images and masks for training the model.

Run testing.py to use the trained model to perform liver segmentation on a new medical image.

Run frontend.py to launch the Streamlit app for visualizing the results of liver segmentation on a selected medical image.

Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.
