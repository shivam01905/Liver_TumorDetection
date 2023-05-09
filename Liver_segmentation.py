from flask import Flask, render_template
from PIL import Image
import base64
import io
from monai.utils import first, set_determinism
from monai.transforms import EnsureChannelFirst, Spacingd
from monai.transforms import(
    Compose,
    AddChanneld,
    LoadImaged,
    Resized,
    ToTensord,
    Spacingd,
    Orientationd,
    ScaleIntensityRanged,
    CropForegroundd,
    Activations,
)

from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.data import CacheDataset, DataLoader, Dataset
import torch
import os
from glob import glob
import numpy as np

from monai.inferers import sliding_window_inference

app = Flask(__name__)

in_dir = 'D:/Liver_Segmentation/Data_Train_Test'
model_dir = 'D:/Liver_Segmentation/result'

complete_transforms = Compose(
    [
        AddChanneld(keys=["vol"]),
        EnsureChannelFirst(),
        LoadImaged(keys=["vol"]),
        Spacingd(keys=["vol"], pixdim=(1.5, 1.5, 1.0), mode="bilinear"),
        Orientationd(keys=["vol"], axcodes="RAS"),
        ScaleIntensityRanged(keys=["vol"], a_min=-200, a_max=200, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["vol"], source_key="vol"),
        Resized(keys=["vol"], spatial_size=[128, 128, 64]),
        ToTensord(keys=["vol"]),
    ]
)


@app.route('/')
def home():
    train_loss = np.load(os.path.join(model_dir, 'loss_train.npy'))
    train_metric = np.load(os.path.join(model_dir, 'metric_train.npy'))
    test_loss = np.load(os.path.join(model_dir, 'loss_test.npy'))
    test_metric = np.load(os.path.join(model_dir, 'metric_test.npy'))
    path_train_volumes = sorted(glob(os.path.join(in_dir, "TrainVolumes", "*.nii.gz")))
    path_train_segmentation = sorted(glob(os.path.join(in_dir, "TrainSegmentation", "*.nii.gz")))
    path_test_volumes = sorted(glob(os.path.join(in_dir, "TestVolumes", "*.nii.gz")))
    path_test_segmentation = sorted(glob(os.path.join(in_dir, "TestSegmentation", "*.nii.gz")))
    train_files = [{"vol": image_name, "seg": label_name} for image_name, label_name in zip(path_train_volumes, path_train_segmentation)]
    test_files = [{"vol": image_name, "seg": label_name} for image_name, label_name in zip(path_test_volumes, path_test_segmentation)]
    test_files = test_files[0:9]
    
   



   
  

# Load the model
device = torch.device("cuda:0")
model = UNet(
    dimensions=3,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128, 256), 
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm=Norm.BATCH).to(device)
model.load_state_dict(torch.load(os.path.join(model_dir, "best_metric_model.pth")))
model.eval()

# Define the prediction function
def predict(image_path):
    image = complete_transforms({"vol": image_path})["vol"]
    image = image.unsqueeze(0)  # add batch dimension
    with torch.no_grad():
        output = sliding_window_inference(image, (128, 128, 64), model)
    output = (output > 0.5).float()
    return output.squeeze().cpu().numpy()

# Define the route to display the results
@app.route('/predict/<path:image_name>')
def prediction(image_name):
    image_path = os.path.join(in_dir, "TestVolumes", image_name)
    seg = predict(image_path)
    # Convert the segmentation array to a base64-encoded PNG image
    buffer = io.BytesIO()
    seg = (seg.squeeze() * 255).astype(np.uint8)  # scale the values to 0-255
    seg_img = Image.fromarray(seg, mode='L')
    seg_img.save(buffer, format='PNG')
    seg_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return render_template("predict.html", image_name=image_name, seg=seg_str)

if __name__ == '__main__':
    app.run(debug=True)

    




