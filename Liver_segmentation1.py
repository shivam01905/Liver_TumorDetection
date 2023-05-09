
import base64
from io import BytesIO

from flask import Flask, render_template, request
import torch
from monai.inferers import sliding_window_inference
from monai.networks.layers import Norm
from monai.networks.nets import UNet
from monai.transforms import (Activations, AddChanneld, Compose, CropForegroundd,
                              LoadImaged, Orientationd, Resized, ScaleIntensityRanged, Spacingd,
                              ToTensord)
from monai.utils import first
from PIL import Image
import numpy as np
import os
from glob import glob
from monai.data import Dataset, DataLoader

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # limit upload size to 16MB

in_dir = 'D:/Liver_Segmentation/Data_Train_Test'
model_dir = 'D:/Liver_Segmentation/result'

device = "cuda" if torch.cuda.is_available() else "cpu"
model = UNet(
    dimensions=3,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128, 256), 
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm=Norm.BATCH,
).to(device)

model.load_state_dict(torch.load(
    os.path.join(model_dir, "best_metric_model.pth"), map_location=device))
model.eval()

test_transforms = Compose(
    [
        LoadImaged(keys=["image"]),
        AddChanneld(keys=["image"]),
        Spacingd(keys=["image"], pixdim=(1.5, 1.5, 1.0), mode=("bilinear",)),
        Orientationd(keys=["image"], axcodes="RAS"),
        ScaleIntensityRanged(keys=["image"], a_min=-200, a_max=200,b_min=0.0, b_max=1.0, clip=True), 
        CropForegroundd(keys=["image"], source_key="image"),
        Resized(keys=["image"], spatial_size=[128, 128, 64]),   
        ToTensord(keys=["image"]),
    ]
)

def predict(image):
    image = np.asarray(image)
    test_ds = Dataset(data=[{"image": image}], transform=test_transforms)
    test_loader = DataLoader(test_ds)

    with torch.no_grad():
        predictions = sliding_window_inference(test_loader, model, sw_batch_size=1, device=device)

    pred = first(predictions)["pred"]
    pred = np.argmax(pred, axis=0)
    return pred

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("file")
        if not file:
            return render_template("index.html", label="No file")

        # Read image file and preprocess
        img = Image.open(file)
        img = img.convert('L')
        img_arr = np.array(img)
        img_arr = np.reshape(img_arr, (1,) + img_arr.shape)
        img_tensor = test_transforms(img_arr)["image"]
        img_tensor = img_tensor.to(device)

        # Perform inference
        with torch.no_grad():
            outputs = sliding_window_inference(img_tensor, (128, 128, 64), model, 4, True, 'mean')

        outputs = outputs.argmax(dim=1, keepdim=True)
        outputs = outputs.cpu().numpy().squeeze()

        # Convert the predicted mask to image
        predicted_img = Image.fromarray(np.uint8(outputs*255), mode="L")

        # Convert image to base64 string
        buffered = BytesIO()
        predicted_img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Render the template with predicted image
        return render_template("index.html", label=img_str)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)

