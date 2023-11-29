import torch
from PIL import Image
import torchvision.transforms as transforms
import utils
import os, json, wandb, argparse
from models.get_model import get_model
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from models.tsai.tsai.models.TCN import TCN

PROJECT_DIR = os.getcwd()
with open(PROJECT_DIR+"/configs/train/base.json") as fp:
    base_config_dict = json.load(fp)
with open(PROJECT_DIR+"/configs/train/models_config/tcn.json") as fp:
    model_config = json.load(fp)

config_dict = {**base_config_dict, **model_config}
config_dict['model_name'] = 'TCN'

model, lossn = get_model(config=config_dict)
#########################################
# load the pre-trained model parameters #
#########################################
model_path = '/data/bchen158/ML4GW/ML4GWsearch/src/results/train_20231025/113109/checkpoints/epoch=0-step=1094.ckpt'
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint)

# load in the false-negative sample image to evaluate
# _id = b147f48068
# utils.plot_sample_ts(_id=_id, target=1, highpass=highpass, bandpass=bandpass, scale=scale, z_norm=z_norm, ts_whiten=whiten)

target_layers = [model.linear]
# Construct the CAM object once, and then re-use it on many images:
cam = GradCAM(model=model, target_layers=target_layers)

# We have to specify the target we want to generate the Class Activation Maps for.
# If targets is None, the highest scoring category will be used for every image 
# in the batch. Here we use ClassifierOutputTarget, but you can define your own custom targets
# That are, for example, combinations of categories, or specific outputs in a non standard model.
targets = [ClassifierOutputTarget(281)]

#######################################
# load the false-negative sample data #
#######################################
# load in the false-negative sample image to evaluate
_id = 'b147f48068'
highpass = False
z_norm = False
whiten = False
bandpass=True
scale=True

ts_dict = utils.get_sample_as_tsdict(_id=_id, target=1)
img = utils.plot_sample_ts(_id=_id, target=0, highpass=highpass, z_norm=z_norm, ts_whiten=whiten)
print(model)
print(config_dict['input_channels'])
DATA_DIR = "/data/rgura001/ML4GWsearch/g2net-gravitational-wave-detection"
# Set up data loaders
from dataloaders.dataloader import get_dataloaders
train_dataloader, \
    val_dataloader, \
        test_dataloader,\
            [train_df, val_df, test_df] \
                = get_dataloaders(DATA_DIR=DATA_DIR,
                                batch_size=config_dict['batch_size']
                                # rng_seed=42 ## Only change this parameter if you want to use a different train/val/test split
                            )
for x, y, id_num in train_dataloader:
    print(x[11].shape)
    print(y[11])
    print(id_num[11])
    break

# # utils.plot_sample_ts(_id=_id, target=1, highpass=highpass, bandpass=bandpass, scale=scale, z_norm=z_norm, ts_whiten=whiten)

# # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
# grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

# # In this example grayscale_cam has only one image in the batch:
# grayscale_cam = grayscale_cam[0, :]
# visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
