import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as snst
import torch
import matplotlib.pyplot as plt

from PIL import Image
import torchvision.transforms as transforms
import utils
import os, json, wandb, argparse
from models.get_model import get_model
from models.tsai.tsai.models.TCN import TCN
from train_pl import GWDetectionLightningModule
import pytorch_lightning as pl
import itertools

PROJECT_DIR = os.getcwd()
with open(PROJECT_DIR+"/configs/train/base.json") as fp:
    base_config_dict = json.load(fp)
with open(PROJECT_DIR+"/configs/train/models_config/fcn_plus.json") as fp:
    model_config = json.load(fp)

config_dict = {**base_config_dict, **model_config}
config_dict['model_name'] = 'FCNPlus'
if config_dict['batch_size']*config_dict['num_batches'] < config_dict['total_datapoints']:
        config_dict['sample_size'] = config_dict['batch_size']*config_dict['num_batches']
else:
    config_dict['sample_size'] = config_dict['total_datapoints']
    config_dict['num_batches'] = config_dict['total_datapoints']//config_dict['batch_size']

#################################################################
# Restoring the model to the training state from the checkpoint #
#################################################################
RESULTS_DIR = "/data/bchen158/ML4GW/ML4GWsearch/src/results/train_20231023/222238"
checkpoint_path = RESULTS_DIR + "/checkpoints/epoch=29-step=32820.ckpt"

from GWDetectionLightningModule import GWDetectionLightningModule
model = GWDetectionLightningModule(config=config_dict).load_from_checkpoint(checkpoint_path, config=config_dict)
# ??? whether the model checkpoint is correctly loaded in
# model.load_from_checkpoint(checkpoint_path, config=config_dict)
model.eval()
from pprint import pprint
print("# the model configuration: ")
pprint(dict(model.hparams))

# load the Time-Series dataset
DATA_DIR = "/data/rgura001/ML4GWsearch/g2net-gravitational-wave-detection"
# Set up data loaders
from dataloaders.dataloader import get_dataloaders

train_dataloader, \
    val_dataloader, \
        test_dataloader,\
            [train_df, val_df, test_df] \
                = get_dataloaders(DATA_DIR=DATA_DIR,
                                batch_size=config_dict['batch_size'],
                                sample_size=config_dict['sample_size'],
                                ifos=config_dict['ifos'],
                                z_norm=config_dict['z_norm'],
                                highpass=config_dict['highpass'],
                                whiten=config_dict['whiten'],
                                scale=config_dict['scale'],
                                bandpass=config_dict['bandpass'],
                                # rng_seed=42 ## Only change this parameter if you want to use a different train/val/test split
                            )

# trainer = pl.Trainer(
#         max_epochs=config_dict['epochs'],
#         # log_every_n_steps=1,
#         accelerator='gpu',
#         ## devices=[config.use_gpu], ## just set using os.environ['CUDA_VISIBLE_DEVICES'] instead
#         accumulate_grad_batches=config_dict['accumulate_grad_batches']
#     )

# outputs = list(itertools.chain(test_preds))
# test_ids = []
# test_labels = []
# test_preds = []
# test_preds_proba = []
# idx = 0
# true_pred_idx = []
# for output in outputs:
#     test_ids.extend(output['ids'])
#     test_labels.extend(output['labels'])
#     test_preds.extend(output['predictions'])
#     test_preds_proba.append(output['prediction_probs'])
#     if test_labels[idx]==0 and test_labels[idx] == test_preds[idx]:
#         true_pred_idx.append(idx)
#     idx += 1
# print("# idx for TN:")
# print(true_pred_idx)

# Fucntion to map the channel to corresponding detector names
def map_channels_to_detectirs(indices):
    detectors = ["LIGO Hanford", "LIGO Livingston", "Virgo"]
    return [detectors[i] for i in indices]

# Function to shuffle time-series data for each sample
def shuffle_time_series_data(dataloader, shuffled_num_batches):
    shuffled_data = []
    iters = 0
    for x, y, id_num in dataloader:
        iters += 1
        shuffled_x = np.zeros_like(x)
        for i in range(x.shape[0]):
            indices = np.random.permutation(3)
            shuffled_x[i] = x[i, indices]
            mapped_detectors = map_channels_to_detectirs(indices)
            # print(f"Sample {i} -Shuffled order: {mapped_detectors}")
            if iters == 1 and mapped_detectors[2] != "Virgo":
                list_idx = iters - 1
                item_idx = i
                print("list idx = ", list_idx)
                print("item_idx = ", item_idx)
                print(f"Sample {i} -Shuffled order: {mapped_detectors}")
        id_num = np.asarray(id_num)
        # shuffled_data.append([shuffled_x, y.numpy(), id_num])
        shuffled_data.append([shuffled_x, y.numpy(), id_num])
        if iters>shuffled_num_batches:
            break
    return shuffled_data, list_idx, item_idx

# shuffle_train_data, list_idx, item_idx = shuffle_time_series_data(train_dataloader, 10)
# shuffle_test_data, list_idx, item_idx = shuffle_time_series_data(test_dataloader, 10)

# for info in shuffle_train_data:
#     train_x = info[0][item_idx]
#     train_y = info[1][item_idx]
#     train_id = info[2][item_idx]
#     break         

# for info in shuffle_test_data:
#     test_x = info[0][item_idx]
#     test_y = info[1][item_idx]
#     test_batch = torch.from_numpy(info[0])
#     test_id = info[2][item_idx]
#     break      

for x, y, id_num in train_dataloader:
    train_x = x[57]
    train_y = y[57]
    train_id = id_num[57]
    break
    # print(x.shape)
    # break

for x, y, id_num in test_dataloader:
    test_x = x[57]
    test_y = y[57]
    test_batch = x
    test_id = id_num
    break
    # print(x.shape)
    # break

# print("test_batch = ", test_batch.shape)
# print("Type of test batch = ", type(test_batch))
device = torch.device("cpu")
model.to(device)
logits = model(test_batch)
test_preds = torch.argmax(logits, dim=1)
print("test_batch: \n", test_batch.numpy().shape)
print("test prediction is : ")
print(test_preds)
print(test_id)

train_x = train_x.numpy()
train_y = train_y.numpy()
test_x = test_x.numpy()
test_y = test_y.numpy()

from TSInterpret.InterpretabilityModels.Saliency.TSR import TSR
int_mod = TSR(model, train_x.shape[-2], train_x.shape[-1], method='IG', mode='time')
item = np.array([test_x[:, :train_x.shape[-1]]])
label = test_y

exp = int_mod.explain(item, labels=label, TSR=True)

int_mod.plot(np.array([item[0].T]), exp.T, save="TSinter.png")

# from tslearn.datasets import UCR_UEA_datasets
# dataset = 'BasicMotions'
# train_x, train_y, test_x, test_y = UCR_UEA_datasets().load_dataset(dataset)

# import sklearn
# enc1 = sklearn.preprocessing.OneHotEncoder(sparse=False).fit(train_y.reshape(-1, 1))
# train_y = enc1.transform(train_y.reshape(-1, 1))
# test_y = enc1.transform(test_y.reshape(-1, 1))
# print(train_x.shape[-2])
# print(train_x.shape[-1])
# print(train_x.shape)
# print(train_x[0,:,:].shape)
# print(test_x[0,:,:].shape)
# print(test_y[0])
# print(int(np.argmax(test_y[0])))

