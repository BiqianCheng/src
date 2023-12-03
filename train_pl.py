'''
File Created: Monday, 21st August 2023 8:24:57 pm
Author: Rutuja Gurav (rgura001@ucr.edu)
'''
'''
Example Run Commands - 
cd ML4GWsearch
python -B src/train_pl.py --model_name FCN --use_gpu 1 --wandb_logging disabled &> stdout/train_$(date "+%Y%m%d%H%M%S").out &
'''
import os, sys
PROJECT_DIR = os.getcwd()
sys.path.append(os.path.join(PROJECT_DIR, 'src'))
print(f"PROJECT_DIR: {PROJECT_DIR}")

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42, help="random seed")
parser.add_argument("--use_gpu", type=int, default=1, help="device id of gpu to use")
parser.add_argument("--model_name", type=str, default="TCN", help="model name")
parser.add_argument("--wandb_logging", type=str, default="disabled", help="wandb logging: disabled, online, dryrun")
parser.add_argument("--DATA_DIR", type=str, default="/data/rgura001/ML4GWsearch/g2net-gravitational-wave-detection", help="path to data directory")

args = parser.parse_args()

import os
os.environ['CUDA_VISIBLE_DEVICES'] = f"{args.use_gpu}"

import random, datetime, json, glob, itertools, copy 
import pdb
from pprint import pprint

import numpy as np
import pandas as pd
from PIL import Image as PILImage
import torch
torch.cuda.empty_cache()
print(f"Using torch version: {torch.__version__}")
print(f"Found {torch.cuda.device_count()} devices.")
import torch.nn.functional as F
from torchmetrics.classification import Accuracy, F1Score, AUROC
from torchinfo import summary
import wandb
import pytorch_lightning as pl
print(f"Using pytorch_lightning version: {pl.__version__}")
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
# from lightning.pytorch.callbacks.early_stopping import EarlyStopping
# from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# import torchvision.transforms as transforms

# import cv2
# import time

from dataloaders.dataloader import get_dataloaders
from models.get_model import get_model ## Add more models here

from clfutils4r.eval_classification import eval_classification as eval_model
from utils import plot_training_metrics
from early_stopping import ValLossEarlyStopping, TrainValLossDiffEarlyStopping

# ### Integrate the basic Grad-CAM utility
# class GradCAM:
#     def __init__(self, model):
#         self.model = model
#         self.gradients = None
#         self.model.eval()
#         self.hook_layers()

#     def hook_layers(self):
#         def hook_fn(module, input, output):
#             self.gradients = input[0]

#         # Hook the first layer in the model. Modify if needed
#         self.hook = self.model.layers[0].register_backward_hook(hook_fn)

#     def generate_cam(self, input_image, target_class):
#         model_output = self.model(input_image)
#         self.model.zero_grad()
#         pred_class = model_output.argmax(dim=1).item()
#         class_score = model_output[0, target_class]
#         class_score.backward()

#         gradients = self.gradients[0].cpu().data.numpy()
#         activations = self.model.layers[0].squeeze().cpu().data.numpy()
        
#         weights = np.mean(gradients, axis=(1, 2))
#         cam = np.zeros(activations.shape[1:], dtype=np.float32)

#         for i, w in enumerate(weights):
#             cam += w * activations[i, :, :]

#         cam = np.maximum(cam, 0)
#         cam = cv2.resize(cam, input_image.shape[2:])
#         cam = cam - np.min(cam)
#         cam = cam / np.max(cam)
#         return cam

class MetricsCallback(pl.Callback):
    """PyTorch Lightning metrics callback."""

    def __init__(self):
        super().__init__()
        # self.metrics = []
        self.train_loss = []
        self.train_acc = []
        self.train_f1 = []
        self.train_auroc = []
        self.train_lr = []

        self.val_loss = []
        self.val_acc = []
        self.val_f1 = []
        self.val_auroc = []

    def on_train_epoch_end(self, trainer, pl_module):
        # print(f"\n\tEpoch {trainer.current_epoch} train loss: {trainer.callback_metrics['train_loss']}, {type(trainer.callback_metrics['train_loss'])}")
        each_me = copy.deepcopy(trainer.callback_metrics)
        self.train_loss.append(each_me['train_loss'].cpu().detach().numpy())
        self.train_acc.append(each_me['train_acc'].cpu().detach().numpy())
        self.train_f1.append(each_me['train_f1'].cpu().detach().numpy())
        self.train_auroc.append(each_me['train_auroc'].cpu().detach().numpy())
        self.train_lr.append(each_me['train_lr'].cpu().detach().numpy())
    
    def on_validation_epoch_end(self, trainer, pl_module):
        # print(f"\n\tEpoch {trainer.current_epoch} val loss: {trainer.callback_metrics['val_loss']}, {type(trainer.callback_metrics['train_loss'])}")
        each_me = copy.deepcopy(trainer.callback_metrics)
        self.val_loss.append(each_me['val_loss'].cpu().detach().numpy())
        self.val_acc.append(each_me['val_acc'].cpu().detach().numpy())
        self.val_f1.append(each_me['val_f1'].cpu().detach().numpy())
        self.val_auroc.append(each_me['val_auroc'].cpu().detach().numpy())
        
class GWDetectionLightningModule(pl.LightningModule):
    ## This is the Pytorch Lightning module that will be used for training and validation
    def __init__(self, config):
        super().__init__()

        ## config is a dictionary containing all the hyperparameters for the model, optimizer, lr_scheduler, etc.
        self.config = config 

        ## Get the model and loss function from src/models/pytorch/get_model.py based on the model_name in config
        self.model, self.lossfn = get_model(config=self.config) ##
        
        # self.early_stopping = ValLossEarlyStopping(patience=10, min_delta=1e-3)
        
        ## Add metrics to be logged by Pytorch Lightning
        self.metrics = torch.nn.ModuleDict({"acc": Accuracy(task="binary"),
                                        "f1": F1Score(task="binary"),
                                        "auroc": AUROC(task="binary")
                                    })
        
        # # Initialize variables for time-recording
        # self.epoch_start_time = 0
        # self.batch_times = []
    
    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        if self.config['optimizer'] == "sgd":
            optimizer = torch.optim.SGD(self.model.parameters(),
                                        lr=self.config['learning_rate'],
                                        momentum=self.config['momentum'],
                                        nesterov=self.config['nesterov'],
                                        weight_decay=self.config['weight_decay'])
        elif self.config['optimizer'] == "adam":
            optimizer = torch.optim.Adam(self.model.parameters(), 
                                        lr=self.config['learning_rate'],
                                        weight_decay=self.config['weight_decay'])

        if self.config['lr_scheduler'] == "step":
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                           step_size=self.config['lr_scheduler__step_size'], 
                                                           gamma=self.config['lr_scheduler__gamma'])
            return [optimizer], [{"scheduler": lr_scheduler, "interval": "epoch"}]
        ## Add more lr_schedulers here as elif statements
        else: ## No lr_scheduler used!
            return optimizer 
    
    def training_step(self, batch, batch_idx):
        # # Start measuring time for this batch
        # batch_start_time = time.time()

        x, y, _ = batch
        logits = self(x)
        loss = self.lossfn(logits, y)
        preds = torch.argmax(logits, dim=1)
        
        # # Integrate Grad-CAM visualization into the training step
        # # if batch_idx % self.config['gradcam_interval'] == 0: # Add a gradcam_interval to your config
        # if batch_idx % 10 == 0:
        #     target_class = preds[0].item()
        #     gradcam = GradCAM(self.model)
        #     cam = gradcam.generate_cam(x[0].unsqueeze(0), target_class)
        #     # Save or visualize this CAM map as needed
        #     print(cam.shape)
        #     print(x.shape)
        #     print(no)
            
        acc = self.metrics['acc'](preds, y)
        f1 = self.metrics['f1'](preds, y)
        auroc = self.metrics['auroc'](preds, y)
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        
        self.log('train_loss', loss, sync_dist=True, batch_size=self.config['batch_size'], on_epoch=True, on_step=False)
        self.log('train_acc', acc, sync_dist=True, batch_size=self.config['batch_size'], on_epoch=True, on_step=False)
        self.log('train_f1', f1, sync_dist=True, batch_size=self.config['batch_size'], on_epoch=True, on_step=False)
        self.log('train_auroc', auroc, sync_dist=True, batch_size=self.config['batch_size'], on_epoch=True, on_step=False)
        self.log('train_lr', lr, sync_dist=True, on_epoch=True, on_step=False)
        
    #     # Stop measuring time for this batch and store it
    #     batch_end_time = time.time()
    #     self.batch_times.append(batch_end_time - batch_start_time)

    #     return {"loss": loss, "accuracy": acc, "f1": f1, "auroc": auroc, "lr": lr}

    # def on_training_epoch_end(self, outputs):
    #     avg_batch_time = sum(self.batch_times) / len(self.batch_times)
    #     avg_epoch_time = avg_batch_time * len(self.batch_times)
    #     self.log('avg_epoch_time', avg_epoch_time, on_epoch=True, on_step=False)

    #     # Reset batch_times for the next epoch
    #     self.batch_times = []
    
    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)
        loss = self.lossfn(logits, y)
        preds = torch.argmax(logits, dim=1)
        
        acc = self.metrics['acc'](preds, y)
        f1 = self.metrics['f1'](preds, y)
        auroc = self.metrics['auroc'](preds, y)
        
        self.log('val_loss', loss, sync_dist=True, batch_size=self.config['batch_size'], on_epoch=True, on_step=False)
        self.log('val_acc', acc, sync_dist=True, batch_size=self.config['batch_size'], on_epoch=True, on_step=False)
        self.log('val_f1', f1, sync_dist=True, batch_size=self.config['batch_size'], on_epoch=True, on_step=False)
        self.log('val_auroc', auroc, sync_dist=True, batch_size=self.config['batch_size'], on_epoch=True, on_step=False)

        return {"loss": loss, "accuracy": acc, "f1": f1, "auroc": auroc}
    
    def predict_step(self, batch, batch_idx):
        x, y, id_ = batch
        # x = x.to(self.device)
        logits = self(x)
        preds = torch.argmax(logits, dim=1)

        return {
            'ids': id_,
            'labels': y,
            'predictions': preds,
            'prediction_probs': logits
        }
    

def main(config=None, logger=None):
    # Create Lightning module for the model, trai/val steps, and optimizer
    model = GWDetectionLightningModule(config)

    # Save model summary
    model_stats = summary(model, (1,config['input_channels'],config['seq_len']), verbose=0)
    summary_str = str(model_stats)
    with open(RUN_DIR+"/model_summary.txt", 'w') as f:
        f.write(summary_str)
    
    # Set up data loaders
    train_dataloader, \
        val_dataloader, \
            test_dataloader,\
                [train_df, val_df, test_df] \
                    = get_dataloaders(DATA_DIR=DATA_DIR,
                                    batch_size=config['batch_size'], 
                                    sample_size=config['sample_size'],
                                    ifos=config['ifos'],
                                    z_norm=config['z_norm'],
                                    highpass=config['highpass'],
                                    whiten=config['whiten'],
                                    scale=config['scale'],
                                    bandpass=config['bandpass'],
                                    # rng_seed=42 ## Only change this parameter if you want to use a different train/val/test split
                                )
    print(f"No. of training samples: {len(train_df)}")
    print(f"No. of validation samples: {len(val_df)}")
    print(f"No. of test samples: {len(test_df)}")

    ## Set up callbacks for the Trainer
    callbacks = []
    if config['stop_early']:
        early_stop_callback = pl.callbacks.early_stopping.EarlyStopping(monitor=config['stop_early__monitor'], 
                                                                        mode=config['stop_early__mode'],
                                                                        patience=config['stop_early__patience'], 
                                                                        verbose=True
                                                                    )
        callbacks.append(early_stop_callback)
    
    metrics_callback = MetricsCallback()
    callbacks.append(metrics_callback)

    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)

    ## BE CAREFUL: This takes too much disk space!
    checkpoint_callback = pl.callbacks.ModelCheckpoint(RUN_DIR+"/checkpoints", monitor='val_loss', mode='min', save_top_k=1)
    callbacks.append(checkpoint_callback)

    # Set up Trainer
    trainer = pl.Trainer(
        max_epochs=config['epochs'],
        # log_every_n_steps=1,
        accelerator='gpu',
        ## devices=[config.use_gpu], ## just set using os.environ['CUDA_VISIBLE_DEVICES'] instead
        accumulate_grad_batches=config['accumulate_grad_batches'],
        callbacks=callbacks,
        logger=logger
    )
    
    # Train the model
    tic = datetime.datetime.now()
    trainer.fit(model, train_dataloader, val_dataloader)
    elapsed_time = datetime.datetime.now()-tic
    print(f"Elapsed time: {elapsed_time}")


    num_epochs_trained = len(metrics_callback.train_loss)
    train_avgTimePerEpoch = elapsed_time / num_epochs_trained

    # Save training metrics
    training_metrics = {
        "train_loss": metrics_callback.train_loss,
        "train_acc": metrics_callback.train_acc,
        "train_f1": metrics_callback.train_f1,
        "train_auroc": metrics_callback.train_auroc,
        "train_lr": metrics_callback.train_lr,
        # "train_avgTimePerEpoch": train_avgTimePerEpoch,
        "val_loss": metrics_callback.val_loss,
        "val_acc": metrics_callback.val_acc,
        "val_f1": metrics_callback.val_f1,
        "val_auroc": metrics_callback.val_auroc,
    }

    # print(training_metrics)
    print("Saving training metrics...")
    training_metrics_df = pd.DataFrame()
    for metric in [metric.split('_',1)[-1] for metric in training_metrics.keys()]:
        training_metrics_df['train_'+metric] = np.asarray(training_metrics['train_'+metric])
        if metric != "lr":
            training_metrics_df['val_'+metric] = np.asarray(training_metrics['val_'+metric])[:-1] ## pl.Trainer logs an extra value for val_loss for some reason.
    training_metrics_df.to_csv(RUN_DIR+"/plots/train_metrics/training_metrics.csv", index=False)

    print("Plotting training metrics...")
    plot_training_metrics(history=training_metrics, RUN_DIR=RUN_DIR+"/plots/train_metrics", save=True)

    # Evaluate the model on the test set
    print("Evaluating the model on test samples...")
    test_preds = trainer.predict(model, test_dataloader)
    outputs = list(itertools.chain(test_preds))
    
    test_ids = []
    test_labels = []
    test_preds = []
    test_preds_proba = []
    for output in outputs:
        test_ids.extend(output['ids'])
        test_labels.extend(output['labels'])
        test_preds.extend(output['predictions'])
        test_preds_proba.append(output['prediction_probs'])

    test_labels = np.array(test_labels)
    test_preds = np.array(test_preds)
    test_preds_proba = np.concatenate(test_preds_proba, axis=0)

    test_df = pd.DataFrame()
    test_df['id'] = test_ids
    test_df['label'] = test_labels
    test_df['prediction'] = test_preds
    test_df['prediction_proba_0'] = test_preds_proba[:, 0]
    test_df['prediction_proba_1'] = test_preds_proba[:, 1]

    test_df.to_csv(RUN_DIR + "/plots/test_metrics/testset_preds.csv", index=False)

    # Plot test metrics
    eval_model(class_names=['no_signal', 'signal'],
                y_test=test_labels, y_pred=test_preds, y_pred_proba=test_preds_proba,
                # titlestr=f"{config['ifos']}\n(z-norm={config['z_norm']}, highpass(20 Hz)={config['highpass']})\nModel: {config['model_name']}",
                titlestr=f"{config['ifos']}, {config['model_name']}",
                save=True, RESULTS_DIR=RUN_DIR + "/plots/test_metrics",
                show=False
            )
    
    # (Optional) Log test metrics plots to wandb
    if os.environ.get("WANDB_MODE") != "disabled":
        print("Logging test metrics plots to wandb...")
        filepaths = glob.glob(RUN_DIR+"/plots/test_metrics/*.png")
        # print(filepaths)
        test_metric_imgs = []
        for filepath in filepaths:
            caption =  filepath.split("/")[-1].split('.png')[0]
            pil_img = PILImage.open(filepath)
            pil_img = pil_img.resize((int(pil_img.size[0]//2), int(pil_img.size[1]//2)))
            wandb_img  = wandb.Image(pil_img, caption=caption)
            test_metric_imgs.append(wandb_img)
        wandb.log({"test_metrics": test_metric_imgs})
    
   
if __name__ == "__main__":

    # args = parser.parse_args()
    seed = args.seed
    DATA_DIR = args.DATA_DIR
    model_name = args.model_name
    device = args.use_gpu
    wandb_logging = args.wandb_logging

    pl.seed_everything(seed)

    PROJECT_DIR = os.getcwd() # os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    print("PROJECT_DIR: {}".format(PROJECT_DIR))
    
    # Create a new run directory
    RESULTS_DIR = PROJECT_DIR+"/results"
    RUN_DIR = RESULTS_DIR+'/train_{}/{}'.format(datetime.datetime.now().strftime("%Y%m%d"), datetime.datetime.now().strftime("%H%M%S"))
    if not os.path.exists(RUN_DIR):
        os.makedirs(RUN_DIR+"/run_configs")
        os.makedirs(RUN_DIR+"/checkpoints")
        os.makedirs(RUN_DIR+"/plots/train_metrics")
        os.makedirs(RUN_DIR+"/plots/test_metrics")
    
    with open(PROJECT_DIR+"/configs/train/base.json") as fp:
        base_config_dict = json.load(fp)

    # Read the chosen model's config
    if model_name == "FCNPlus":
        with open(PROJECT_DIR+"/configs/train/models_config/fcn_plus.json") as fp:
            model_config = json.load(fp)
    elif model_name == "TCN":
        with open(PROJECT_DIR+"/configs/train/models_config/tcn.json") as fp:
            model_config = json.load(fp)
    elif "ResNet" in model_name:
        with open(PROJECT_DIR+"/configs/train/models_config/resnet.json") as fp:
            model_config = json.load(fp)
    elif model_name=="GRUAttention" or model_name=="LSTMAttention" or model_name=="RNNAttention":
        with open(PROJECT_DIR+"/configs/train/models_config/rnn_attn.json") as fp:
            model_config = json.load(fp)
    elif model_name=="G2NetWinner":
        with open(PROJECT_DIR+"/configs/train/models_config/g2net_winner.json") as fp:
            model_config = json.load(fp)
    elif model_name == "MLSTM_FCN":
        with open(PROJECT_DIR+"/configs/train/models_config/mlstm_fcn.json") as fp:
            model_config = json.load(fp)
    elif model_name == "MLP":
        with open(PROJECT_DIR+"/configs/train/models_config/mlp.json") as fp:
            model_config = json.load(fp)
    # elif "MyCustomModel" in model_name:
        ## You must have implemnted your custom model in src/models/pytorch/ 
        ## Or you are choosing a model from tsai which I haven't included above
        ## As shown above you will load the JSON config file for said model that you created and stored in src/configs/train/models_config
        ## Add as many elif statements as needed for the models you want to train
    else:
        raise ValueError("Invalid model name!")
        sys.exit()
    
    config_dict = {**base_config_dict, **model_config}

    # Read the optimizer, LR scheduler and early stopping config
    if config_dict['optimizer'] == "sgd":
        with open(PROJECT_DIR+"/configs/train/optim.json") as fp:
            optimizer_config_dict = json.load(fp)
        config_dict = {**config_dict, **optimizer_config_dict["sgd"]}
    elif config_dict['optimizer'] == "adam":
        with open(PROJECT_DIR+"/src/configs/train/optim.json") as fp:
            optimizer_config_dict = json.load(fp)
        config_dict = {**config_dict, **optimizer_config_dict["adam"]}

    if config_dict['lr_scheduler'] == "step":
        with open(PROJECT_DIR+"/configs/train/lr_schd.json") as fp:
            lr_scheduler_config_dict = json.load(fp)
        config_dict = {**config_dict, **lr_scheduler_config_dict["step"]}
    
    if config_dict['stop_early']:
        with open(PROJECT_DIR+"/configs/train/stop_early.json") as fp:
            early_stopping_config_dict = json.load(fp)
        config_dict = {**config_dict, **early_stopping_config_dict}

    config_dict['use_gpu'] = device
    config_dict['model_name'] = model_name
    if config_dict['batch_size']*config_dict['num_batches'] < config_dict['total_datapoints']:
        config_dict['sample_size'] = config_dict['batch_size']*config_dict['num_batches']
    else:
        config_dict['sample_size'] = config_dict['total_datapoints']
        config_dict['num_batches'] = config_dict['total_datapoints']//config_dict['batch_size']
    
    pprint(config_dict)

    # pdb.set_trace()

    # Save config in RUN_DIR
    with open(RUN_DIR+"/run_configs/full_config.json", 'w') as fp:
        json.dump(config_dict, fp, indent=4)
    # Save configs in RUN_DIR/run_configs
    with open(RUN_DIR+"/run_configs/base.json", 'w') as fp:
        json.dump(base_config_dict, fp, indent=4)
    with open(RUN_DIR+"/run_configs/model.json", 'w') as fp:
        model_config['model_name'] = model_name
        json.dump(model_config, fp, indent=4)
    with open(RUN_DIR+"/run_configs/optim.json", 'w') as fp:
        json.dump(optimizer_config_dict, fp, indent=4)
    with open(RUN_DIR+"/run_configs/lr_schd.json", 'w') as fp:
        json.dump(lr_scheduler_config_dict, fp, indent=4)
    with open(RUN_DIR+"/run_configs/stop_early.json", 'w') as fp:
        json.dump(early_stopping_config_dict, fp, indent=4)

    # Set up logging
    os.environ["WANDB_MODE"] = args.wandb_logging ## set this to "disabled" if you don't want to do any wandb logging. No further modifs to code needed!
    if os.environ["WANDB_MODE"] != "disabled":
        # Setup wandb logging
        with open(PROJECT_DIR+"/wandb_api_key.txt",'r') as f:
            wandb_api_key = f.read().rstrip()
        # print(wandb_api_key)
        os.environ['WANDB_API_KEY'] = wandb_api_key # my wandb project api key from https://wandb.ai/madlab-rutuja/ML4GWsearch

        # Initialize WandB logger
        wandb.init(project="ML4GWsearch", config=config_dict) # group="one-vs-three ifos",
        logger = pl.loggers.WandbLogger()
        
        tic  = datetime.datetime.now()
        main(config=wandb.config, logger=logger) # config_dict
        print("Elapsed time: {}".format(datetime.datetime.now()-tic))
        
        wandb.finish()
    
    else:
        logger = pl.loggers.CSVLogger(RUN_DIR+"/logs", name="train_logs")
        
        tic  = datetime.datetime.now()
        main(config=config_dict, logger=logger) # <---- This is the main function doing all the hard work. Everything else is just setup!
        print("Elapsed time: {}".format(datetime.datetime.now()-tic))