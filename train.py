import sys
print("[DEPRECATED] This script is deprecated. Use train_pl.py instead.")
sys.exit()

'''
File Created: Sunday, 4th June 2023 10:53:35 pm
Author: Rutuja Gurav (rgura001@ucr.edu)
'''

'''
Example Run Command - 
cd ML4GWsearch
python -B src/train.py --model_name ResNet --use_gpu 2 --wandb_logging disabled &> stdout/train_$(date "+%Y%m%d%H%M%S").out &
'''
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42, help="random seed")
parser.add_argument("--use_gpu", type=int, default=0, help="device id of gpu to use")
parser.add_argument("--model_name", type=str, default="ResNet", help="model name")
parser.add_argument("--wandb_logging", type=str, default="disabled", help="wandb logging: disabled, online, dryrun")
parser.add_argument("--DATA_DIR", type=str, default="/data/rgura001/ML4GWsearch/g2net-gravitational-wave-detection", help="data directory")

args = parser.parse_args()

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = f"{args.use_gpu}"

import random, datetime, copy, json, glob
from pprint import pprint
import numpy as np
import pandas as pd
from PIL import Image as PILImage

import torch
print(f"Using torch version: {torch.__version__}")
print(f"Found {torch.cuda.device_count()} devices.")
import torch.nn.functional as F
from torchmetrics.classification import Accuracy, F1Score, AUROC
from torchinfo import summary
import  wandb

from dataloaders.dataloader import get_dataloaders
from models.get_model import get_model ## Add more models here

from clfutils4r.eval_classification import eval_classification as eval_model
from utils import plot_training_metrics
from early_stopping import ValLossEarlyStopping, TrainValLossDiffEarlyStopping

def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

def build_optimizer(model=None, config={},
                    optimizer='adam', learning_rate=None, lr_scheduler=None):
    if config['optimizer'] == "sgd":
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=config['learning_rate'],
                                    momentum=config['momentum'],
                                    nesterov=config['nesterov'],
                                    weight_decay=config['weight_decay'])
    elif config['optimizer'] == "adam":
        optimizer = torch.optim.Adam(model.parameters(), 
                                    lr=config['learning_rate'],
                                    weight_decay=config['weight_decay'])

    if config['lr_scheduler'] == "step":
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                        step_size=config['lr_scheduler__step_size'], 
                                                        gamma=config['lr_scheduler__gamma'])
        return optimizer, lr_scheduler
    else:
        return optimizer, None

def train_epoch(model=None, loader=None, lossfn=None, 
                optimizer=None, epoch=None, steps=None):

    batch_losses = []
    batch_acc = []
    batch_f1 = []
    batch_auroc = []
    for batch_idx, (x, y, id_) in enumerate(loader):
        optimizer.zero_grad()

        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = lossfn(logits, y)

        batch_losses.append(loss.cpu().detach().numpy())
        
        # preds = np.argmax(logits.cpu().detach().numpy(), axis=1)
        preds = torch.argmax(logits, dim=1)
        acc = Accuracy(task="binary").to(device)(preds, y).cpu().detach().numpy()
        f1 = F1Score(task="binary").to(device)(preds, y).cpu().detach().numpy()
        auroc = AUROC(task="binary").to(device)(preds, y).cpu().detach().numpy()
        mean_acc, mean_f1, mean_auroc = np.mean(acc), np.mean(f1), np.mean(auroc)
        
        batch_acc.append(mean_acc)
        batch_f1.append(mean_f1)
        batch_auroc.append(mean_auroc)

        if os.environ.get("WANDB_MODE") == "disabled":
            print("\t[TRAIN] Batch {}: loss={}, f1={} auroc={}".format(
                        epoch*steps+batch_idx, 
                        loss.item(), mean_f1, mean_auroc
                    ))
        # wandb.log({ 
        #             "batch": epoch*steps+batch_idx,
        #             "batch/train_loss": loss.item(), 
        #             "batch/train_acc": mean_acc, 
        #             "batch/train_f1": mean_f1, 
        #             "batch/train_auroc": mean_auroc
        #            },
        #         #    step = epoch*steps+batch_idx
        #         )
        
        loss.backward()
        optimizer.step()
    
    return np.mean(batch_losses), np.mean(batch_acc), np.mean(batch_f1), np.mean(batch_auroc)

def val_epoch(model=None, loader=None, lossfn=None, epoch=None, steps=None):

    batch_losses = []
    batch_acc = []
    batch_f1 = []
    batch_auroc = []
    for batch_idx, (x, y, id_) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = lossfn(logits, y)
        batch_losses.append(loss.cpu().detach().numpy())

        # preds = np.argmax(logits.cpu().detach().numpy(), axis=1)
        preds = torch.argmax(logits, dim=1)
        acc = Accuracy(task="binary").to(device)(preds, y).cpu().detach().numpy()
        f1 = F1Score(task="binary").to(device)(preds, y).cpu().detach().numpy()
        auroc = AUROC(task="binary").to(device)(preds, y).cpu().detach().numpy()
        mean_acc, mean_f1, mean_auroc = np.mean(acc), np.mean(f1), np.mean(auroc)

        batch_acc.append(mean_acc)
        batch_f1.append(mean_f1)
        batch_auroc.append(mean_auroc)
        
        if os.environ.get("WANDB_MODE") == "disabled":
            print("\t[VAL] Batch {}: loss={}, f1={} auroc={}".format(
                        epoch*steps+batch_idx, loss.item(), mean_f1, mean_auroc
                    ))
        # wandb.log({ 
        #             "batch": epoch*steps+batch_idx,
        #             "batch/val_loss": loss.item(),
        #             "batch/val_acc": mean_acc,  
        #             "batch/val_f1": mean_f1, 
        #             "batch/val_auroc": mean_auroc
        #            },
        #         #    step = epoch*steps+batch_idx
        #         )
    
    return np.mean(batch_losses), np.mean(batch_acc), np.mean(batch_f1), np.mean(batch_auroc)

def predict(model=None, loader=None, config={}):
    model.eval()
    with torch.no_grad():
        test_ids = []
        test_labels = []
        test_preds = []
        test_preds_proba = []
        for batch_idx, (x, y, id_) in enumerate(loader):
            test_ids.extend(id_)
            test_labels.extend(y)
            x, y = x.to(device), y.to(device)
            logits = model(x)
            logits = logits.cpu().detach().numpy()
            # logits = np.exp(logits)
            preds = np.argmax(logits, axis=1)
            test_preds.extend(preds)
            test_preds_proba.append(logits)

        test_labels = np.array(test_labels)
        test_preds = np.array(test_preds)
        test_preds_proba = np.concatenate(test_preds_proba, axis=0)
        
        test_df = pd.DataFrame() 
        test_df['id'] = test_ids
        test_df['label'] = test_labels
        test_df['prediction'] = test_preds
        test_df['prediction_proba_0'] = test_preds_proba[:,0]
        test_df['prediction_proba_1'] = test_preds_proba[:,1]

        test_df.to_csv(RUN_DIR+"/plots/test_metrics/testset_preds.csv", index=False)

        eval_model(class_names=['no_signal','signal'],
        y_test=test_labels, y_pred=test_preds, y_pred_proba=test_preds_proba,
        # titlestr=f"{config['ifos']}\n(z-norm={config['z_norm']}, highpass(20 Hz)={config['highpass']})\nModel: {config['model_name']}",
        titlestr=f"{config['ifos']}, {config['model_name']}",
        save=True, RESULTS_DIR=RUN_DIR+"/plots/test_metrics",
        show=False
        )

def train(config=None):

    '''
    Setting up dataloaders
    '''
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
                                    # rng_seed=42
                                )
    print(f"No. of training samples: {len(train_df)}")
    print(f"No. of validation samples: {len(val_df)}")
    print(f"No. of test samples: {len(test_df)}")
    
    num_train_steps = (len(train_df) // config['batch_size'])+1
    num_val_steps = (len(val_df) // config['batch_size'])+1

    '''
    Setting up model
    '''
    # seq_length = int(4096 / wandb.config.input_channels) # datapoint duration is 2 s sampled at 2048 Hz.
    
    model, lossfn = get_model(config=config)
    model = model.to(device)

    # model_stats = summary(model, (1,1,seq_length), verbose=0)
    # summary_str = str(model_stats)
    # with open(RUN_DIR+"/model_summary.txt", 'w') as f:
    #     f.write(summary_str)
    # print(summary_str)
    
    '''
    Setting up optimizer
    '''
    optimizer, lr_scheduler = build_optimizer(model, 
                                              config=config)

    '''
    Training loop
    '''
    # early_stopping = TrainValLossDiffEarlyStopping(tolerance=5, min_delta=0.5)
    early_stopping = ValLossEarlyStopping(patience=10, min_delta=1e-3)
    
    ## Log gradients
    # wandb.watch(model, lossfn, log="all", log_freq=1)
    
    epoch_train_losses = []
    epoch_val_losses = []
    epoch_train_accs = []
    epoch_val_accs = []
    epoch_train_f1s = []
    epoch_val_f1s = []
    epoch_train_aurocs = []
    epoch_val_aurocs = []
    epoch_lrs = []
    for epoch in range(config['epochs']):
        epoch_train_loss, \
            epoch_train_acc, \
                epoch_train_f1, \
                    epoch_train_auroc = train_epoch(model=model, 
                                                    loader=train_dataloader, 
                                                    optimizer=optimizer,
                                                    lossfn=lossfn,
                                                    epoch=epoch, 
                                                    steps = num_train_steps
                                                )
        if os.environ.get("WANDB_MODE") == "disabled":
            print("[TRAIN] Epoch {}: loss={}, f1={} auroc={}".format(epoch, epoch_train_loss, epoch_train_f1, epoch_train_auroc))
        epoch_train_losses.append(epoch_train_loss)
        epoch_train_accs.append(epoch_train_acc)
        epoch_train_f1s.append(epoch_train_f1)
        epoch_train_aurocs.append(epoch_train_auroc)        
        
        with torch.no_grad():
            epoch_val_loss, \
                epoch_val_acc, \
                    epoch_val_f1, \
                        epoch_val_auroc = val_epoch(model=model, 
                                                    loader=val_dataloader,
                                                    lossfn=lossfn,
                                                    epoch=epoch,
                                                    steps = num_val_steps
                                                )
        if config['stop_early']:
            ## Early Stopping
            early_stopping(epoch_val_loss)
            if early_stopping.early_stop:
                print("We are at epoch:", epoch)
                break

        if lr_scheduler is not None:
            lr_scheduler.step()
            # last_lr = lr_scheduler.get_last_lr()[0]
        last_lr = optimizer.param_groups[0]["lr"]
        epoch_lrs.append(last_lr)
        
        if os.environ.get("WANDB_MODE") == "disabled":
            print("[VAL] Epoch {}: loss={}, f1={} auroc={}".format(epoch, epoch_val_loss, epoch_val_f1, epoch_val_auroc))
        epoch_val_losses.append(epoch_val_loss)
        epoch_val_accs.append(epoch_val_acc)
        epoch_val_f1s.append(epoch_val_f1)
        epoch_val_aurocs.append(epoch_val_auroc)

        if os.environ.get("WANDB_MODE") != "disabled":
            wandb.log({ 
                        "epoch": epoch,
                        "epoch/train_loss": epoch_train_loss, 
                        "epoch/train_accuracy": epoch_train_acc,
                        "epoch/train_f1": epoch_train_f1,
                        "epoch/train_auroc": epoch_train_auroc,
                        "epoch/val_loss": epoch_val_loss, 
                        "epoch/val_accuracy": epoch_val_acc,
                        "epoch/val_f1": epoch_val_f1,
                        "epoch/val_auroc": epoch_val_auroc,
                        "epoch/lr": last_lr
                        },
                    #    step = epoch
                    )
    
    training_metrics = {
        "train_loss": epoch_train_losses,
        "train_acc": epoch_train_accs,
        "train_f1": epoch_train_f1s,
        "train_auroc": epoch_train_aurocs,
        "val_loss": epoch_val_losses,
        "val_acc": epoch_val_accs,
        "val_f1": epoch_val_f1s,
        "val_auroc": epoch_val_aurocs,
        "train_lr": epoch_lrs
    }

    print("Saving training metrics...")
    training_metrics_df = pd.DataFrame()
    for metric in [metric.split('_',1)[-1] for metric in training_metrics.keys()]:
        training_metrics_df['train_'+metric] = np.asarray(training_metrics['train_'+metric])
        if metric != 'lr':
            training_metrics_df['val_'+metric] = np.asarray(training_metrics['val_'+metric])
    training_metrics_df.to_csv(RUN_DIR+"/plots/train_metrics/training_metrics.csv", index=False)

    print("Plotting training metrics...")
    plot_training_metrics(history=training_metrics, RUN_DIR=RUN_DIR+"/plots/train_metrics", save=True)

    '''
    Evaluating model on test samples
    '''
    predict(model=model, loader=test_dataloader, config=config)

    if os.environ.get("WANDB_MODE") != "disabled":
        print("Logging test metrics plots to wandb...")
        filepaths = glob.glob(RUN_DIR+"/plots/test_metrics/*.png")
        # print(filepaths)
        test_metric_imgs = []
        for filepath in filepaths:
            caption =  filepath.split("/")[-1].split('.png')[0]
            # print(caption)
            pil_img = PILImage.open(filepath)
            pil_img = pil_img.resize((int(pil_img.size[0]//2), int(pil_img.size[1]//2)))
            wandb_img  = wandb.Image(pil_img, caption=caption)
            test_metric_imgs.append(wandb_img)
        wandb.log({"test_metrics": test_metric_imgs})

    '''
    Saving trained model
    '''
    # wandb.unwatch(model)
    # checkpoint = {'model': model,'model_state_dict': model.state_dict(),'optimizer': optimizer,'optimizer_state_dict': optimizer.state_dict()}
    # torch.save(checkpoint, RUN_DIR+"/saved_model.pth")

    '''
    Loading saved model
    '''
    # saved_model_dict = torch.load(RUN_DIR+"/saved_model.pth")
    # saved_model = saved_model_dict['model']
    # saved_model.load_state_dict(saved_model_dict['model_state_dict'])
    # optimizer = saved_model_dict['optimizer']
    # optimizer.load_state_dict(saved_model_dict['optimizer_state_dict'])

    # predict(model=saved_model.to(device), loader=test_dataloader)

if __name__ == "__main__":

    # args = parser.parse_args()
    seed = args.seed
    DATA_DIR = args.DATA_DIR
    model_name = args.model_name
    device = args.use_gpu
    wandb_logging = args.wandb_logging

    set_seed(seed)

    PROJECT_DIR = os.getcwd() # os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    print("PROJECT_DIR: {}".format(PROJECT_DIR))

    # Create a new run directory
    RESULTS_DIR = PROJECT_DIR+"/results"
    RUN_DIR = RESULTS_DIR+'/train_{}/{}'.format(datetime.datetime.now().strftime("%Y%m%d"), datetime.datetime.now().strftime("%H%M%S"))
    if not os.path.exists(RUN_DIR):
        os.makedirs(RUN_DIR+"/plots/train_metrics")
        os.makedirs(RUN_DIR+"/plots/test_metrics")
    
    with open(PROJECT_DIR+"/src/configs/train/base.json") as fp:
        base_config_dict = json.load(fp)

    # Read the chosen model's config
    if "FCN" in model_name:
        with open(PROJECT_DIR+"/src/configs/train/models_config/fcn.json") as fp:
            model_config = json.load(fp)
    elif "TCN" in model_name:
        with open(PROJECT_DIR+"/src/configs/train/models_config/tcn.json") as fp:
            model_config = json.load(fp)
    elif "ResNet" in model_name:
        with open(PROJECT_DIR+"/src/configs/train/models_config/resnet.json") as fp:
            model_config = json.load(fp)
    elif "GRUAttention" in model_name or "LSTMAttention" in model_name or "RNNAttention" in model_name:
        with open(PROJECT_DIR+"/src/configs/train/models_config/rnn_attn.json") as fp:
            model_config = json.load(fp)
    elif "G2NetWinner" in model_name:
        with open(PROJECT_DIR+"/src/configs/train/models_config/g2net_winner.json") as fp:
            model_config = json.load(fp)
    
    config_dict = {**base_config_dict, **model_config}

    # Read the optimizer, LR scheduler and early stopping config
    if config_dict['optimizer'] == "sgd":
        with open(PROJECT_DIR+"/src/configs/train/optim.json") as fp:
            optimizer_config_dict = json.load(fp)
        config_dict = {**config_dict, **optimizer_config_dict["sgd"]}

    if config_dict['lr_scheduler'] == "step":
        with open(PROJECT_DIR+"/src/configs/train/lr_schd.json") as fp:
            lr_scheduler_config_dict = json.load(fp)
        config_dict = {**config_dict, **lr_scheduler_config_dict["step"]}
    
    if config_dict['stop_early']:
        with open(PROJECT_DIR+"/src/configs/train/stop_early.json") as fp:
            early_stopping_config_dict = json.load(fp)
        config_dict = {**config_dict, **early_stopping_config_dict}

    config_dict['model_name'] = model_name
    config_dict['dataset_name'] = "g2net-gravitational-wave-detection"
    # config_dict['total_datapoints'] = 560000
    if config_dict['batch_size']*config_dict['num_batches'] < config_dict['total_datapoints']:
        config_dict['sample_size'] = config_dict['batch_size']*config_dict['num_batches']
    else:
        config_dict['sample_size'] = config_dict['total_datapoints']
        config_dict['num_batches'] = config_dict['total_datapoints']//config_dict['batch_size']
    
    pprint(config_dict)

    # Save config in RUN_DIR
    with open(RUN_DIR+"/config.json", 'w') as fp:
        json.dump(config_dict, fp, indent=4)

    # sys.exit()

    os.environ["WANDB_MODE"] = wandb_logging ## set this to "disabled" if you don't want to do any wandb logging. No further modifs to code needed!
    if os.environ["WANDB_MODE"] != "disabled":
        # Setup wandb logging
        with open(PROJECT_DIR+"/wandb_api_key.txt",'r') as f:
            wandb_api_key = f.read().rstrip()
        # print(wandb_api_key)
        os.environ['WANDB_API_KEY'] = wandb_api_key # my wandb project api key from https://wandb.ai/madlab-rutuja/ML4GWsearch

        # Initialize WandB logger
        wandb.init(project="ML4GWsearch", config=config_dict)
        train(config=wandb.config)
        wandb.finish()
    else:
        tic = datetime.datetime.now()
        train(config=config_dict)
        print("Elapsed time: {}".format(datetime.datetime.now()-tic))
    
