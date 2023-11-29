from torchmetrics.classification import Accuracy, F1Score, AUROC
from models.get_model import get_model 
import pytorch_lightning as pl
import torch

class GWDetectionLightningModule(pl.LightningModule):
    ## This is the Pytorch Lightning module that will be used for training and validation
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

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
        x, y, _ = batch
        logits = self(x)
        loss = self.lossfn(logits, y)
        preds = torch.argmax(logits, dim=1)
        
        acc = self.metrics['acc'](preds, y)
        f1 = self.metrics['f1'](preds, y)
        auroc = self.metrics['auroc'](preds, y)
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        
        self.log('train_loss', loss, sync_dist=True, batch_size=self.config['batch_size'], on_epoch=True, on_step=False)
        self.log('train_acc', acc, sync_dist=True, batch_size=self.config['batch_size'], on_epoch=True, on_step=False)
        self.log('train_f1', f1, sync_dist=True, batch_size=self.config['batch_size'], on_epoch=True, on_step=False)
        self.log('train_auroc', auroc, sync_dist=True, batch_size=self.config['batch_size'], on_epoch=True, on_step=False)
        self.log('train_lr', lr, sync_dist=True, on_epoch=True, on_step=False)
        
        return {"loss": loss, "accuracy": acc, "f1": f1, "auroc": auroc, "lr": lr}
    
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