import torch
import numpy as np
import pytorch_lightning as pl
import wandb
import collections
import torch.nn as nn

from torchvision.models import resnet
from torch.optim import Adam
import torch.nn.functional as F

from sklearn.metrics import precision_recall_fscore_support

from holograpy.layers import Classifier


#def init_weights(*models):
#    for model in models:
#        for name, param in model.named_parameters():
#            if "weight" in name or "bias" in name:
#                param.data.uniform_(-0.1, 0.1)

class ClassificationModel(pl.LightningModule):
    def __init__(self, optimizer_args: Dict[str, int]):
        super().__init__()
        self.optimizer_args = optimizer_args
        self.model = resnet.resnet34(pretrained=False)

        self.model.conv1= nn.Conv2d(1, 64, kernel_size=(7,7), stride=2, padding=3, bias =False)
        self.model.fc = Classifier(input_siz=512, num_classes = 4, dropout= 0)

        #init_weights(self.model)
        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), **self.optimizer_args)
        return optimizer
     
    def forward(self, x) -> torch.Tensor:
        return self.model(x)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch

        logits = self(x)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct( logits, y)

        with torch.no_grad():
            preds = F.softmax(logits, dim=-1).argmax(dim=-1)
            acc = ((preds == y).sum().float()) / len(y)

        #preds = preds.unsqueeze(0) if preds.dim() == 0 else preds
        return {"loss": loss, "preds": preds, "targets": y}

    def training_epoch_end(self, outs):
        preds = []
        targets = []
        losses = []
        for out in outs:
            losses.append(out["loss"] * len(out["targets"]))
            targets.append(out["targets"])
            preds.append(out["preds"])

        preds = torch.cat(preds)
        targets = torch.cat(targets)
        loss = sum(losses) / len(targets)
        acc = ((preds == targets).sum().float()) / len(targets)
        print()
        print(f"===>TRAIN BATCH ACCUR: {acc}")
        print()
        self.log_dict({"train_loss": loss, "train_acc": acc})

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self(x)

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits, y)
        with torch.no_grad():

            preds = F.softmax(logits, dim=-1).argmax(dim=-1)
            acc = ((preds == y).sum().float()) / len(y)
            print()
            print(f"===>VAL PREDS: {preds}")
            print(f"===>VAL LABEL: {y}")
            print(f"===>VAL ACCUR: {acc}")
            print()
        
        self.log("val_acc", acc)
        return {"val_acc": acc, "val_loss": loss, "preds": preds, "targets": y}

    def validation_epoch_end(self, outs):
        
        preds = []
        targets = []
        losses = []
        for out in outs:
            losses.append(out["val_loss"] * len(out["targets"]))
            targets.append(out["targets"])
            preds.append(out["preds"])
        preds = torch.cat(preds)
        targets = torch.cat(targets)
        loss = sum(losses) / len(targets)
        acc = ((preds == targets).sum().float()) / len(targets)
        print()
        print(f"===>VAL BATCH ACCUR: {acc}")
        print()
        self.log_dict({"val_loss": loss, "val_acc": acc})


    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        logits = self(x)
        loss_fct = nn.CrossEntropyLoss()
        
        preds = F.softmax(logits, dim=-1).argmax(dim=-1)
        probas = F.softmax(logits, dim=-1)
        loss = loss_fct(logits, y).cpu()

        y = y.cpu().tolist()
        probas = probas.cpu().tolist()
        preds = preds.cpu().tolist()
        return (y, probas, preds, loss)

    def test_epoch_end(self, outputs):
        labels = []
        probas = []
        preds = []
        losses = []
        source_indices = []
        target_indices = []
        for i in outputs:
            labels.extend(i[0])
            probas.extend(i[1])
            preds.extend(i[2])

        self.logger.experiment.log(
            {
                "cm": wandb.sklearn.plot_confusion_matrix(
                    np.array(labels), np.array(preds), ["80", "150", "400", "600"]
                )
            }
        )
        precision, recall, fscore, _ = precision_recall_fscore_support(
            labels, preds, average=None
        )
        self.log_dict({"precision": precision, "recall": recall, "fscore": fscore})

