import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

import pytorch_lightning as pl


class LitEfficientNetV2S(pl.LightningModule):
    def __init__(self, num_classes: int, lr: float = 1e-3, pretrained: bool = True):
        super().__init__()
        self.save_hyperparameters()

        weights = EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        self.model = efficientnet_v2_s(weights=weights)
        
        # Modify first conv layer to accept 1 channel instead of 3
        self.model.features[0][0] = nn.Conv2d(1, 24, kernel_size=3, stride=2, padding=1, bias=False)
        
        # Modify classifier to output num_classes
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log_dict({"train_loss": loss, "train_acc": acc}, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log_dict({"val_loss": loss, "val_acc": acc}, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)




