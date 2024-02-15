import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import BinaryF1Score

class Segmentation(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.learning_rate = 1e-3

        # Organize encoder and decoder under self.features
        self.features = nn.ModuleDict({
            'encoder1': nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            ),
            'pool1': nn.MaxPool2d(2),
            'encoder2': nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            ),
            'pool2': nn.MaxPool2d(2),
            # Decoder
            'upconv1': nn.ConvTranspose2d(128, 64, 2, stride=2),
            'decoder1': nn.Sequential(
                nn.Conv2d(128, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            ),
            # Adjust the upconv2 dimensions to match the final output size
            'upconv2': nn.ConvTranspose2d(64, 1, kernel_size=(30, 40), stride=(30, 40)),
        })

        self.binary_f1_score = BinaryF1Score(threshold=0.5)

    def forward(self, x):
        # Encoder
        e1 = self.features['encoder1'](x)
        p1 = self.features['pool1'](e1)
        e2 = self.features['encoder2'](p1)
        p2 = self.features['pool2'](e2)
        # Decoder
        up1 = self.features['upconv1'](p2)
        concat1 = torch.cat([up1, e1], dim=1)
        d1 = self.features['decoder1'](concat1)
        up2 = self.features['upconv2'](d1)
        return up2

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = F.binary_cross_entropy_with_logits(y_pred, y)
        f1_score = self.binary_f1_score(torch.sigmoid(y_pred), y)
        self.log('train_loss', loss)
        self.log('train_f1', f1_score)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = F.binary_cross_entropy_with_logits(y_pred, y)
        f1_score = self.binary_f1_score(torch.sigmoid(y_pred), y)
        self.log('val_loss', loss)
        self.log('val_f1', f1_score)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = F.binary_cross_entropy_with_logits(y_pred, y)
        f1_score = self.binary_f1_score(torch.sigmoid(y_pred), y)
        self.log('test_loss', loss)
        self.log('test_f1', f1_score)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)