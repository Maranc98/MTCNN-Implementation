import torch
import pytorch_lightning as pl

from torch import nn
from torch.nn import functional as F

import numpy as np
import os

class PNet(pl.LightningModule):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 10, kernel_size=3)
        self.prelu1 = nn.PReLU(10)
        self.pool1 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.conv2 = nn.Conv2d(10, 16, kernel_size=3)
        self.prelu2 = nn.PReLU(16)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3)
        self.prelu3 = nn.PReLU(32)

        # Classification
        self.conv4_1 = nn.Conv2d(32, 2, kernel_size=1)
        self.softmax4_1 = nn.Softmax(dim=1)
        # Regression
        self.conv4_2 = nn.Conv2d(32, 4, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.prelu2(x)
        x = self.conv3(x)
        x = self.prelu3(x)
        face_class = self.conv4_1(x)
        face_class = self.softmax4_1(face_class)
        bb_reg = self.conv4_2(x)

        return face_class, bb_reg

    def training_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self.forward(x)

        det_loss, box_loss = self.compute_loss(y_hat, y)
        loss = torch.add(det_loss, box_loss).sum()

        self.log('train_loss', loss)
        self.log('det_loss', det_loss.sum())
        self.log('box_loss', box_loss.sum())

        acc = (y_hat[0].max(dim=1).indices.view(y_hat[0].size()[0]) == y[:,3]).float().mean()

        self.log('train_acc', acc)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self.forward(x)

        det_loss, box_loss = self.compute_loss(y_hat, y)
        loss = torch.add(det_loss, box_loss).sum()

        acc = (y_hat[0].max(dim=1).indices.view(y_hat[0].size()[0]) == y[:,3]).float().mean()

        self.log("val_loss", loss)
        self.log("val_acc", acc)

    def compute_loss(self, y_hat, y):
        """
        Computes two kinds of losses: detection e regression.
        y = [b0, b1, b2, class, bb_x, bb_y, bb_w, bb_h, lx1, ly1, lx2...]
        With b multipliers for the two losses
        """
        (face_class, bb_reg) = y_hat

        class_gt = y[:, 3].long()
        bb_gt = y[:, 4:8]

        det_idx = torch.nonzero(y[:,0] != 0)[:,0]
        reg_idx = torch.nonzero(y[:,1] != 0)[:,0]

        class_gt = class_gt[det_idx][:,None,None]
        face_class = face_class[det_idx]

        bb_gt = bb_gt[reg_idx]
        bb_reg = bb_reg[reg_idx]

        det_loss = F.cross_entropy(face_class, class_gt)
        box_loss = F.mse_loss(bb_reg, bb_gt.view(bb_reg.size())) * 10

        return det_loss, box_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

class RNet(pl.LightningModule):

    def __init__(self, pretrained=True):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 28, kernel_size=3)
        self.prelu1 = nn.PReLU(28)
        self.pool1 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.conv2 = nn.Conv2d(28, 48, kernel_size=3)
        self.prelu2 = nn.PReLU(48)
        self.pool2 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.conv3 = nn.Conv2d(48, 64, kernel_size=2)
        self.prelu3 = nn.PReLU(64)
        self.dense4 = nn.Linear(576, 128)
        self.prelu4 = nn.PReLU(128)

        # Classification
        self.dense5_1 = nn.Linear(128, 2)
        self.softmax5_1 = nn.Softmax(dim=1)
        # Regression
        self.dense5_2 = nn.Linear(128, 4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.prelu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.prelu3(x)
        x = x.permute(0, 3, 2, 1).contiguous()
        x = self.dense4(x.view(x.shape[0], -1))
        x = self.prelu4(x)


        face_class = self.dense5_1(x)
        face_class = self.softmax5_1(face_class)
        bb_reg = self.dense5_2(x)
        return face_class, bb_reg

    def training_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self.forward(x)

        det_loss, box_loss = self.compute_loss(y_hat, y)
        loss = torch.add(det_loss, box_loss).sum()

        self.log('train_loss', loss)
        self.log('det_loss', det_loss.sum())
        self.log('box_loss', box_loss.sum())

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self.forward(x)

        det_loss, box_loss = self.compute_loss(y_hat, y)
        loss = torch.add(det_loss, box_loss).sum()

        # Compute accuracy
        det_idx = torch.nonzero(y[:,0] != 0)[:,0]

        class_gt = y[:, 3].long()
        class_gt = class_gt[det_idx]

        face_class = y_hat[0][det_idx]
        class_pred = face_class.max(1).indices

        acc = (class_gt == class_pred.view(class_gt.size())).sum() / class_gt.size(0)

        self.log("val_loss", loss)
        self.log("val_acc", acc)

    def compute_loss(self, y_hat, y):
        """
        Computes two kinds of losses: detection e regression.
        y = [b0, b1, b2, class, bb_x, bb_y, bb_w, bb_h, lx1, ly1, lx2...]
        With b multipliers for the two losses
        """
        (face_class, bb_reg) = y_hat

        class_gt = y[:, 3].long()
        bb_gt = y[:, 4:8]

        det_idx = torch.nonzero(y[:,0] != 0)[:,0]
        reg_idx = torch.nonzero(y[:,1] != 0)[:,0]

        class_gt = class_gt[det_idx]
        face_class = face_class[det_idx]

        bb_gt = bb_gt[reg_idx]
        bb_reg = bb_reg[reg_idx]

        det_loss = F.cross_entropy(face_class, class_gt)
        box_loss = F.mse_loss(bb_reg, bb_gt.view(bb_reg.size())) * 10

        return det_loss, box_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

class ONet(pl.LightningModule):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.prelu1 = nn.PReLU(32)
        self.pool1 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.prelu2 = nn.PReLU(64)
        self.pool2 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3)
        self.prelu3 = nn.PReLU(64)
        self.pool3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=2)
        self.prelu4 = nn.PReLU(128)
        self.dense5 = nn.Linear(1152, 256)
        self.prelu5 = nn.PReLU(256)

        # Classification
        self.dense6_1 = nn.Linear(256, 2)
        self.softmax6_1 = nn.Softmax(dim=1)
        # BB Regression
        self.dense6_2 = nn.Linear(256, 4)
        # Landmark regression
        self.dense6_3 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.prelu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.prelu3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.prelu4(x)
        x = x.permute(0, 3, 2, 1).contiguous()
        x = self.dense5(x.view(x.shape[0], -1))
        x = self.prelu5(x)

        face_class = self.dense6_1(x)
        face_class = self.softmax6_1(face_class)

        bb_reg = self.dense6_2(x)

        landmark_reg = self.dense6_3(x)

        return face_class, bb_reg, landmark_reg

    def training_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self.forward(x)

        det_loss, box_loss, landmark_loss = self.compute_loss(y_hat, y)
        loss = det_loss.sum() + box_loss.sum() + landmark_loss.sum()

        self.log('train_loss', loss)
        self.log('det_loss', det_loss.sum())
        self.log('box_loss', box_loss.sum())
        self.log('landmark_loss', landmark_loss.sum())

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self.forward(x)

        det_loss, box_loss, landmark_loss = self.compute_loss(y_hat, y)
        loss = det_loss.sum() + box_loss.sum() + landmark_loss.sum()

        # Compute accuracy
        det_idx = torch.nonzero(y[:,0] != 0)[:,0]

        class_gt = y[:, 3].long()
        class_gt = class_gt[det_idx]

        face_class = y_hat[0][det_idx]
        class_pred = face_class.max(1).indices

        acc = (class_gt == class_pred.view(class_gt.size())).sum() / class_gt.size(0)

        self.log("val_loss", loss)
        self.log("val_acc", acc)

    def compute_loss(self, y_hat, y):
        """
        Computes two kinds of losses: detection e regression.
        y = [b0, b1, b2, class, bb_x, bb_y, bb_w, bb_h, lx1, ly1, lx2...]
        With b multipliers for the two losses
        """
        (face_class, bb_reg, landmark_reg) = y_hat

        class_gt = y[:, 3].long()
        bb_gt = y[:, 4:8]
        landmark_gt = y[:, 8:]

        # Only compute loss portions for the correct samples

        det_idx = torch.nonzero(y[:,0] != 0)[:,0]
        bb_idx = torch.nonzero(y[:,1] != 0)[:,0]
        landmark_idx = torch.nonzero(y[:,2] != 0)[:,0]

        class_gt = class_gt[det_idx]
        face_class = face_class[det_idx]

        bb_gt = bb_gt[bb_idx]
        bb_reg = bb_reg[bb_idx]

        landmark_gt = landmark_gt[landmark_idx]
        landmark_reg = landmark_reg[landmark_idx]

        det_loss = F.cross_entropy(face_class, class_gt)
        box_loss = F.mse_loss(bb_reg, bb_gt) * 50
        landmark_loss = F.mse_loss(landmark_reg, landmark_gt) * 500

        return det_loss, box_loss, landmark_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), weight_decay=0.001)
