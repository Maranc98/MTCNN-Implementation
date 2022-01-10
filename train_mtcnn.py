import torch
import pytorch_lightning as pl

from models.mtcnn import PNet, RNet, ONet
from mtcnn_datasets import MTCNN_datasets

import wandb

model_type = "onet"
use_wandb = True

if model_type == "pnet":
    model = PNet()
elif model_type == "rnet":
    model = RNet()
elif model_type == "onet":
    model = ONet()

if use_wandb:
    wandb.init(project="MTCNN")

    wandb.config.model_type = model_type
    wandb.config.id = wandb.run.id

    wandb.config.info = "Cascade training"

    wandb.watch(model, log='all', log_freq=1)
    wandb_logger = pl.loggers.WandbLogger()
else:
    wandb_logger = None

early_stop_callback = pl.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=0.00,
    patience=3,
    verbose=False,
    mode='min'
)

trainer = pl.Trainer(
    weights_summary='full',
    gpus=-1,
    callbacks=[early_stop_callback],
    weights_save_path=".",
    logger=wandb_logger,
    log_every_n_steps=50
)

train_dataloader = MTCNN_datasets[model_type]['train']
val_dataloader = MTCNN_datasets[model_type]['val']

trainer.fit(model, train_dataloader, val_dataloader)
