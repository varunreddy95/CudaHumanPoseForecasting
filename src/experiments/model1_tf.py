from torch.utils.data import DataLoader
from src.data.datasets import Hpose
from torchvision.utils import draw_keypoints
import torch

import matplotlib.pyplot as plt

from locale import normalize
from src.configs.config import cfg 
import torch.nn as nn
from src.training.training import training, training_joints, training_last_frames_first, training_tf


from src.models.model1 import StateSpaceModel_Cell, AutoregressiveModel_Cell


if True:
    cfg.EPOCHS = 100
    cfg.STEPSIZE = 40
    train_dataset = Hpose()
    valid_dataset = Hpose("valid")
    print(len(train_dataset))
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.BATCH_SIZE)

    model = AutoregressiveModel_Cell(256, 256, residual_connection=False, teacher_forcing=True).to(cfg.DEVICE)
    criterion = nn.L1Loss(reduction="sum")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg.STEPSIZE, gamma=cfg.GAMMA)

    training_loss, _ = training_tf(model=model, optimizer=optimizer, scheduler=scheduler, criterion=criterion,
                                train_loader=train_loader, valid_loader=valid_loader, logdir="Autoregressive-256-256-MAE-Sum-NoResidual-TF-step40-lr-0001", cfg=cfg)