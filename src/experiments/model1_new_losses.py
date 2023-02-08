from torch.utils.data import DataLoader
from src.data.datasets import Hpose
from torchvision.utils import draw_keypoints
import torch

import matplotlib.pyplot as plt

from src.configs.config import cfg 
import torch.nn as nn
from src.training.training import training, training_joints


from src.models.model1 import StateSpaceModel_Cell, AutoregressiveModel_Cell

from src.training.losses import L1Custom


if False:
    cfg.EPOCHS = 100
    train_dataset = Hpose()
    valid_dataset = Hpose("valid")
    print(len(train_dataset))
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.BATCH_SIZE)

    model = AutoregressiveModel_Cell(256, 256, residual_connection=False).to(cfg.DEVICE)
    criterion = nn.L1Loss(reduction="sum")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg.STEPSIZE, gamma=cfg.GAMMA)

    training_loss, _ = training_joints(model=model, optimizer=optimizer, scheduler=scheduler, criterion=criterion,
                                train_loader=train_loader, valid_loader=valid_loader, logdir="Autoregressive-256-256-MAE-Sum-NoResidual-noTF", cfg=cfg)

if False:
    cfg.EPOCHS = 100
    train_dataset = Hpose()
    valid_dataset = Hpose("valid")
    print(len(train_dataset))
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.BATCH_SIZE)

    model = AutoregressiveModel_Cell(256, 256, residual_connection=False).to(cfg.DEVICE)
    criterion = L1Custom()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg.STEPSIZE, gamma=cfg.GAMMA)

    training_loss, _ = training_joints(model=model, optimizer=optimizer, scheduler=scheduler, criterion=criterion,
                                train_loader=train_loader, valid_loader=valid_loader, logdir="Autoregressive-256-256-MAE-Custom-NoResidual-noTF", cfg=cfg)


if True:
    cfg.EPOCHS = 100
    train_dataset = Hpose()
    valid_dataset = Hpose("valid")
    print(len(train_dataset))
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.BATCH_SIZE)

    model = StateSpaceModel_Cell(256, 256, residual_connection=True).to(cfg.DEVICE)
    criterion = nn.L1Loss(reduction="sum")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg.STEPSIZE, gamma=cfg.GAMMA)

    training_loss, _ = training_joints(model=model, optimizer=optimizer, scheduler=scheduler, criterion=criterion,
                                train_loader=train_loader, valid_loader=valid_loader, logdir="Statespace-256-256-MAE-Sum-Residual-noTF-cloned", cfg=cfg)