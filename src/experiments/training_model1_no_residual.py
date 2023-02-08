from torch.utils.data import DataLoader
from src.data.datasets import Hpose
from torchvision.utils import draw_keypoints
import torch

import matplotlib.pyplot as plt

from locale import normalize
from src.configs.config import cfg 
import torch.nn as nn
from src.training.training import training, training_joints


from src.models.model1 import StateSpaceModel_Cell, AutoregressiveModel_Cell, AutoregressiveModel_Cell_4LSTM

from src.training.losses import MPJPE

if False:
    cfg.EPOCHS = 100
    cfg.GAMMA = 0.1
    cfg.STEPSIZE = 15
    train_dataset = Hpose()
    valid_dataset = Hpose("valid")
    print(len(train_dataset))
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.BATCH_SIZE)

    model = AutoregressiveModel_Cell(256, 256, residual_connection=False).to(cfg.DEVICE)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg.STEPSIZE, gamma=cfg.GAMMA)

    training_loss, _ = training_joints(model=model, optimizer=optimizer, scheduler=scheduler, criterion=criterion,
                                train_loader=train_loader, valid_loader=valid_loader, logdir="Autoregressive-256-256-MAE-NoResidual-noTF-lr-05-step-15", cfg=cfg)

if False:
    cfg.EPOCHS = 100
    cfg.GAMMA = 0.1
    cfg.STEPSIZE = 15
    train_dataset = Hpose()
    valid_dataset = Hpose("valid")
    print(len(train_dataset))
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.BATCH_SIZE)

    model = AutoregressiveModel_Cell(256, 256, residual_connection=False).to(cfg.DEVICE)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg.STEPSIZE, gamma=cfg.GAMMA)

    training_loss, _ = training_joints(model=model, optimizer=optimizer, scheduler=scheduler, criterion=criterion,
                                train_loader=train_loader, valid_loader=valid_loader, logdir="Autoregressive-256-256-MAE-NoResidual-noTF-lr-03-step-15", cfg=cfg)

if False:
    cfg.EPOCHS = 100
    cfg.GAMMA = 0.1
    cfg.STEPSIZE = 15
    train_dataset = Hpose()
    valid_dataset = Hpose("valid")
    print(len(train_dataset))
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.BATCH_SIZE)

    model = AutoregressiveModel_Cell(256, 256, residual_connection=False).to(cfg.DEVICE)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg.STEPSIZE, gamma=cfg.GAMMA)

    training_loss, _ = training_joints(model=model, optimizer=optimizer, scheduler=scheduler, criterion=criterion,
                                train_loader=train_loader, valid_loader=valid_loader, logdir="Autoregressive-256-256-MAE-NoResidual-noTF-lr-01-step-15", cfg=cfg)


if False:
    cfg.EPOCHS = 100
    cfg.GAMMA = 0.1
    cfg.STEPSIZE = 10
    train_dataset = Hpose()
    valid_dataset = Hpose("valid")
    print(len(train_dataset))
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.BATCH_SIZE)

    model = AutoregressiveModel_Cell(256, 256, residual_connection=False).to(cfg.DEVICE)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg.STEPSIZE, gamma=cfg.GAMMA)

    training_loss, _ = training_joints(model=model, optimizer=optimizer, scheduler=scheduler, criterion=criterion,
                                train_loader=train_loader, valid_loader=valid_loader, logdir="Autoregressive-256-256-MAE-NoResidual-noTF-lr-05-step-10", cfg=cfg)

if False:
    cfg.EPOCHS = 100
    cfg.GAMMA = 0.1
    cfg.STEPSIZE = 10
    train_dataset = Hpose()
    valid_dataset = Hpose("valid")
    print(len(train_dataset))
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.BATCH_SIZE)

    model = AutoregressiveModel_Cell(256, 256, residual_connection=False).to(cfg.DEVICE)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg.STEPSIZE, gamma=cfg.GAMMA)

    training_loss, _ = training_joints(model=model, optimizer=optimizer, scheduler=scheduler, criterion=criterion,
                                train_loader=train_loader, valid_loader=valid_loader, logdir="Autoregressive-256-256-MAE-NoResidual-noTF-lr-03-step-10", cfg=cfg)

if True:
    cfg.EPOCHS = 100
    cfg.GAMMA = 0.1
    cfg.STEPSIZE = 10
    train_dataset = Hpose()
    valid_dataset = Hpose("valid")
    print(len(train_dataset))
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.BATCH_SIZE)

    model = AutoregressiveModel_Cell(256, 256, residual_connection=False).to(cfg.DEVICE)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg.STEPSIZE, gamma=cfg.GAMMA)

    training_loss, _ = training_joints(model=model, optimizer=optimizer, scheduler=scheduler, criterion=criterion,
                                train_loader=train_loader, valid_loader=valid_loader, logdir="Autoregressive-256-256-MAE-NoResidual-noTF-lr-01-step-10", cfg=cfg)



if True:
    cfg.GAMMA = 0.5
    cfg.STEPSIZE = 20
    train_dataset = Hpose()
    valid_dataset = Hpose("valid")
    print(len(train_dataset))
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.BATCH_SIZE)

    model = AutoregressiveModel_Cell_4LSTM(256, 256, residual_connection=False).to(cfg.DEVICE)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg.STEPSIZE, gamma=cfg.GAMMA)

    training_loss, _ = training_joints(model=model, optimizer=optimizer, scheduler=scheduler, criterion=criterion,
                                train_loader=train_loader, valid_loader=valid_loader, logdir="Autoregressive-256-256-MAE-NoResidual-noTF-4LSTM", cfg=cfg)