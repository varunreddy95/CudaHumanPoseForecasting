from torch.utils.data import DataLoader
from src.data.datasets import Hpose
from src.data.datasetsHeatmap import HposeHeat
from torchvision.utils import draw_keypoints
import torch

import matplotlib.pyplot as plt

from src.configs.config import cfg 
import torch.nn as nn
from src.training.training import training, training_heatmap, training_joints


from src.models.model2 import StateSpaceModel_Cell


train_dataset = HposeHeat(subset="train")
train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE)

valid_dataset = HposeHeat(subset="valid")
valid_loader = DataLoader(valid_dataset, batch_size=cfg.BATCH_SIZE)

model = StateSpaceModel_Cell(encode_dim=128, hidden_dim=128).to(cfg.DEVICE)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN_LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

training_loss, _ = training_heatmap(model=model, optimizer=optimizer, scheduler=scheduler, criterion=criterion,
                            train_loader=train_loader, valid_loader=valid_loader, logdir="Heatmap-Statespace-128-128-MSE-training-heatmap", cfg=cfg)