import os
from paths import CHECKPOINT_DIR, LSTM_GAN, STATE_SPACE_JOINTS_DIR, AUTOREGRESSIVE_JOINTS_DIR, SEQ2SEQ_JOINTS_DIR
from src.data.datasets import Hpose
from src.training.training import eval, load_model
from src.models.model1 import AutoregressiveModel_Cell, StateSpaceModel_Cell
from src.models.seq2seq import Seq2Seq
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from src.configs.config import cfg
from src.training.losses import MPJPE
from src.training.losses import pck
from tqdm import tqdm
from src.util.visualization import draw_keypoint_sequence
import numpy as np
import matplotlib.pyplot as plt

cfg.DEVICE = torch.device("cpu")
cfg.BATCH_SIZE = 16

def eval_model(model, dataloader):
    """evaluates a given model on all 4 metrics we used

    Args:
        model (nn.Module): the model to be evaluated
        dataloader (torch.utils.data.Dataloader): a dataloader with the evaluation set

    Returns:
        Tuple: Tuple of the 4 metrics for the given model and dataloader
    """
    criterion = nn.MSELoss()
    mse = eval(model, dataloader, criterion, cfg)
    mse = mse/(cfg.BATCH_SIZE*len(dataloader))
    criterion = nn.L1Loss()
    mae = eval(model, dataloader, criterion, cfg)
    mae = mae/(cfg.BATCH_SIZE*len(dataloader))
    criterion = MPJPE()
    mpjpe = eval(model, dataloader, criterion, cfg)
    mpjpe = mpjpe/(cfg.BATCH_SIZE*len(dataloader))
    pck_score = eval_pck(model, dataloader, cfg)
    pck_score = pck_score/(cfg.BATCH_SIZE*len(dataloader))
    return mse, mae, mpjpe, pck_score
    
def eval_pck(model, dataloader, cfg):
    """evaluate pck metric for a given model

    Args:
        model (nn.Module): the model to be evaluated
        dataloader (torch.utils.data.Dataloader): dataloader with evaluation set
        cfg (src.config.config.DefaultConfig): a config object from our src code

    Returns:
        float: sum of all pck values in the dataset
    """
    model.eval()
    correct = 0
    loss_sum = 0

    with torch.no_grad():
        for i, (x, labels) in enumerate(tqdm(dataloader)):
            x, labels = x.to(cfg.DEVICE), labels.to(cfg.DEVICE)

            predictions = model(x)
            loss = pck(predictions, labels, 0.048)
            loss_sum += loss
            # _, predicted_labels = torch.max(predictions, 1)
            # predictions_tensor = torch.cat((predictions_tensor, predicted_labels))
            # ground_truth_tensor = torch.cat((ground_truth_tensor, labels))
            # # print(predicted_labels)
            # correct += torch.sum(predicted_labels == labels).item()

    # return correct, loss_sum / len(dataloader), predictions_tensor, ground_truth_tensor
    return loss_sum



dataset = Hpose(subset="valid")


state_space = StateSpaceModel_Cell(256,256, residual_connection=False)
state_optimizer = torch.optim.Adam(state_space.parameters())
dataloader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE)

state_space, state_optimizer, _ = load_model(state_space, state_optimizer, os.path.join(STATE_SPACE_JOINTS_DIR, "best_model.pth"))
state_space = state_space.to(cfg.DEVICE)
mse, mae, mpjpe, pck_score = eval_model(state_space, dataloader)
print("Best State Space")
print(mse, mae, mpjpe, pck_score)

autoregressive = AutoregressiveModel_Cell(256,256, residual_connection=False)
autoregressive_optimizer = torch.optim.Adam(autoregressive.parameters())

autoregressive, autoregressive_optimizer, _ = load_model(autoregressive, autoregressive_optimizer, os.path.join(AUTOREGRESSIVE_JOINTS_DIR, "best_model.pth"))
autoregressive = autoregressive.to(cfg.DEVICE)
mse, mae, mpjpe, pck_score = eval_model(autoregressive, dataloader)
print("Best Autoregressive")
print(mse, mae, mpjpe, pck_score)


seq_model = Seq2Seq(256)
seq_model_optimizer = torch.optim.Adam(seq_model.parameters())

seq_model, seq_model_optimizer, _ = load_model(seq_model, seq_model_optimizer, os.path.join(SEQ2SEQ_JOINTS_DIR, "best_model.pth"))
seq_model = seq_model.to(cfg.DEVICE)
mse, mae, mpjpe, pck_score = eval_model(seq_model, dataloader)
print("Best Seq2Seq")
print(mse, mae, mpjpe, pck_score)


generator= Seq2Seq(256)
generator_optimizer = torch.optim.Adam(generator.parameters())

generator, generator_optimizer, _ = load_model(generator, generator_optimizer, os.path.join(LSTM_GAN, "best_model.pth"))
generator = generator.to(cfg.DEVICE)
mse, mae, mpjpe, pck_score = eval_model(generator, dataloader)
print("Best LSTM GAN")
print(mse, mae, mpjpe, pck_score)


iterator = iter(dataloader)
seed_frames, gt = next(iterator)
seed_frames, gt = next(iterator)

prediction = state_space(seed_frames)

seed_img, gt_img, pred_img = draw_keypoint_sequence(torch.mul(seed_frames, 1000.0), torch.mul(gt, 1000.0), torch.mul(prediction, 1000.0), 1)

whole = torch.cat([seed_img, gt_img, pred_img])

prediction = autoregressive(seed_frames)

seed_img, gt_img, pred_img = draw_keypoint_sequence(torch.mul(seed_frames, 1000.0), torch.mul(gt, 1000.0), torch.mul(prediction, 1000.0), 1)

whole = torch.cat([whole, pred_img])

prediction = seq_model(seed_frames)

seed_img, gt_img, pred_img = draw_keypoint_sequence(torch.mul(seed_frames, 1000.0), torch.mul(gt, 1000.0), torch.mul(prediction, 1000.0), 1)

whole = torch.cat([whole, pred_img])

prediction = generator(seed_frames)

seed_img, gt_img, pred_img = draw_keypoint_sequence(torch.mul(seed_frames, 1000.0), torch.mul(gt, 1000.0), torch.mul(prediction, 1000.0), 1)

whole = torch.cat([whole, pred_img])

fig = plt.figure(figsize=(17, 9))
columns = 10
rows = 6


# ax enables access to manipulate each of subplots
ax = []

for i in range(columns*rows):
    img = torch.permute(whole[i], (1,2,0)).numpy()
    ax.append( fig.add_subplot(rows, columns, i+1) )
    plt.axis('off')
    plt.imshow(img)

ax[0].set_title("Seed Frames", loc="center")
ax[10].set_title("Ground Truth", loc="center")
ax[20].set_title("State Space Model Prediction", loc="center")
ax[30].set_title("Autoregressive Model Prediction", loc="center")
ax[40].set_title("Sequence to Sequence Model Prediction", loc="center")
ax[50].set_title("LSTM GAN Prediction", loc="center")

plt.show()
