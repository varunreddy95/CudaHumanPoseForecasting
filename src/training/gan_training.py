import torch
import torch.nn as nn
from src.models.LSTMGAN import Discriminator, Seq2Seq
from src.configs.config import cfg
from torch.utils.data import DataLoader
import os
from src.data.datasets import Hpose
from paths import LOG_DIR
from torch.utils.tensorboard import SummaryWriter
from src.training.training import eval, save_model
import numpy as np

from src.util.visualization import draw_keypoint_sequence
from tqdm import tqdm

generator= Seq2Seq(256)
discriminator = Discriminator(64)

optim_discriminator = torch.optim.Adam(discriminator.parameters(), lr=3e-4, betas=(0.5, 0.999))
optim_generator = torch.optim.Adam(generator.parameters(), lr=3e-4, betas=(0.5, 0.999))

discriminator.to(cfg.DEVICE)
generator.to(cfg.DEVICE)

criterion = nn.MSELoss()

train_dataset = Hpose()
valid_dataset = Hpose("valid")

train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=cfg.BATCH_SIZE)

logdir="LSTM_GAN-256-64"
eval_criterion = nn.MSELoss()
best_val_loss = 10000.0
for epoch in range(100):
    if (not os.path.exists(os.path.join(LOG_DIR, logdir))):
        os.mkdir(os.path.join(LOG_DIR, logdir))
    """ Training a model for a given number of epochs"""
    writer = SummaryWriter(os.path.join(LOG_DIR, logdir))
    generator.train()
    discriminator.train()
    G_loss = []
    D_loss = []

    for batch, (seed_frames, gt_frames) in enumerate(tqdm(train_loader)):
        seed_frames, gt_frames = seed_frames.to(cfg.DEVICE), gt_frames.to(cfg.DEVICE) 

        generated_sequence = generator(seed_frames)

        true_labels = torch.ones(cfg.BATCH_SIZE).to(cfg.DEVICE)

        optim_discriminator.zero_grad()

        discriminator_output_true_data = discriminator(gt_frames)

        true_discriminator_loss = criterion(torch.squeeze(discriminator_output_true_data), true_labels)

        discriminator_output_for_generated_data = discriminator(generated_sequence.detach())

        generator_discriminator_loss = criterion(
            discriminator_output_for_generated_data, torch.zeros(cfg.BATCH_SIZE).to(cfg.DEVICE)
        )

        discriminator_loss = (
            true_discriminator_loss + generator_discriminator_loss
        )

        discriminator_loss.backward()
        optim_discriminator.step()

        D_loss.append(discriminator_loss.item())

        optim_generator.zero_grad()

        generated_sequence = generator(seed_frames)

        discriminator_output_generated_data = discriminator(generated_sequence)

        generator_loss = criterion(discriminator_output_generated_data, true_labels)

        generator_loss.backward()
        optim_generator.step()

        G_loss.append(generator_loss.item())

    v_loss = eval(generator, valid_loader, eval_criterion, cfg)
    writer.add_scalar("Loss/validation", v_loss, epoch)

    writer.add_scalar("LSTMGAN/Generator Loss", np.mean(G_loss), epoch)
    writer.add_scalar("LSTMGAN/Discriminator Loss", np.mean(D_loss), epoch)

    if epoch%10 == 0:
            seed_frames, gt = next(iter(valid_loader))
            generator.eval()
            prediction = generator(seed_frames.to(cfg.DEVICE))
            seed_img, gt_img, pred_img = draw_keypoint_sequence(torch.mul(seed_frames, 1000.0), torch.mul(gt, 1000.0), torch.mul(prediction, 1000.0))
            #seed_img, gt_img, pred_img = draw_heatmap(seed_frames, gt, prediction)

            writer.add_images("Seed Images", seed_img.detach().cpu(), epoch)
            writer.add_images("Ground Truth", gt_img.detach().cpu(), epoch)
            writer.add_images("Prediction", pred_img.detach().cpu(), epoch)

    if v_loss < best_val_loss:
        save_model(generator, optim_generator, epoch+1, logdir)
        best_val_loss = v_loss
        epochs_better = 0 


