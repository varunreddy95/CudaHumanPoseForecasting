import torch
from tqdm import tqdm
import numpy as np


class Trainer:
    """
    Class for initializing GAN and training it
    """

    def __init__(self, latent_dim=128, hidden_dims=[256, 512, 1024], img_dim=1024, writer=None):
        """ Initialzer """
        assert writer is not None, f"Tensorboard writer not set..."

        self.latent_dim = latent_dim
        self.writer = writer
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        generator = Generator(latent_dim=latent_dim,
                              hidden_dims=hidden_dims, out_dim=img_dim)
        discriminator = Discriminator(
            img_shape=img_dim, hidden_dims=hidden_dims[::-1], out_dim=1)
        self.generator = generator.to(self.device)
        self.discriminator = discriminator.to(self.device)

        self.optim_discriminator = torch.optim.Adam(
            self.discriminator.parameters(), lr=3e-4, betas=(0.5, 0.999))
        self.optim_generator = torch.optim.Adam(
            self.generator.parameters(), lr=3e-4, betas=(0.5, 0.999))

        # alternatively, you can use the BCE loss from PyTorch
        eps = 1e-10
        # self.criterion_d_real = lambda pred: torch.clip(-torch.log(1 - pred + eps), min=-10).mean()
        # self.criterion_d_fake = lambda pred: torch.clip(-torch.log(pred + eps), min=-10).mean()
        # self.criterion_g = lambda pred: torch.clip(-torch.log(1-pred + eps), min=-10).mean()
        self.criterion = nn.MSELoss()
        self.real_label = 1
        self.fake_label = 0

        self.hist = {
            "d_real": [],
            "d_fake": [],
            "g": []
        }
        return

    def train_one_step(self, imgs):
        """ Training both models for one optimization step """

        self.generator.train()
        self.discriminator.train()

        # Sample from the latent distribution
        B = imgs.shape[0]
        latent = torch.randn(B, self.latent_dim).to(self.device)

        ##################################
        # ==== Training Discriminator ====
        ##################################
        self.optim_discriminator.zero_grad()
        # Get discriminator outputs for the real samples
        prediction_real = self.discriminator(imgs)
        # Compute the loss function
        d_loss_real = self.criterion(
            input=prediction_real,
            target=self.real_label * torch.ones(B, device=self.device)
        )

        # Generating fake samples with the generator
        fake_samples = self.generator(latent)
        # Get discriminator outputs for the fake samples
        prediction_fake_d = self.discriminator(
            fake_samples.detach())  # why detach?
        # Compute the loss function
        d_loss_fake = self.criterion(
            input=prediction_fake_d,
            target=self.fake_label * torch.ones(B, device=self.device)
        )
        (d_loss_real + d_loss_fake).backward()

        # optimization step
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 3.0)
        self.optim_discriminator.step()

        #############################
        # === Train the generator ===
        #############################
        self.optim_generator.zero_grad()
        # Get discriminator outputs for the fake samples
        prediction_fake_g = self.discriminator(fake_samples)
        # Compute the loss function
        g_loss = self.criterion(
            input=prediction_fake_g,
            target=self.real_label * torch.ones(B, device=self.device)
        )
        g_loss.backward()
        # optimization step
        self.optim_generator.step()

        return d_loss_real + d_loss_fake, g_loss

    @torch.no_grad()
    def generate(self, N=64):
        """ Generating a bunch of images using current state of generator """
        self.generator.eval()
        latent = torch.randn(N, self.latent_dim).to(self.device)
        samples = self.generator(latent)
        imgs = vector_to_image(samples)
        imgs = imgs * 0.5 + 0.5
        return imgs

    def train(self, data_loader, N_iters=10000, init_step=0):
        """ Training the models for several iterations """

        progress_bar = tqdm(total=N_iters, initial=init_step)
        running_d_loss = 0
        running_g_loss = 0

        iter_ = 0
        for i in range(N_iters):
            for real_batch, _ in data_loader:
                real_samples = image_to_vector(real_batch)
                real_samples = real_samples.to(self.device)
                d_loss, g_loss = self.train_one_step(imgs=real_samples)

                # updating progress bar
                progress_bar.set_description(
                    f"Ep {i+1} Iter {iter_}: D_Loss={round(d_loss.item(),5)}, G_Loss={round(g_loss.item(),5)})")

                # adding stuff to tensorboard
                self.writer.add_scalar(
                    f'Loss/Discriminator Loss', d_loss.item(), global_step=iter_)
                self.writer.add_scalar(
                    f'Loss/Generator Loss', g_loss.item(), global_step=iter_)
                self.writer.add_scalars(f'Comb_Loss/Losses', {
                    'Discriminator': d_loss.item(),
                    'Generator':  g_loss.item()
                }, iter_)
                if(iter_ % 500 == 0):
                    imgs = self.generate()
                    grid = torchvision.utils.make_grid(imgs, nrow=8)
                    self.writer.add_image('images', grid, global_step=iter_)
                    torchvision.utils.save_image(grid, os.path.join(
                        os.getcwd(), "imgs", "training", f"imgs_{iter_}.png"))

                iter_ = iter_ + 1

        return
