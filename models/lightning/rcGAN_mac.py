import torch

import pytorch_lightning as pl
from torch.nn import functional as F
import torch.autograd as autograd
import numpy as np
import matplotlib.pyplot as plt
from architectures.unet import Unet
from architectures.discriminator import Discriminator
from torchmetrics.functional import peak_signal_noise_ratio
from models.lightning.mnist_autoencoder import MNISTAutoencoder

class rcGAN(pl.LightningModule):
    def __init__(self, args, exp_name):
        super().__init__()
        self.args = args
        self.exp_name = exp_name

        self.in_chans = args.in_chans
        self.out_chans = args.out_chans

        self.generator = Unet(
            in_chans=self.in_chans + 1,
            out_chans=self.in_chans,
        )

        self.discriminator = Discriminator(
            in_chans=self.in_chans * 2
        )
        self.resolution = self.args.im_size
        self.betastd = 1
        self.automatic_optimization = False
        self.val_outputs = []

        self.save_hyperparameters()  # Save passed values

    def readd_measures(self, samples, measures):
        mask = torch.ones(samples.size(0), 1, 28, 28).to(samples.device)
        mask[:, :, 0:21, :] = 0
        samples = (1 - mask) * samples + measures

        return samples

    def get_noise(self, num_vectors):
        z = torch.randn(num_vectors, self.resolution, self.resolution, 1, device=self.device)

        return z.permute(0, 3, 1, 2)

    def forward(self, y):
        z = self.get_noise(y.size(0))
        input = torch.cat([y, z], dim=1)
        samples = self.generator(input)
        return self.readd_measures(samples, y)

    def compute_gradient_penalty(self, real_samples, fake_samples, y):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = torch.rand((real_samples.size(0), 1, 1, 1)).to(self.device)
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = self.discriminator(interpolates, y)
        # fake = Tensor(real_samples.shape[0], 1, d_interpolates.shape[-1], d_interpolates.shape[-1]).fill_(1.0).to(
        #     self.device)
        fake = torch.ones(real_samples.shape[0], 1).to(self.device)

        # Get gradient w.r.t. interpolates
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty + interpolates[:, 0, 0, 0].mean()*0

    def gradient_penalty(self, x_hat, x, y):
        gradient_penalty = self.compute_gradient_penalty(x, x_hat, y)

        return self.args.gp_weight * gradient_penalty

    def adversarial_loss_generator(self, y, gens):
        fake_pred = torch.zeros(size=(y.shape[0], self.args.num_z_train), device=self.device)
        for k in range(y.shape[0]):
            cond = torch.zeros(1, gens.shape[2], gens.shape[3], gens.shape[4], device=self.device)
            cond[0, :, :, :] = y[k, :, :, :]
            cond = cond.repeat(self.args.num_z_train, 1, 1, 1)
            temp = self.discriminator(gens[k], cond)
            fake_pred[k] = temp[:, 0]

        gen_pred_loss = torch.mean(fake_pred[0])
        for k in range(y.shape[0] - 1):
            gen_pred_loss += torch.mean(fake_pred[k + 1])

        adv_weight = 1e-5
        if self.current_epoch <= 4:
            adv_weight = 1e-2
        elif self.current_epoch <= 22:
            adv_weight = 1e-4

        return - adv_weight * gen_pred_loss.mean()

    def l1_std_p(self, avg_recon, gens, x):
        return F.l1_loss(avg_recon, x) - self.betastd * np.sqrt(
            2 / (np.pi * self.args.num_z_train * (self.args.num_z_train+ 1))) * torch.std(gens, dim=1).mean()

    def training_step(self, batch, batch_idx):
        x, _ = batch
        mask = torch.ones(x.size(0), 1, 28, 28).to(x.device)
        mask[:, :, 0:21, :] = 0
        y = x * mask
        x = (x - 0.1307) / 0.3081
        y = (y - 0.1307) / 0.3081

        opt_g, opt_d = self.optimizers()

        x_hat = self.forward(y)
        x_hat = self.readd_measures(x_hat, y)

        fake_pred = self.discriminator(x_hat, y)
        real_pred = self.discriminator(x, y)

        adv_loss_d = fake_pred.mean() - real_pred.mean()
        gp_loss_d = self.gradient_penalty(x_hat, x, y)
        drift_loss_d = 1e-3 * (real_pred ** 2).mean()

        self.log('adv_loss_d', adv_loss_d, prog_bar=True)
        self.log('gp_loss_d', gp_loss_d, prog_bar=True)
        self.log('drift_loss_d', drift_loss_d, prog_bar=True)

        d_loss = adv_loss_d + gp_loss_d + drift_loss_d

        opt_d.zero_grad()
        self.manual_backward(d_loss)
        opt_d.step()

        self.log('d_loss', d_loss, prog_bar=True)

        gens = torch.zeros(
            size=(y.size(0), self.args.num_z_train, self.args.in_chans, self.args.im_size, self.args.im_size),
            device=self.device)
        for z in range(self.args.num_z_train):
            gens[:, z, :, :, :] = self.forward(y)

        avg_recon = torch.mean(gens, dim=1)

        g_loss = self.adversarial_loss_generator(y, gens)
        g_loss += self.l1_std_p(avg_recon, gens, x)

        opt_g.zero_grad()
        self.manual_backward(g_loss)
        opt_g.step()

        self.log('g_loss', g_loss, prog_bar=True)


    def validation_step(self, batch, batch_idx, external_test=False):
        x, _ = batch
        mask = torch.ones(x.size(0), 1, 28, 28).to(x.device)
        mask[:, :, 0:21, :] = 0
        y = x * mask
        x = (x - 0.1307) / 0.3081
        y = (y - 0.1307) / 0.3081

        gens = torch.zeros(size=(y.size(0), self.args.num_z_valid, self.args.in_chans, self.args.im_size, self.args.im_size),
                           device=self.device)
        for z in range(self.args.num_z_valid):
            gens[:, z, :, :, :] = self.forward(y) * 0.3081 + 0.1307

        x = x * 0.3081 + 0.1307
        y = y * 0.3081 + 0.1307

        avg = torch.mean(gens, dim=1)

        psnr_8 = peak_signal_noise_ratio(avg, x)
        psnr_1 = peak_signal_noise_ratio(gens[:, 0, :, :, :], x)

        self.val_outputs.append({'psnr_8': psnr_8, 'psnr_1': psnr_1})

        if batch_idx <= 2:
            if self.global_rank == 0:
                images = []

                x_np = x[0, 0, :, :].cpu().numpy()
                x_hat_np = avg[0, 0, :, :].cpu().numpy()

                plt.figure()
                plt.imshow(x_hat_np, cmap='gray')
                plt.savefig(f'test_recon_rcgan_{batch_idx}.png')
                plt.close()

                plt.figure()
                plt.imshow(np.abs(x_np - x_hat_np), cmap='jet')
                plt.savefig(f'test_error_rcgan_{batch_idx}.png')
                plt.close()

            self.trainer.strategy.barrier()

        return {'psnr_8': psnr_8, 'psnr_1': psnr_1}

    def on_validation_epoch_end(self):
        psnr_8 = torch.stack([x['psnr_8'] for x in self.val_outputs]).mean().mean()
        psnr_1 = torch.stack([x['psnr_1'] for x in self.val_outputs]).mean().mean()

        self.log('psnr_8', psnr_8)
        self.log('psnr_1', psnr_1)

        psnr_diff = (psnr_1 + 2.5) - psnr_8

        mu_0 = 2e-2
        self.betastd += mu_0 * psnr_diff

        self.val_outputs = []

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.args.lr,
                                 betas=(self.args.beta_1, self.args.beta_2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.args.lr,
                                   betas=(self.args.beta_1, self.args.beta_2))
        return [[opt_g, opt_d], []]

class rcGANLatent(pl.LightningModule):
    def __init__(self, args, exp_name):
        super().__init__()
        self.args = args
        self.exp_name = exp_name

        self.in_chans = args.in_chans
        self.out_chans = args.out_chans

        self.generator = Unet(
            in_chans=self.in_chans + 1,
            out_chans=self.in_chans,
        )

        self.discriminator = Discriminator(
            in_chans=self.in_chans * 2
        )
        self.resolution = self.args.im_size
        self.betastd = 1
        self.automatic_optimization = False
        self.val_outputs = []

        self.autoencoder = MNISTAutoencoder.load_from_checkpoint(
            '/storage/matt_models/mnist/autoencoder/best.ckpt').autoencoder
        self.autoencoder.eval()

        self.save_hyperparameters()  # Save passed values

    def readd_measures(self, samples, measures):
        mask = torch.ones(samples.size(0), 1, 28, 28).to(samples.device)
        mask[:, :, 0:21, :] = 0
        samples = (1 - mask) * samples + measures

        return samples

    def get_noise(self, num_vectors):
        z = torch.randn(num_vectors, self.resolution, self.resolution, 1, device=self.device)

        return z.permute(0, 3, 1, 2)

    def forward(self, y):
        z = self.get_noise(y.size(0))
        input = torch.cat([y, z], dim=1)
        samples = self.generator(input)
        return self.readd_measures(samples, y)

    def compute_gradient_penalty(self, real_samples, fake_samples, y):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = torch.rand((real_samples.size(0), 1, 1, 1)).to(self.device)
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = self.discriminator(interpolates, y)
        # fake = Tensor(real_samples.shape[0], 1, d_interpolates.shape[-1], d_interpolates.shape[-1]).fill_(1.0).to(
        #     self.device)
        fake = torch.ones(real_samples.shape[0], 1).to(self.device)

        # Get gradient w.r.t. interpolates
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty + interpolates[:, 0, 0, 0].mean()*0

    def gradient_penalty(self, x_hat, x, y):
        gradient_penalty = self.compute_gradient_penalty(x, x_hat, y)

        return self.args.gp_weight * gradient_penalty

    def adversarial_loss_generator(self, y, gens):
        fake_pred = torch.zeros(size=(y.shape[0], self.args.num_z_train), device=self.device)
        for k in range(y.shape[0]):
            cond = torch.zeros(1, gens.shape[2], gens.shape[3], gens.shape[4], device=self.device)
            cond[0, :, :, :] = y[k, :, :, :]
            cond = cond.repeat(self.args.num_z_train, 1, 1, 1)
            temp = self.discriminator(gens[k], cond)
            fake_pred[k] = temp[:, 0]

        gen_pred_loss = torch.mean(fake_pred[0])
        for k in range(y.shape[0] - 1):
            gen_pred_loss += torch.mean(fake_pred[k + 1])

        adv_weight = 1e-5
        if self.current_epoch <= 4:
            adv_weight = 1e-2
        elif self.current_epoch <= 22:
            adv_weight = 1e-4

        return - adv_weight * gen_pred_loss.mean()

    def l1_std_p(self, avg_recon, gens, x):
        return F.l1_loss(avg_recon, x) - self.betastd * np.sqrt(
            2 / (np.pi * self.args.num_z_train * (self.args.num_z_train+ 1))) * torch.std(gens, dim=1).mean()

    def training_step(self, batch, batch_idx):
        x, _ = batch
        mask = torch.ones(x.size(0), 1, 28, 28).to(x.device)
        mask[:, :, 0:21, :] = 0
        y = x * mask
        x = (x - 0.1307) / 0.3081
        y = (y - 0.1307) / 0.3081

        opt_g, opt_d = self.optimizers()

        x_hat = self.forward(y)
        x_hat = self.readd_measures(x_hat, y)

        fake_pred = self.discriminator(x_hat, y)
        real_pred = self.discriminator(x, y)

        adv_loss_d = fake_pred.mean() - real_pred.mean()
        gp_loss_d = self.gradient_penalty(x_hat, x, y)
        drift_loss_d = 1e-3 * (real_pred ** 2).mean()

        self.log('adv_loss_d', adv_loss_d, prog_bar=True)
        self.log('gp_loss_d', gp_loss_d, prog_bar=True)
        self.log('drift_loss_d', drift_loss_d, prog_bar=True)

        d_loss = adv_loss_d + gp_loss_d + drift_loss_d

        opt_d.zero_grad()
        self.manual_backward(d_loss)
        opt_d.step()

        self.log('d_loss', d_loss, prog_bar=True)

        gens = torch.zeros(
            size=(y.size(0), self.args.num_z_train, 1, 28, 28),
            device=self.device)
        for z in range(self.args.num_z_train):
            gens[:, z, :, :, :] = self.forward(y)

        g_loss = self.adversarial_loss_generator(y, gens)

        new_gens = torch.zeros(
            size=(y.size(0), self.args.num_z_train, 8, 7, 7),
            device=self.device)
        for z in range(self.args.num_z_train):
            new_gens[:, z, :, :, :] = self.autoencoder(gens[:, z, :, :, :], features=True)

        avg_recon = torch.mean(new_gens, dim=1)

        g_loss += self.l1_std_p(avg_recon, new_gens, self.autoencoder(x))

        opt_g.zero_grad()
        self.manual_backward(g_loss)
        opt_g.step()

        self.log('g_loss', g_loss, prog_bar=True)


    def validation_step(self, batch, batch_idx, external_test=False):
        x, _ = batch
        mask = torch.ones(x.size(0), 1, 28, 28).to(x.device)
        mask[:, :, 0:21, :] = 0
        y = x * mask
        x = (x - 0.1307) / 0.3081
        y = (y - 0.1307) / 0.3081

        gens = torch.zeros(size=(y.size(0), self.args.num_z_valid, self.args.in_chans, self.args.im_size, self.args.im_size),
                           device=self.device)
        for z in range(self.args.num_z_valid):
            gens[:, z, :, :, :] = self.forward(y) * 0.3081 + 0.1307

        x = x * 0.3081 + 0.1307
        y = y * 0.3081 + 0.1307

        avg = torch.mean(gens, dim=1)

        psnr_8 = peak_signal_noise_ratio(avg, x)
        psnr_1 = peak_signal_noise_ratio(gens[:, 0, :, :, :], x)

        self.val_outputs.append({'psnr_8': psnr_8, 'psnr_1': psnr_1})

        if batch_idx <= 2:
            if self.global_rank == 0:
                images = []

                x_np = x[0, 0, :, :].cpu().numpy()
                x_hat_np = avg[0, 0, :, :].cpu().numpy()

                plt.figure()
                plt.imshow(x_hat_np, cmap='gray')
                plt.savefig(f'test_recon_rcgan_{batch_idx}.png')
                plt.close()

                plt.figure()
                plt.imshow(np.abs(x_np - x_hat_np), cmap='jet')
                plt.savefig(f'test_error_rcgan_{batch_idx}.png')
                plt.close()

            self.trainer.strategy.barrier()

        return {'psnr_8': psnr_8, 'psnr_1': psnr_1}

    def on_validation_epoch_end(self):
        psnr_8 = torch.stack([x['psnr_8'] for x in self.val_outputs]).mean().mean()
        psnr_1 = torch.stack([x['psnr_1'] for x in self.val_outputs]).mean().mean()

        self.log('psnr_8', psnr_8)
        self.log('psnr_1', psnr_1)

        psnr_diff = (psnr_1 + 2.5) - psnr_8

        mu_0 = 2e-2
        self.betastd += mu_0 * psnr_diff

        self.val_outputs = []

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.args.lr,
                                 betas=(self.args.beta_1, self.args.beta_2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.args.lr,
                                   betas=(self.args.beta_1, self.args.beta_2))
        return [[opt_g, opt_d], []]
