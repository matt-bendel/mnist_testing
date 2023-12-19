import torch

import pytorch_lightning as pl

from torch.nn import functional as F

from torchmetrics.functional import peak_signal_noise_ratio

from architectures.autoencoder import Autoencoder


class MNISTAutoencoder(pl.LightningModule):
    def __init__(self, args, exp_name):
        super().__init__()
        self.args = args
        self.exp_name = exp_name

        self.in_chans = args.in_chans
        self.out_chans = args.out_chans

        self.autoencoder = Autoencoder()

        self.resolution = self.args.im_size
        self.automatic_optimization = False
        self.val_outputs = []

        self.save_hyperparameters()  # Save passed values

    def forward(self, x):
        return self.autoencoder(x)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x = (x - 0.1307) / 0.3081

        opt1 = self.optimizers()

        x_hat = self.mean_net(x)
        # x_hat = self.readd_measures(x_hat, y)
        mu_loss = F.mse_loss(x_hat, x)

        opt1.zero_grad()
        self.manual_backward(mu_loss)
        opt1.step()

        self.log('mse_loss', mu_loss, prog_bar=True)

    def validation_step(self, batch, batch_idx, external_test=False):
        x, _ = batch
        x = (x - 0.1307) / 0.3081

        x_hat = self.autoencoder(x) * 0.3081 + 0.1307

        x = x * 0.3081 + 0.1307
        psnr_val = peak_signal_noise_ratio(x_hat, x)

        self.log('psnr_val_step', psnr_val, on_step=True, on_epoch=False, prog_bar=True)

        self.val_outputs.append({'psnr_val': psnr_val})

        return {'psnr_val': psnr_val}

    def on_validation_epoch_end(self):
        psnr_val = torch.stack([x['psnr_val'] for x in self.val_outputs]).mean().mean()
        self.log('psnr_val', psnr_val)
        self.val_outputs = []

    def on_train_epoch_end(self):
        sch = self.lr_schedulers()

        sch.step(self.trainer.callback_metrics["psnr_val"])

    def configure_optimizers(self):
        opt_mean = torch.optim.Adam(self.mean_net.parameters(), lr=self.args.lr,
                                    betas=(self.args.beta_1, self.args.beta_2))
        reduce_lr_on_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt_mean,
            mode='max',
            factor=0.1,
            patience=3,
            min_lr=1e-6,
        )
        return {'optimizer': opt_mean, 'lr_scheduler': reduce_lr_on_plateau}
