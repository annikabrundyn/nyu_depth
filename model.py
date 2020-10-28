from argparse import ArgumentParser, Namespace

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision

from unet import UNet
from data import NYUDepthDataModule

import numpy as np
import matplotlib.pyplot as plt


class DepthMap(pl.LightningModule):
    def __init__(
            self,
            lr: float = 0.001,
            num_classes: int = 1,
            input_channels: int = 1,
            num_layers: int = 5,
            features_start: int = 64,
            bilinear: bool = False,
            output_img_freq : int = 100,
            batch_size : int = 32,
            **kwargs
    ):

        super().__init__()

        self.num_classes = num_classes
        self.input_channels = input_channels
        self.num_layers = num_layers
        self.features_start = features_start
        self.bilinear = bilinear
        self.lr = lr
        self.output_img_freq = output_img_freq
        self.batch_size = batch_size

        self.net = UNet(num_classes=num_classes,
                        input_channels=input_channels,
                        num_layers=self.num_layers,
                        features_start=self.features_start,
                        bilinear=self.bilinear)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        img, target = batch
        pred = self(img)
        if batch_idx % self.output_img_freq == 0:
            self._log_images(target, pred, step_name='train')
        loss_val = F.mse_loss(pred.squeeze(), target.squeeze())
        self.log('train_loss', loss_val, on_step=True)
        return loss_val

    def validation_step(self, batch, batch_idx):
        img, target = batch
        pred = self(img)
        if batch_idx % self.output_img_freq == 0:
            self._log_images(target, pred, step_name='valid')
        loss_val = F.mse_loss(pred.squeeze(), target.squeeze())
        self.log('valid_loss', loss_val, on_epoch=True)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        #sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
        return [opt]

    # def _matplotlib_imshow(self, img, one_channel=False):
    #     npimg = img.detach().numpy()
    #     figure = plt.figure()
    #     if one_channel:
    #         plt.imshow(npimg.squeeze(0), cmap="Greys")
    #     else:
    #         plt.imshow(np.transpose(npimg, (1, 2, 0)))
    #     return figure

    def _log_images(self, target, pred, step_name, limit=1):
        # TODO: Randomly select image from batch
        target = target[:limit]
        pred = pred[:limit]

        target_images = torchvision.utils.make_grid(target)
        pred_images = torchvision.utils.make_grid(pred)

        self.logger.experiment.add_image(f'{step_name}_target', target_images, self.trainer.global_step)
        self.logger.experiment.add_image(f'{step_name}_predicted', pred_images, self.trainer.global_step)

        #print(target.shape)
        #plt_target = target.squeeze(0).detach().numpy()
        #fig = plt.figure()
        #plt.imshow(plt_target, cmap="Greys")
        #self._matplotlib_imshow(torch_target, one_channel=True)
        #self._matplotlib_imshow(torch_pred, one_channel=True)

        #self.logger.experiment.add_figure(f'{step_name}_plt_torch_target', fig, self.trainer.global_step)
        #self.logger.experiment.add_image(f'{step_name}_torch_target', torch_target, self.trainer.global_step)
        #self.logger.experiment.add_figure(f'{step_name}_plt_torch_pred', torch_pred, self.trainer.global_step)

        #torch_new = 1 - torch_pred
        #print(torch_pred.shape)
        #print(pred.shape)
        #print(pred.squeeze(0).shape)
        #print(pred.squeeze(0).permute(1,2,0).shape)

        #plt_pred_raw = plt.imshow(pred.squeeze(0).permute(1, 2, 0).detach())
        #plt_pred_new = plt.imshow(torch_new.numpy().squeeze())
        #plt_pred_spectral = plt.imshow(torch_new.numpy().squeeze(), cmap='Spectral')
        #plt_pred_magma = plt.imshow(torch_new.numpy().squeeze(), cmap='magma')


        #self.logger.experiment.add_figure(f'{step_name}_plt_pred_raw', plt_pred_raw, self.trainer.global_step)
        # self.logger.experiment.add_figure(f'{step_name}_plt_pred_new', plt_pred_new, self.trainer.global_step)
        # self.logger.experiment.add_figure(f'{step_name}_plt_pred_spectral', plt_pred_spectral, self.trainer.global_step)
        # self.logger.experiment.add_figure(f'{step_name}_plt_plt_pred_magma', plt_pred_magma, self.trainer.global_step)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--data_dir", type=str, default='.', help="path to nyu depth data")
        parser.add_argument("--resize", type=float, default=1, help="percent to downsample images")
        parser.add_argument("--input_channels", type=int, default=1, help="number of frames to use as input")
        parser.add_argument("--num_classes", type=int, default=1)
        parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
        parser.add_argument("--output_img_freq", type=int, default=100)
        parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
        parser.add_argument("--num_layers", type=int, default=5, help="number of layers on u-net")
        parser.add_argument("--features_start", type=float, default=64, help="number of features in first layer")
        parser.add_argument("--bilinear", action='store_true', default=False,
                            help="whether to use bilinear interpolation or transposed")

        return parser


if __name__ == '__main__':
    pl.seed_everything(1234)

    parser = ArgumentParser()

    # trainer args
    parser = pl.Trainer.add_argparse_args(parser)

    # model args
    parser = DepthMap.add_model_specific_args(parser)
    args = parser.parse_args()

    # data
    dm = NYUDepthDataModule(args.data_dir, frames_per_sample=args.input_channels,
                            resize=args.resize,
                            batch_size=args.batch_size)

    # sanity checks
    print("size of dataset:", len(dm.dataset))
    print("size of trainset:", len(dm.trainset))
    print("size of validset:", len(dm.valset))

    # model
    model = DepthMap(**args.__dict__)

    # train
    trainer = pl.Trainer().from_argparse_args(args)
    trainer.fit(model, dm.train_dataloader(), dm.val_dataloader())
