from argparse import ArgumentParser, Namespace

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from unet import UNet
from data import NYUDepthDataModule

class DepthMap(pl.LightningModule):
    def __init__(
            self,
            lr: float = 0.001,
            num_classes: int = 1,
            input_channels: int = 1,
            num_layers: int = 5,
            features_start: int = 64,
            bilinear: bool = False,
            **kwargs
    ):

        super().__init__()

        self.num_classes = num_classes
        self.input_channels = input_channels
        self.num_layers = num_layers
        self.features_start = features_start
        self.bilinear = bilinear
        self.lr = lr

        self.net = UNet(num_classes=num_classes,
                        input_channels=input_channels,
                        num_layers=self.num_layers,
                        features_start=self.features_start,
                        bilinear=self.bilinear)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_nb):
        img, target = batch
        out = self(img)
        loss_val = F.mse_loss(out.squeeze(), target.squeeze())
        log_dict = {'train_loss': loss_val}
        return {'loss': loss_val, 'log': log_dict, 'progress_bar': log_dict}

    def validation_step(self, batch, batch_idx):
        img, target = batch
        out = self(img)
        loss_val = F.mse_loss(out.squeeze(), target.squeeze())
        return {'val_loss': loss_val}

    def validation_epoch_end(self, outputs):
        loss_val = torch.stack([x['val_loss'] for x in outputs]).mean()
        log_dict = {'val_loss': loss_val}
        return {'log': log_dict, 'val_loss': log_dict['val_loss'], 'progress_bar': log_dict}

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
        return [opt], [sch]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--data_dir", type=str, default='.', help="path to nyu depth data")
        parser.add_argument("--resize", type=float, default=1, help="percent to downsample images")
        parser.add_argument("--input_channels", type=int, default=1, help="number of frames to use as input")
        parser.add_argument("--num_classes", type=int, default=1)
        parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
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
    dm = NYUDepthDataModule(args.data_dir, frames_per_sample=args.input_channels, resize=args.resize, batch_size=args.batch_size)

    # model
    model = DepthMap(**args.__dict__)

    # train
    trainer = pl.Trainer().from_argparse_args(args)
    trainer.fit(model, dm)