from argparse import ArgumentParser, Namespace

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from unet import UNet

from data import NYUDepthDataModule

class DepthMap(pl.LightningModule):
    def __init__(
            self,
            lr: float = 0.01,
            num_classes: int = 1,
            input_channels: int = 5,
            num_layers: int = 5,
            features_start: int = 64,
            bilinear: bool = False
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
        #img = img.float()
        #target = target.long()
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
        parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
        parser.add_argument("--lr", type=float, default=0.01, help="adam: learning rate")
        parser.add_argument("--num_layers", type=int, default=5, help="number of layers on u-net")
        parser.add_argument("--features_start", type=float, default=64, help="number of features in first layer")
        parser.add_argument("--bilinear", action='store_true', default=False,
                            help="whether to use bilinear interpolation or transposed")

        return parser

dm = NYUDepthDataModule('/Users/annikabrundyn/Developer/nyu_depth/data', batch_size=2)
model = DepthMap(num_classes=1, input_channels=3)

# train
trainer = pl.Trainer(fast_dev_run=True)
trainer.fit(model, dm)


# if __name__ == '__main__':
#     dm = NYUDepthDataModule('/Users/annikabrundyn/Developer/nyu_depth/data', batch_size=2)
#     model = DepthMap(num_classes=1, input_channels=5)
#
#     # train
#     trainer = pl.Trainer(fast_dev_run=True)
#     trainer.fit(model, dm)
