import os
import random

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.data import random_split

import pytorch_lightning as pl
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

class NYUDepth(Dataset):

    def __init__(self,
                 root_dir,
                 image_set='train',
                 frames_per_sample=1,
                 resize=1,
                 img_transform=None,
                 target_transform=None
                 ):
        self.root_dir = root_dir
        self.image_set = image_set

        new_height = round(480*resize)
        new_width = round(640*resize)

        if not img_transform:
            self.img_transform = transforms.Compose([transforms.Grayscale(),
                                                     transforms.Resize((new_height, new_width)),
                                                     transforms.ToTensor()])
        else:
            self.img_transform = img_transform

        if not target_transform:
            self.target_transform = transforms.Compose([transforms.Resize((new_height, new_width)),
                                                        transforms.ToTensor()])
        else:
            self.target_transform = target_transform

        # create dict with each video name (of diff. scenes) as a key and a list of corresponding frames for that video
        self.videos = {}
        self.frames_per_sample = frames_per_sample
        img_list = self.read_image_list(os.path.join(root_dir, '{:s}.csv'.format(image_set)))

        for (img_filename, target_filename) in img_list:
            key, jpg = img_filename.split('/')[2:]
            frame_num = jpg.split('.')[0]
            if key in self.videos:
                self.videos[key].append(int(frame_num))
            else:
                self.videos[key] = [int(frame_num)]

        # sort the frames and create samples containing k frames per sample
        # TODO: add random dropping of frames - reduce correlation between samples
        self.all_samples = []
        for key, value in self.videos.items():
            self.videos[key].sort()
            step_size = 1 # sample overlap size
            self.all_samples += ([(key, self.videos[key][i:i+self.frames_per_sample]) for i in range(0, len(self.videos[key])-self.frames_per_sample, step_size)])
        print("len of all samples:", len(self.all_samples))

        # shuffle
        random.shuffle(self.all_samples)


    def read_image_list(self, filename):
        """
        Read one of the image index lists
        Parameters:
            filename (string):  path to the image list file
        Returns:
            list (int):  list of strings that correspond to image names
        """
        list_file = open(filename, 'r')
        img_list = []
        while True:
            next_line = list_file.readline()
            if not next_line:
                break
            jpg, png = next_line.rstrip().split(',')

            img_list.append((jpg, png))
        return img_list

    def __len__(self):
        return len(self.all_samples)

    def __getitem__(self, index):
        sample = self.all_samples[index]
        video_name = sample[0]
        frames = sample[1]

        images = []
        for frame in frames:
            img_path = os.path.join(self.root_dir, 'nyu2_{}'.format(self.image_set), video_name, '{}.jpg'.format(frame))
            image = Image.open(img_path)
            image = self.img_transform(image)
            images.append(image)

        image_tensor = torch.stack(images)
        image_tensor = torch.squeeze(image_tensor, 1)

        target_path = os.path.join(self.root_dir, 'nyu2_{}'.format(self.image_set), video_name, '{}.png'.format(frames[-1]))
        target = Image.open(target_path)
        target = self.target_transform(target)

        return image_tensor, target


class NYUDepthDataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_dir: str,
            frames_per_sample: int = 1,
            resize: float = 0.5,
            val_split: float = 0.2,
            num_workers: int = 4,
            batch_size: int = 32,
            seed: int = 42,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.data_dir = data_dir if data_dir is not None else os.getcwd()
        self.frames_per_sample = frames_per_sample
        self.resize = resize
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

        self.dataset = NYUDepth(self.data_dir, frames_per_sample=self.frames_per_sample, resize=self.resize)

        val_len = int(val_split * len(self.dataset))
        train_len = len(self.dataset) - val_len

        print(train_len)
        print(val_len)

        self.trainset, self.valset = random_split(self.dataset, lengths=[train_len, val_len])

    def train_dataloader(self):
        loader = DataLoader(self.trainset,
                            batch_size=self.batch_size,
                            shuffle=True,
                            num_workers=self.num_workers)
        return loader

    def val_dataloader(self):
        loader = DataLoader(self.valset,
                            batch_size=self.batch_size,
                            shuffle=False,
                            num_workers=self.num_workers)
        return loader

    # def test_dataloader(self):
    #     loader = DataLoader(self.testset,
    #                         batch_size=self.batch_size,
    #                         shuffle=False,
    #                         num_workers=self.num_workers)
    #     return loader


# dm = NYUDepthDataModule('/Users/annikabrundyn/Developer/nyu_depth/data/', batch_size=1)
# dl = dm.train_dataloader()
#
# img, target = next(iter(dl))
# img = img[0]
# target = target[0]
# print(img.shape)
# print(target.shape)
# new = 1 - target
# save_image(img, 'original_image.png', normalize=False)
# save_image(target, 'original_dm.png', normalize=False)
# save_image(new, 'new_dm.png', normalize=False)
#
# plt.imsave('plt_original_dm.png', target.numpy().squeeze())
# plt.imsave('new_dm.png', new.numpy().squeeze())
# plt.imsave('spectral_dm.png', new.numpy().squeeze(), cmap='Spectral')
# plt.imsave('viridis_dm.png', new.numpy().squeeze(), cmap='viridis')
# plt.imsave('plasma_dm.png', new.numpy().squeeze(), cmap='plasma')
# plt.imsave('inferno_dm.png', new.numpy().squeeze(), cmap='inferno')
# plt.imsave('magma_dm.png', new.numpy().squeeze(), cmap='magma')
# plt.imsave('cividis_dm.png', new.numpy().squeeze(), cmap='cividis')
#
#
# cm = plt.get_cmap('gist_rainbow')
# colored_image = cm(target.numpy().squeeze())
# Image.fromarray((colored_image[:, :, :3] * 255).astype(np.uint8)).save('test.png')