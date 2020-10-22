import os
import math
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random


class NYUDepth(Dataset):
    """https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html"""

    def __init__(self, root_dir, image_set='train', frames_per_sample=5, img_transform=None, target_transform=None):
        """
        Parameters:
            root_dir (string): Root directory of the dumped NYU-Depth dataset.
            image_set (string, optional): Select the image_set to use, ``train``, ``val``
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.image_set = image_set
        self.img_transform = transforms.Compose([transforms.Grayscale(),
                                                 transforms.ToTensor()])
        self.target_transform = transforms.Compose([transforms.ToTensor()])
        self.images = []
        self.targets = []
        self.videos = {}
        self.frames_per_sample = frames_per_sample
        img_list = self.read_image_list(os.path.join(root_dir, '{:s}.csv'.format(image_set)))

        for (img_filename, target_filename) in img_list:
            if os.path.isfile(img_filename) and os.path.isfile(target_filename):
                self.images.append(img_filename)
                self.targets.append(target_filename)
                key, jpg = img_filename.split('/')[2:]
                frame_num = jpg.split('.')[0]
                if key in self.videos:
                    self.videos[key].append(int(frame_num))
                else:
                    self.videos[key] = [int(frame_num)]

        # sort the frames and split into video snippets
        # TODO: add random dropping of frames
        self.all_samples = []
        for key, value in self.videos.items():
            self.videos[key].sort()
            # sample overlap size
            step_size = 1
            self.all_samples += ([(key, self.videos[key][i:i+self.frames_per_sample]) for i in range(0, len(self.videos[key])-self.frames_per_sample, step_size)])
            #self.videos[key] = samples_list
            #sample_path = f"data/nyu2_{image_set}/{key}/"
            #all_samples.append((key))

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
        return len(self.images)

    def __getitem__(self, index):
        sample = self.all_samples[index]
        video_name = sample[0]
        frames = sample[1]
        sample_path = f"data/nyu2_{self.image_set}/{video_name}/"

        images = []
        for frame in frames:
            img_path = sample_path + str(frame) + ".jpg"
            image = Image.open(img_path)
            image = self.img_transform(image)
            images.append(image)
        image_tensor = torch.stack(images)
        image_tensor = torch.squeeze(image_tensor)

        target_path = sample_path + str(frames[-1]) + ".png"
        target = Image.open(target_path)
        target = self.target_transform(target)

        return image_tensor, target


d = NYUDepth('/Users/annikabrundyn/Developer/nyu_depth/data')
loader = DataLoader(d, batch_size=32)
for img, target in loader:
    print(img.shape)
    print(target.shape)
    break
