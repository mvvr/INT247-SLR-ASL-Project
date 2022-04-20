import torch
import gc
from torch.utils.data import RandomSampler, DataLoader, random_split
from torchvision import datasets, transforms
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

training_set = []

gc.collect()
torch.cuda.empty_cache()

data_dir = 'ASL Model\\'
train_dir = os.path.join(data_dir, 'Data\\asl_alphabet_train\\asl_alphabet_train')

classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

data_transform = transforms.Compose([transforms.Resize((128, 128)), transforms.RandomHorizontalFlip(0.5), transforms.ToTensor()])
seed = torch.random.initial_seed()
train_data = datasets.ImageFolder(train_dir, transform=data_transform)
train_data, valid_data = torch.utils.data.random_split(train_data, [79832, 7168], generator=torch.Generator().manual_seed(seed))

num_workers = 0
batch_size = 128
training_loader = DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)
validating_loader = DataLoader(valid_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)

# iterator = iter(validating_loader)
# image, labels = iterator.next()
# image = image.numpy()

# for index in range(5):
#     image_png = np.transpose(image[index], (1, 2, 0))
#     cv2.imshow(str(labels[index]), image_png)
#     cv2.waitKey(0)
