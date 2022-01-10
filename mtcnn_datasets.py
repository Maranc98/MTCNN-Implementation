from torch.utils.data import DataLoader
import torch
from skimage import io, transform
import numpy as np
import os
import torchvision
from torchvision import transforms, datasets
from torch.utils.data.sampler import SubsetRandomSampler

import matplotlib.pyplot as plt

VALIDATION_SPLIT = 0.3
BATCH_SIZE = 200

data_transforms = torchvision.transforms.Compose([
    transforms.ToTensor(),
    torchvision.transforms.ColorJitter(),
    #torchvision.transforms.RandomAffine(10),
    #torchvision.transforms.RandomHorizontalFlip(),
    #torchvision.transforms.RandomPerspective(),
    #torchvision.transforms.RandomRotation(10),
])

types = ["pnet", "rnet", "onet"]
extraction_types = ["proximity/negative", "proximity/partfaces", "proximity/positive", "landmarks"]
loss_factors = {
    "proximity/negative": [1, 0, 0],
    "proximity/partfaces": [0, 1, 0],
    "proximity/positive": [1, 1, 0],
    "landmarks": [0, 0, 1],
}

class WIDERDataset(torch.utils.data.Dataset):
    """Extracts data from the WIDER_FACES extracted data

    From the original paper:
    - Negatives and positives are used for detection
    - Positives and part faces for bb regression
    - Only landmark faces for landmark regression

    WIDER_FACES does not contain landmark information.
    """

    def __init__(self, data_folder, type, extraction_type, transform=None):
        """
        Args:
            data_folder: Folder path of the data
            type: [pnet, rnet, onet] the kind of dataset we want
            extraction_type: the type of extraction used
        """

        assert type in types
        assert extraction_type in extraction_types

        self.data_folder = data_folder
        self.type = type
        self.extraction_type = extraction_type
        self.transform = transform

        self.data_path_prefix = f"{self.data_folder}/{self.type}/{self.extraction_type}/"

        self.loss_factors = loss_factors[self.extraction_type]


    def __len__(self):
        if self.type == "rnet":
            return {
                "proximity/negative": 164072,
                "proximity/partfaces": 62793,
                "proximity/positive": 16217,
            }[self.extraction_type]

        if self.type == "onet":
            return {
                "proximity/negative": 147028,
                "proximity/partfaces": 56727,
                "proximity/positive": 16172,
                "landmarks": 68489,
            }[self.extraction_type]

        return 200000

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = np.load(self.data_path_prefix + f"input{idx}.npy")
        output = np.load(self.data_path_prefix + f"output{idx}.npy")

        class_ = 1 if output[0] > 0.65 else 0

        labels = self.loss_factors + [class_] + output[1:].tolist()

        if "landmarks" not in self.extraction_type:
            labels += [0]*10

        if self.transform:
            image = self.transform(image)

        return image, torch.Tensor(labels)

class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets
        self.lens = [d.__len__() for d in self.datasets]
        self.len = sum(self.lens)

    def __getitem__(self, i):
        for dataset_idx, l in enumerate(self.lens):

            if i >= l:
                i -= l
            else:
                break

        return self.datasets[dataset_idx].__getitem__(i)

    def __len__(self):
        return self.len

def split_dataset(dataset, validation_split):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(VALIDATION_SPLIT * dataset_size))

    np.random.seed(127)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=16)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, sampler=valid_sampler, num_workers=16)

    return train_loader, validation_loader

MTCNN_datasets = {}

for type in types:

    datasets = [
        WIDERDataset('datasets/WIDER/MTCNN_face_crops/', type, 'proximity/negative', data_transforms),
        WIDERDataset('datasets/WIDER/MTCNN_face_crops/', type, 'proximity/partfaces', data_transforms),
        WIDERDataset('datasets/WIDER/MTCNN_face_crops/', type, 'proximity/positive', data_transforms),
    ]

    if type == "onet":
        datasets += [WIDERDataset('datasets/CelebA/', type, 'landmarks', data_transforms),]

    cd = ConcatDataset(*datasets)
    train_loader, validation_loader = split_dataset(cd, VALIDATION_SPLIT)

    MTCNN_datasets[type] = {"train": train_loader, "val": validation_loader}



if __name__ == "__main__":
    fig = plt.figure()

    datasets = [
        WIDERDataset('datasets/WIDER/MTCNN_face_crops/', 'rnet', 'proximity/negative', data_transforms),
        WIDERDataset('datasets/WIDER/MTCNN_face_crops/', 'rnet', 'proximity/positive', data_transforms)
    ]

    cd = ConcatDataset(*datasets)

    np.random.seed(14)
    for i in range(len(cd)):
        (image, y) = cd[np.random.randint(0,len(cd)-1)]

        print(i,image.shape, y)

        #s, bb = onet(image[None,:])
        #print(bb)

        ax = plt.subplot(1, 4, i % 4 + 1)
        image = image.numpy()
        image = np.moveaxis(image, 0, 2)
        plt.imshow(image)
        plt.tight_layout()
        ax.set_title(f'IOU: {y[3]:.2f}')
        ax.axis('off')



        if i % 4 == 3:
            plt.show()
