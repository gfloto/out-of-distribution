import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

# converts black and white images to color by stacking
class Grey2Color:
    def __call__(self, img):
        img = torch.vstack((img, img, img)) 
        return img

# constant random data
class FakeData(Dataset):
    def __init__(self, dataset_type):
        assert dataset_type in ['constant', 'random']
        self.dataset_type = dataset_type
        self.len = int(1e5)

    def __len__(self):
        return self.len

    def __getitem__(self, _):
        if self.dataset_type == 'constant':
            f = torch.rand(3, 1, 1)
            x = f * torch.ones(3, 32, 32)
        elif self.dataset_type == 'random':
            x = torch.rand((3, 32, 32))
        else: raise NotImplementedError

        return x, torch.zeros(1)

# list of all datasets 
def all_datasets():
    datasets = [
        'constant', 'random',
        'mnist', 'fmnist', 'kmnist',
        'svhn', 'lsun', 'celeba', 
        'cifar100', 'imagenet',
        'cifar10',
    ]
    return datasets

def get_loader(path, dataset_name, split, batch_size):
    assert split in ['train', 'test']
    assert dataset_name in all_datasets() 

    train = split == 'train'

    # resize everything to 32x32
    colour_transforms = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
    ])

    # if grey, convert to color by stacking channels
    grey_transforms = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        Grey2Color(),
    ]) 

    # get dataset, assumes that '__main__' was run to download images already
    if dataset_name == 'mnist':
        dataset = datasets.MNIST(root=path, train=train, download=True, transform=grey_transforms)
    elif dataset_name == 'fmnist':
        dataset = datasets.FashionMNIST(root=path, train=train, download=True, transform=grey_transforms)
    elif dataset_name == 'kmnist':
        dataset = datasets.KMNIST(root=path, train=train, download=True, transform=grey_transforms)
    elif dataset_name == 'svhn':
        dataset = datasets.SVHN(root=path, split=split, download=True, transform=colour_transforms)
    elif dataset_name == 'cifar10':
        dataset = datasets.CIFAR10(root=path, train=train, download=True, transform=colour_transforms)
    elif dataset_name == 'cifar100':
        dataset = datasets.CIFAR100(root=path, train=train, download=True, transform=colour_transforms)
    elif dataset_name == 'celeba':
        dataset = datasets.CelebA(root=path, split=split, download=True, transform=colour_transforms)
    elif dataset_name == 'lsun':
        split = 'bedroom_train' #if train else 'bedroom_val'
        path = '/drive2/ood/lsun/'
        dataset = datasets.LSUN(root=path, classes=[split], transform=colour_transforms)
    elif dataset_name == 'imagenet':
        split = 'train'
        dataset = datasets.ImageNet(root=path, split=split, transform=colour_transforms)
    elif dataset_name in ['constant', 'random']:
        dataset = FakeData(dataset_type=dataset_name)
    else: raise NotImplementedError

    # get loader
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    return loader


# latent space datasets
# ---------------------------- #
import os 
import json 
from tqdm import tqdm
from argparse import Namespace

from models.autoenc import get_autoenc

# save latent space representations of all datasets
@torch.no_grad()
def z_store(model, loader, device):
    z_all = None; recon_all = None
    for i, (x, _) in enumerate(tqdm(loader)):
        if i == 24: break
        x = x.to(device)

        _, mu, x_out = model(x)
        recon = (x - x_out).square().mean(dim=(1,2,3))

        if z_all is None:
            z_all = mu
            recon_all = recon
        else:
            z_all = torch.cat((z_all, mu), dim=0)
            recon_all = torch.cat((recon_all, recon), dim=0)
    
    return z_all, recon_all

# given experiment name, save latent space representations of all datasets
def save_latents(name, train_dataset, device):
    batch_size = 2048

    # make dir of compressed representations
    save_path = os.path.join(name, 'lat')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # get args
    with open(os.path.join(name, 'args.json'), 'r') as f:
        args = json.load(f)
        args = Namespace(**args)

    # load model
    model = get_autoenc(args).to(args.device)
    model.load_state_dict(torch.load(os.path.join(name, 'model.pt')))

    datasets = all_datasets()
    for i, dataset in enumerate(datasets):
        if dataset == train_dataset: 
            modes = ['train', 'test']
        else: 
            modes = ['test']

        for mode in modes:
            loader = get_loader(args.data_path, dataset, mode, batch_size)
            z, recon = z_store(model, loader, device)

            # save z
            torch.save(z, os.path.join(save_path, f'{dataset}_{mode}_z.pt'))
            torch.save(recon, os.path.join(save_path, f'{dataset}_{mode}_recon.pt'))

def get_latents(path, dataset, mode):
    assert mode in ['train', 'test']

    z = torch.load(os.path.join(path, 'lat', f'{dataset}_{mode}_z.pt'))
    return z

import matplotlib.pyplot as plt

if __name__ == '__main__':
    path = '/drive2/ood/'

    names = ['mnist', 'fmnist', 'kmnist', 'svhn', 'cifar10', 'cifar100', 'celeba', 'constant', 'random', 'imagenet']
    for name in names:
        # plot some images
        loader = get_loader(path, name, 'train', 2, 1)
        for x, _ in loader:
            print(x.shape)
            plt.imshow(x[0].permute(1, 2, 0))
            plt.title(name)
            plt.show()
            break

