import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
import lightning as pl

class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = r'/home/muthu/GitHub/DATA 📁/CIFAR', batch_size: int = 32, num_workers: int = 4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Define data transforms for train, validation and test
        self.transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.Resize((32,32)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])

        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])

    def prepare_data(self):
        # Download CIFAR-10 dataset
        datasets.CIFAR10(root=self.data_dir, train=True, download=True)
        datasets.CIFAR10(root=self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # Split dataset for training, validation and test
        if stage == 'fit' or stage is None:
            full_train_dataset = datasets.CIFAR10(root=self.data_dir, train=True, transform=self.transform_train)
            self.train_dataset, self.val_dataset = random_split(full_train_dataset, [45000, 5000])

        if stage == 'test' or stage is None:
            self.test_dataset = datasets.CIFAR10(root=self.data_dir, train=False, transform=self.transform_test)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
