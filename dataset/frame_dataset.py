import os
import cv2
import torch
import random
import torchvision.transforms as transforms
from torch.utils.data import Dataset

class FrameDataset(Dataset):
    """
    Cada item: (f1, f3) => concat(3+3=6) => input_6c,
    target => f2.
    """
    def __init__(self, triple_paths, root_dir, device):
        super().__init__()
        self.triple_paths = triple_paths
        self.root_dir = root_dir
        self.device = device
        self.to_tensor = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ])

    def __len__(self):
        return len(self.triple_paths)

    def __getitem__(self, idx):
        folder_str = self.triple_paths[idx]
        f1_path = os.path.join(self.root_dir, folder_str, "im1.png")
        f2_path = os.path.join(self.root_dir, folder_str, "im2.png")
        f3_path = os.path.join(self.root_dir, folder_str, "im3.png")

        frame1 = cv2.imread(f1_path)
        frame2 = cv2.imread(f2_path)
        frame3 = cv2.imread(f3_path)
        
        if frame1 is None or frame2 is None or frame3 is None:
            raise ValueError(f"Erro ao carregar as imagens da pasta {folder_str}")

        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
        frame3 = cv2.cvtColor(frame3, cv2.COLOR_BGR2RGB)

        t1 = self.to_tensor(frame1)
        t2 = self.to_tensor(frame2)
        t3 = self.to_tensor(frame3)

        input_6c = torch.cat([t1, t3], dim=0)

        return input_6c, t2, t1, t3

class AugmentWrapper(Dataset):
    """
    Recebe um dataset base e aplica RandomCrop + RandomHorizontalFlip
    em cada item.
    """
    def __init__(self, base_dataset, crop_size=256):
        super().__init__()
        self.base_dataset = base_dataset
        self.crop_size = crop_size

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        input_6c, t2, t1, t3 = self.base_dataset[idx]
        _, H, W = input_6c.shape
        if H >= self.crop_size and W >= self.crop_size:
            top = random.randint(0, H - self.crop_size)
            left = random.randint(0, W - self.crop_size)
            input_6c = input_6c[:, top:top+self.crop_size, left:left+self.crop_size]
            t2       = t2[:,       top:top+self.crop_size, left:left+self.crop_size]
            t1       = t1[:,       top:top+self.crop_size, left:left+self.crop_size]
            t3       = t3[:,       top:top+self.crop_size, left:left+self.crop_size]

        if random.random() < 0.5:
            input_6c = torch.flip(input_6c, dims=[2])
            t2       = torch.flip(t2,       dims=[2])
            t1       = torch.flip(t1,       dims=[2])
            t3       = torch.flip(t3,       dims=[2])

        return input_6c, t2, t1, t3
