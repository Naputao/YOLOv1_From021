import os
import zipfile
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import torchvision.transforms as transforms

from Config import Config

class ImageNetDataset(Dataset):
    def __init__(self, zip_file, folder_name, transform=None):
        self.zip_file = zip_file
        self.folder_name = folder_name
        self.transform = transform
        self.file_list = [f for f in self.zip_file.namelist() if f.startswith(folder_name) and f.endswith('.JPEG')]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        with self.zip_file.open(img_path) as img_file:
            img = Image.open(img_file).convert('RGB')
            if self.transform:
                img = self.transform(img)
        return img

if __name__ == '__main__':
    # 定义数据转换
    cfg = Config()
    transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
    ])

    # 打开ZIP文件
    zip_path = cfg.zip_dataset_path
    with zipfile.ZipFile(zip_path, 'r') as zip_file:

        # 获取train目录下的所有子文件夹
        train_dir = 'ILSVRC/Data/CLS-LOC/train'
        subfolders = [f for f in zip_file.namelist() if f.startswith(train_dir) and f.count('/') == 5]

        # 手动控制加载和训练
        for folder in subfolders:
            folder_name = folder.split('/')[-2]
            print(f"Training on folder: {folder_name}")

            # 创建数据集和数据加载器
            dataset = ImageNetDataset(zip_file, folder, transform=transform)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

            # 模型训练
            for images in dataloader:
                # 在这里执行训练步骤
                # 如 model(images)
                print(f"Processed batch from {folder_name} with {len(images)} images.")

            # 模拟手动切换到下一个文件夹
            input("Press Enter to proceed to the next folder...")

