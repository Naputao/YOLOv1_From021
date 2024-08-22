import zipfile
import Config
import xml.etree.ElementTree as ET
from PIL import Image
import torch
from Annotation import Annotation


def collate_fn(batch):
    data_batch = torch.stack([item[0] for item in batch])
    labels_batch = [item[1] for item in batch]  # 保持标签为列表
    return data_batch, labels_batch

class Dataset:
    def __init__(self, config):
        self.config = config
        self.file=zipfile.ZipFile(config.zip_dataset_path, 'r')
        self.file_list = list({f[33:-4] for f in self.file.namelist() if f.startswith(config.train_annotations_path)} &
                          {f[26:-5] for f in self.file.namelist() if f.startswith(config.train_images_path)})
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.config.train_images_path+"/"+self.file_list[idx]+".JPEG"
        target_path = self.config.train_annotations_path+"/"+self.file_list[idx]+".xml"
        # print(f"loading {self.file_list[idx]}")
        with self.file.open(img_path) as img_file:
            img = self.config.transform(Image.open(img_file).convert('RGB'))
        with self.file.open(target_path) as xml_file:
            xml = Annotation(ET.parse(xml_file),self.config).to_list()
        return img,xml


if __name__ == '__main__':
    config = Config.Config()
    dataset=Dataset(config)
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    count = 0
    # from YOLO import YOLO
    # model = YOLO(config)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model.to(device)
    for data_batch, labels_batch in dataloader:
        # output = model.forward(data_batch.to(device))
        count+=1
        print(count,labels_batch)
    print(count)
