import Config
import xml.etree.ElementTree as ET
from Annotation_VOC2012 import Annotation
import torch
from PIL import Image

import tarfile
import io

class Dataset:
    def __init__(self, cfg):
        self.cfg = cfg
        self.file = tarfile.open(self.cfg.zip_dataset_path, 'r')
        self.file_list = sorted(list(
                        {member.name[-15:-4]
                        for member in self.file.getmembers()
                        if member.isfile()
                        and member.name.startswith(self.cfg.images_path)
                        and member.name.endswith(('png', 'jpg', 'jpeg'))}&
                        {member.name[-15:-4]
                        for member in self.file.getmembers()
                        if member.isfile()
                        and member.name.startswith(self.cfg.images_path)
                        and member.name.endswith(('png', 'jpg', 'jpeg'))}))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        img_file = self.file.extractfile(self.cfg.images_path +file_path+".jpg")
        input = self.cfg.transform(Image.open(io.BytesIO(img_file.read())).convert('RGB')).to(self.cfg.device)
        annotation_file = self.file.extractfile(self.cfg.annotations_path +file_path+".xml")
        target = Annotation(ET.parse(annotation_file),self.cfg).to_list()
        return input,target

    def collate_fn(self, batch):
        data_batch = torch.stack([item[0] for item in batch])
        labels_batch = torch.stack(
            [torch.cat((ts, torch.tensor([item_id], device=self.cfg.device))) for item_id, item in enumerate(batch) for
             ts in
             item[1]])
        return data_batch, labels_batch

if __name__ == '__main__':
    config = Config.Config()
    dataset = Dataset(config)
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=dataset.collate_fn)

    count = 0
    from YOLO import YOLO
    model = YOLO(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    for data_batch, labels_batch in dataloader:
        output = model.forward(data_batch.to(device))
        count += 1
        print(count, labels_batch)
    print(count)