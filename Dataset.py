import zipfile
import Config
import xml.etree.ElementTree as ET
from PIL import Image
import torch
from Annotation import Annotation




class Dataset:
    def __init__(self, cfg):
        self.cfg = cfg
        self.file=zipfile.ZipFile(cfg.zip_dataset_path, 'r')
        self.file_list = list({f[33:-4] for f in self.file.namelist() if f.startswith(cfg.train_annotations_path)} &
                          {f[26:-5] for f in self.file.namelist() if f.startswith(cfg.train_images_path)})
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.cfg.train_images_path+"/"+self.file_list[idx]+".JPEG"
        target_path = self.cfg.train_annotations_path+"/"+self.file_list[idx]+".xml"
        # print(f"loading {self.file_list[idx]}")
        with self.file.open(img_path) as img_file:
            img = self.cfg.transform(Image.open(img_file).convert('RGB')).to(self.cfg.device)
        with self.file.open(target_path) as xml_file:
            xml = Annotation(ET.parse(xml_file),self.cfg).to_list()
        return img,xml

    def collate_fn(self,batch):
        data_batch = torch.stack([item[0] for item in batch])
        labels_batch = torch.stack(
            [torch.cat((ts, torch.tensor([item_id], device=self.cfg.device))) for item_id, item in enumerate(batch) for ts in
             item[1]])
        return data_batch, labels_batch

if __name__ == '__main__':
    config = Config.Config()
    dataset=Dataset(config)
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=dataset.collate_fn)
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
