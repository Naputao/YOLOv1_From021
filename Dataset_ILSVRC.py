import zipfile
import Config
import xml.etree.ElementTree as ET
from PIL import Image
import torch
from Annotation_ILSVRC import Annotation

class Dataset:
    def __init__(self, cfg):
        self.cfg = cfg
        self.file=zipfile.ZipFile(cfg.ILSVRC_dataset_path, 'r')
        self.file_list = [f[33:-4] for f in self.file.namelist() if f.startswith(cfg.ILSVRC_train_annotations_path)]
        with open('LOC_synset_mapping.txt', 'r') as file:
            loc_synset_mapping = {}
            for line_number, line in enumerate(file):
                id_part = line.split()[0]
                loc_synset_mapping[id_part] = line_number
        self.loc_synset_mapping = loc_synset_mapping
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        img_path = self.cfg.ILSVRC_train_images_path+"/"+file_path+".JPEG"
        with self.file.open(img_path) as img_file:
            img = self.cfg.transform(Image.open(img_file).convert('RGB')).to(self.cfg.device)
        target = self.loc_synset_mapping[file_path[0:9]]
        return img,target

    def collate_fn(self,batch):
        data_batch = torch.stack([item[0] for item in batch])
        labels_batch = torch.stack([torch.tensor(item[1]) for item in batch])
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
