import os
import torchvision.transforms as transforms
import torch
class Config:
    def __init__(self):
        #model config
        self.grid=7
        self.bounding_boxes=2
        self.clazz=20
        self.grid_length = self.bounding_boxes * 5 + self.clazz

        self.input_width = 448
        self.input_height = 448
        self.grid_width = self.input_height // self.grid
        self.grid_height = self.input_height // self.grid

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #Loss config
        self.lamda_coord = 5
        self.lamda_noobj = 0.5

        #dataset config
        self.zip_dataset_path = "E:/Download/imagenet-object-localization-challenge.zip"
        self.cwd_path = os.getcwd()
        self.dateset_path = "ILSVRC"
        self.annotations_path = self.dateset_path + "/Annotations/CLS-LOC"
        self.images_path = self.dateset_path + "/Data/CLS-LOC"
        self.train_images_path = self.images_path + "/train"
        self.test_images_path = self.images_path + "/test"
        self.val_images_path = self.images_path + "/val"
        self.train_annotations_path = self.annotations_path + "/train"
        self.val_annotations_path = self.annotations_path + "/val"

        self.current_annotations_path = self.train_annotations_path + "/n01440764/n01440764_10040.xml"
        self.current_images_path = self.train_images_path + "/n01440764/n01440764_10040.JPEG"

        self.transform = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor()
        ])
        self.batch_size = 16

if __name__ == '__main__':
    import Image
    cfg = Config()
    print(cfg.train_annotations_path)