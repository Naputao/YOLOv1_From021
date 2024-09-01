import os
import torchvision.transforms as transforms
import torch
from torch.cuda import device


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
        with open('LOC_synset_mapping.txt', 'r') as file:
            loc_synset_mapping = {}
            for line_number, line in enumerate(file):
                id_part = line.split()[0]
                loc_synset_mapping[id_part] = line_number
        self.loc_synset_mapping = loc_synset_mapping

        #Loss config
        self.lambda_confidence = 16
        self.lamda_coord = 5
        self.lamda_noobj = 0.5
        self.lamda_obj = 1
        self.lamda_size = 5
        # NMS config
        self.minimum_confidence = 0.3
        self.maximum_iou = 0.2
        #dataset config
        self.cwd_path = os.getcwd()
        # self.zip_dataset_path = "E:/Downloads/imagenet-object-localization-challenge.zip"
        self.ILSVRC_dateset_path = "ILSVRC"
        self.ILSVRC_annotations_path = self.ILSVRC_dateset_path + "/Annotations/CLS-LOC"
        self.ILSVRC_images_path = self.ILSVRC_dateset_path + "/Data/CLS-LOC"
        self.ILSVRC_train_images_path = self.ILSVRC_images_path + "/train"
        self.ILSVRC_val_images_path = self.ILSVRC_images_path + "/val"
        # self.test_images_path = self.images_path + "/test"
        # self.val_images_path = self.images_path + "/val"
        self.ILSVRC_train_annotations_path = self.ILSVRC_annotations_path + "/train"
        self.ILSVRC_val_annotations_path = self.ILSVRC_annotations_path + "/val"
        # self.val_annotations_path = self.annotations_path + "/val"
        # self.current_annotations_path = self.train_annotations_path + "/n01440764/n01440764_10040.xml"
        self.current_images_path = "E:/WorkDir/YOLOv1_From021/Dataset/VOC2012/JPEGImages/2007_000032.jpg"

        self.VOC2012_dataset_path = "E:/Downloads/VOCtrainval_11-May-2012.tar"
        self.ILSVRC_dataset_path = "E:/Downloads/imagenet-object-localization-challenge.zip"
        self.dateset_path = "VOCdevkit/VOC2012"
        self.annotations_path = self.dateset_path + "/Annotations/"
        self.images_path = self.dateset_path + "/JPEGImages/"
        self.current_annotations_path = "E:/WorkDir/YOLOv1_From021/Dataset/VOC2012/Annotations/2007_000032.xml"
        self.VOC2012_transform = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor()
        ])
        self.ILSVRC_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.batch_size = 16
        # self.saved_model_path = "YOLO_0_1724736447.720819.pth"
        self.saved_model_path = "YOLO_9_1725199940.5631301.pth"
        self.device = torch.device('cuda')
if __name__ == '__main__':
    import Image
    cfg = Config()
    print(cfg.train_annotations_path)