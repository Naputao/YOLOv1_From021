import NMS
from Dataset_VOC2012 import Dataset
from torch.utils.data import DataLoader
import torch
import Config
from YOLO import YOLO
from Image import Image
from Loss import Loss
cfg = Config.Config()
dataset = Dataset(cfg)
image = Image(cfg)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=dataset.collate_fn)
model = YOLO(cfg)
device = torch.device('cpu')
model.to(device)
model.eval()
criterion = Loss(cfg).to(device)
saved_model_path = cfg.saved_model_path
nms = NMS.NMS(cfg).filter_max_confident
if saved_model_path is not None:
    print(f"loading models on {saved_model_path}")
    model.load_state_dict(torch.load(saved_model_path, weights_only=True))
# torch.set_printoptions(threshold=torch.inf)
with open('model_weights_on_nan.txt', 'w') as f:
    for name, param in model.named_parameters():
        f.write(f"{name}:/n{param.data}/n/n")
from PIL import Image
with open("C:/Users/Naputao/Desktop/8DA223856C5793905AEB33A22FF2D28C.jpg", "rb") as img_file:
    img = Image.open(img_file)
    img = cfg.VOC2012_transform(img)
    img = img.unsqueeze(0)
    output = model(img.to(device))
    with open("tensor.log", 'w') as f:
        f.write(str(output))
    image.show_with_detection(img,output)