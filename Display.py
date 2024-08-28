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
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=dataset.collate_fn)
model = YOLO(cfg)
device = torch.device('cuda')
model.to(device)
criterion = Loss(cfg).to(device)
saved_model_path = cfg.saved_model_path
if saved_model_path is not None:
    print(f"loading models on {saved_model_path}")
    model.load_state_dict(torch.load(saved_model_path, weights_only=True))
torch.set_printoptions(threshold=torch.inf)
for batch_id, (data_batch, target) in enumerate(dataloader):
    output = model(data_batch)
    with torch.no_grad():
        print(criterion(output, target).item())
    with open("tensor.log", 'w') as f:
        f.write(str(output))
    image.show_with_annotation_and_detection(data_batch, target, output,NMS.NMS(cfg).filter_max_confident)
    break