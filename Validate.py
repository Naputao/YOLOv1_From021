import torch
from torch import nn

from Config import Config
from Dataset_ILSVRC_Val import Dataset
from YOLO import PreTrainYOLO
import torch.nn.functional as F
if __name__ == '__main__':
    cfg = Config()
    dataset=Dataset(cfg)
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=dataset.collate_fn)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PreTrainYOLO(cfg).to(device)
    saved_model_path = cfg.saved_model_path
    if saved_model_path is not None:
        print(f"loading models on {saved_model_path}")
        try:
            model.load_state_dict(torch.load(saved_model_path,weights_only=True))
        except Exception as e:
            print(f"loading models saved on Check Point")
            checkpoint = torch.load(saved_model_path,weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint['epoch']
    else:
        def initialize_weights(m):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
        print("initializing...")
        model.apply(initialize_weights)
        print("done")
    true_positive = 0
    total = 0
    for batch_id,(data_batch, labels_batch) in enumerate(dataloader):
        print(f"batch_id:{batch_id}")
        output = model(data_batch.to(device))
        output = F.softmax(output, dim=1)
        target = labels_batch.to(device)
        top5_values, top5_indices = torch.topk(output[target[...,1]], 5,dim=1)
        true_positive += (top5_indices == target[..., 0].unsqueeze(1)).any(dim=1).sum()
        total += target.shape[0]
    print(true_positive/total)