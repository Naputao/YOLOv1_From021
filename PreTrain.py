import time

import Config
import torch
import YOLO
from Dataset_ILSVRC import Dataset
import Loss
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import torch.nn as nn

if __name__ == '__main__':
    cfg = Config.Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = Dataset(cfg)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=dataset.collate_fn)
    model = YOLO.PreTrainYOLO(cfg).to(device)
    saved_model_path = cfg.saved_model_path
    if saved_model_path is not None:
        print(f"loading models on {saved_model_path}")
        model.load_state_dict(torch.load(saved_model_path,weights_only=True))
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
    criterion = Loss.PreTrainLoss(cfg).to(device)
    num_epochs = 1000
    # optimizer = optim.SGD(model.parameters(), lr=5e-6, momentum=0.1, weight_decay=0.0005)
    optimizer = optim.Adam(model.parameters(), lr=5e-6)
    loss_total_last = 0.0
    from Image import Image
    from NMS import NMS
    img = Image(cfg)
    for epoch in range(num_epochs):
        loss_total = 0.0
        for batch_id,(data_batch, target) in enumerate(dataloader):
            output = model(data_batch.to(device))
            if torch.isnan(output).any():
                print(f"batch_id {batch_id}: Output is NaN. Pausing training.")
                with open('model_weights_on_nan.txt', 'w') as f:
                    for name, param in model.named_parameters():
                        f.write(f"{name}:\n{param.data}\n\n")
                raise ValueError("Output contains NaN values.")
            loss = criterion(output, target)

            loss.backward()
            loss_total += loss.item()
            optimizer.step()
            optimizer.zero_grad()
            if batch_id%25==0:
                print(f'Epoch [{epoch + 1}/135] Batch [{batch_id}/???], Average Loss: {loss.item():.4f}')
        print(f'Epoch [{epoch + 1}/135], Loss: {loss_total:.4f}')
        torch.save(model.state_dict(), f"YOLO_PreTrain_{epoch}_{time.time()}.pth")