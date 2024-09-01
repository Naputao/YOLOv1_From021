import logging
import time

import Config
import torch
import YOLO
from Dataset_VOC2012 import Dataset
import Loss
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import torch.nn as nn
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
def adjust_learning_rate(optimizer, epoch):
    warmup_epochs, target_lr, initial_lr = 5,1e-3,1e-5
    if epoch < warmup_epochs:
        lr = initial_lr + (target_lr - initial_lr) * (epoch / warmup_epochs)
    elif epoch < 50:
        lr = target_lr
    elif epoch < 105:
        lr = 1e-3
    else:
        lr = 1e-4
    for param_group in optimizer.param_groups:
        param_group['lr'] = 0.5*lr
if __name__ == '__main__':
    logging.basicConfig(filename='grad.log', level=logging.INFO)
    cfg = Config.Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = Dataset(cfg)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=dataset.collate_fn)
    model = YOLO.YOLO(cfg).to(device)
    saved_model_path = cfg.saved_model_path
    if saved_model_path is not None:
        print(f"loading models on {saved_model_path}")
        try:
            model.load_state_dict(torch.load(saved_model_path,weights_only=True))
            print("freezing...")
            for name, param in model.backbone.named_parameters():
                if name not in ['40.weight','40.bias',
                            '42.weight', '42.bias',
                            '44.weight', '44.bias',
                            '46.weight', '46.bias']:
                    param.requires_grad = False
            print("done")
        except Exception as e:
            print(f"loading backbone_models on {saved_model_path}")
            pretrain_model = YOLO.PreTrainYOLO(cfg)
            pretrain_model.load_state_dict(torch.load(saved_model_path))
            model.backbone.load_state_dict(pretrain_model.backbone.state_dict())
            print("initializing...")
            model.conv_layers.apply(initialize_weights)
            model.conn_layers.apply(initialize_weights)
            print("done")
            print("freezing...")
            for param in model.backbone.parameters():
                param.requires_grad = False
            print("done")
    else:
        print("initializing...")
        model.apply(initialize_weights)
        print("done")
    criterion = Loss.Loss(cfg).to(device)
    num_epochs = 135
    # optimizer = optim.SGD(model.parameters(), lr=5e-6, momentum=0.1, weight_decay=0.0005)
    optimizer = optim.Adam(model.parameters(), lr=1e-7)
    loss_total_last = 0.0
    from Image import Image
    from NMS import NMS
    img = Image(cfg)
    nms = NMS(cfg).filter_max_confident
    print("START:")
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
            if batch_id%100==0:
                logging.info(f'\n\n========Epoch [{epoch + 1}/135] Batch [{batch_id}/34025], Average Loss: {loss_total:.4f}========')
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        logging.info(f"参数: {name}, 梯度大小: {param.grad.norm().item()}")
                    else:
                        logging.info(f"参数: {name}, 梯度大小: None")
                img.show_with_annotation_and_detection(data_batch[0].unsqueeze(0),target[target[...,-1].int()==0],output[0].unsqueeze(0),nms)
                # img.show_with_annotation_and_detection(data_batch, target,output, nms)
                print(f'Epoch [{epoch + 1}/135] Batch [{batch_id}/1070], Average Loss: {loss_total:.4f}')
                loss_total = 0.0

            optimizer.step()
            optimizer.zero_grad()
        print(f'Epoch [{epoch + 1}/135], Loss: {loss_total:.4f}')
        torch.save(model.state_dict(), f"YOLO_{epoch}_{time.time()}.pth")