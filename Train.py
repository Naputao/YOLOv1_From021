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
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    cfg = Config.Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = Dataset(cfg)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=dataset.collate_fn)
    model = YOLO.YOLO(cfg).to(device)
    saved_model_path = cfg.saved_model_path
    if saved_model_path is not None:
        print(f"loading models on {saved_model_path}")
        model.load_state_dict(torch.load(saved_model_path,weights_only=True))
    else:
        # 定义一个初始化函数
        def initialize_weights(m):
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
        # 对模型应用初始化
        print("initializing...")
        model.apply(initialize_weights)
        print("done")
    criterion = Loss.Loss(cfg).to(device)
    num_epochs = 3
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
    optimizer = optim.Adam(model.parameters(), lr=0.0003)
    loss_total_last = 0.0
    from Image import Image
    from NMS import NMS
    img = Image(cfg)
    nms = NMS(cfg).filter_max_confident
    for epoch in range(num_epochs):
        loss_total = 0.0
        for batch_id,(data_batch, target) in enumerate(dataloader):
            output = model(data_batch.to(device))
            loss = criterion(output, target)
            loss.backward()
            loss_total += loss.item()
            if batch_id%4==0:
                optimizer.step()
                optimizer.zero_grad()
            print(f'Epoch [{epoch+1}/135] Batch [{batch_id}/1070], Average Loss: {loss.item():.4f}')
            if batch_id%50==0:
                img.show_with_annotation_and_detection_no_filter(data_batch[0].unsqueeze(0),target[target[...,-1].int()==0],output[0].unsqueeze(0))
        print(f'Epoch [{epoch + 1}/135], Loss: {loss_total:.4f}')
        torch.save(model.state_dict(), f"YOLO_{epoch}_{time.time()}.pth")