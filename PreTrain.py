import time
from sys import exception
import logging
import Config
import torch
import YOLO
from Dataset_ILSVRC import Dataset
import Loss
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import torch.nn as nn
import torch
import signal
import sys

def save_model(model, optimizer, epoch, path=f"YOLO_PreTrain_{time.time()}.pth"):
    # 将模型、优化器状态和当前epoch保存到文件中
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)
    print(f"\n模型在第 {epoch} 轮时保存到 {path}")
def signal_handler(sig, frame):
    print("\n训练被中止，保存模型...")
    save_model(model, optimizer, epoch)
    sys.exit(0)


if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    logging.basicConfig(filename='grad.log', level=logging.INFO)
    cfg = Config.Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = Dataset(cfg)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=dataset.collate_fn)
    model = YOLO.PreTrainYOLO(cfg).to(device)
    saved_model_path = cfg.saved_model_path

    criterion = Loss.PreTrainLoss(cfg).to(device)
    num_epochs = 1000
    # optimizer = optim.SGD(model.parameters(), lr=5e-6, momentum=0.9, weight_decay=0.0005)
    optimizer = optim.Adam(model.parameters(), lr=5e-6)
    loss_total_last = 0.0
    from Image import Image
    from NMS import NMS
    img = Image(cfg)

    if saved_model_path is not None:
        print(f"loading models on {saved_model_path}")
        try:
            model.load_state_dict(torch.load(saved_model_path,weights_only=True))
        except Exception as e:
            print(f"loading models saved on Check Point")
            checkpoint = torch.load(saved_model_path,weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
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

    print("START")
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
                print(f'Epoch [{epoch + 1}/135] Batch [{batch_id}/34025], Average Loss: {loss_total:.4f}')
                loss_total = 0.0
            optimizer.step()
            optimizer.zero_grad()
        print(f'Epoch [{epoch + 1}/135], Loss: {loss_total:.4f}')
        torch.save(model.state_dict(), f"YOLO_PreTrain_{epoch}_{time.time()}.pth")