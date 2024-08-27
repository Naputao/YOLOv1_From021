import time

import Config
import torch
import YOLO
from Dataset_VOC2012 import Dataset
import Loss
import torch.optim as optim
from torch.utils.data import DataLoader
import os

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
    criterion = Loss.Loss(cfg).to(device)
    num_epochs = 1
    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.3, weight_decay=0.0005)
    loss_total_last = 0.0

    for epoch in range(num_epochs):
        loss_total = 0.0
        for batch_id,(data_batch, target) in enumerate(dataloader):
            output = model(data_batch.to(device))
            loss = criterion(output, target)
            loss.backward()
            loss_total += loss.item()
            optimizer.step()
            optimizer.zero_grad()
            print(f'Epoch [{epoch+1}/135] Batch [{batch_id}/1070], Average Loss: {loss.item():.4f}')
        print(f'Epoch [{epoch + 1}/135], Loss: {loss_total:.4f}')
        torch.save(model.state_dict(), f"YOLO_{epoch}_{time.time()}.pth")