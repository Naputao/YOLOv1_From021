import Config
import torch
import YOLO
import Dataset
import Loss
import torch.optim as optim
from torch.utils.data import DataLoader
cfg = Config.Config()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = Dataset.Dataset(cfg)
dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, collate_fn=Dataset.collate_fn)
model = YOLO.YOLO(cfg).to(device)
criterion = Loss.Loss(cfg).to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
num_epochs = 100
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, epochs = num_epochs, steps_per_epoch=10,pct_start=0.5)
for epoch in range(num_epochs):
    count = 0
    loss_total = 0.0
    for data_batch, target in dataloader:
        output = model(data_batch.to(device))
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_total += loss.item()

        print(f'Epoch [{epoch + 1}/100], Loss: {loss.item():.4f}')
        count += 1
        if count > 10:
            print(output[0,0,0])
            break


    print(f'Epoch [{epoch + 1}/100], Loss: {loss_total:.4f}')
    if epoch % 20 == 0:
        torch.save(model.state_dict(), f"model{epoch//20}.pth")