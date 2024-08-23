import Config
import torch
import YOLO
import Dataset
import Loss
import torch.optim as optim
from torch.utils.data import DataLoader
import time
if __name__ == '__main__':
    cfg = Config.Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = Dataset.Dataset(cfg)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, collate_fn=dataset.collate_fn)
    model = YOLO.YOLO(cfg).to(device)
    criterion = Loss.Loss(cfg).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 100
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
            print(f'Epoch [{epoch+1}/100] Batch [{batch_id}/200], Loss: {loss.item()/cfg.batch_size:.4f}')
            if batch_id >9:
                print(output[0,4,4])
                break
        optimizer.step()
        optimizer.zero_grad()
        print(f'Epoch [{epoch + 1}/100], Loss: {loss_total:.4f}')
        if epoch % 20 == 0 or loss_total_last == loss_total:
            torch.save(model.state_dict(), f"model{epoch}.pth")