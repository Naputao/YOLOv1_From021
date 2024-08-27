import torch
import torch.nn as nn
class ALLZERO(nn.Module):
    def __init__(self, config):
        self.config = config

        super(ALLZERO, self).__init__()
    def forward(self, x):
        return torch.zeros(x.shape[0],7,7,30,device=self.config.device)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    import Config
    cfg = Config.Config()
    model = ALLZERO(cfg)
    model.to(device)
    input_tensor = torch.randn(32, 3, 448, 448)
    input_tensor = input_tensor.to(device)
    output = model(input_tensor)
    print(output.shape)
