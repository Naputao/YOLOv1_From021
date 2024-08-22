import torch
import torch.nn as nn
class YOLO(nn.Module):
    def __init__(self, config):
        self.config = config

        super(YOLO, self).__init__()
        self.conv_layers_1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LeakyReLU(negative_slope=0.01),
        )
        self.conv_layers_2 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LeakyReLU(negative_slope=0.01),
        )
        self.conv_layers_3 = nn.Sequential(
            nn.Conv2d(192, 128, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LeakyReLU(negative_slope=0.01),
        )
        self.conv_layers_4 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LeakyReLU(negative_slope=0.01),
        )
        self.conv_layers_5 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
        )
        self.conv_layers_6 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
        )
        self.flatten = nn.Flatten()
        self.conn_layers_7 = nn.Sequential(
            nn.Linear(7 * 7 * 1024, 4096),
            nn.LeakyReLU(negative_slope=0.01),
        )
        self.conn_layers_8 = nn.Sequential(
            nn.Linear(4096, 7 * 7 * (5*self.config.bounding_boxes+self.config.clazz)),
            nn.LeakyReLU(negative_slope=0.01),
        )

    def forward(self, x):
        x = self.conv_layers_1(x)
        x = self.conv_layers_2(x)
        x = self.conv_layers_3(x)
        x = self.conv_layers_4(x)
        x = self.conv_layers_5(x)
        x = self.conv_layers_6(x)
        x = self.flatten(x)
        x = self.conn_layers_7(x)
        x = self.conn_layers_8(x)
        x = (torch.tanh(x)+1)/2
        return x.view(-1,self.config.grid,self.config.grid,self.config.clazz + self.config.bounding_boxes * 5)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    import Config
    cfg = Config.Config()
    model = YOLO(cfg)
    model.to(device)
    input_tensor = torch.randn(32, 3, 448, 448)
    input_tensor = input_tensor.to(device)
    output = model(input_tensor)
    print(output.shape)
