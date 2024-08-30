import torch
import torch.nn as nn
class YOLO(nn.Module):
    def __init__(self, config):
        self.config = config

        super(YOLO, self).__init__()
        self.backbone = BackBoneYOLO(config).conv_layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1),nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),nn.LeakyReLU(0.1),
        )
        self.flatten = nn.Flatten()
        self.conn_layers = nn.Sequential(
            nn.Linear(7 * 7 * 1024, 4096),nn.LeakyReLU(0.1),
            nn.Linear(4096, 7 * 7 * (5*self.config.bounding_boxes+self.config.clazz))
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.conn_layers(x)
        return x.view(-1,self.config.grid,self.config.grid,self.config.clazz + self.config.bounding_boxes * 5)
class PreTrainYOLO(nn.Module):
    def __init__(self, config):
        self.config = config
        super(PreTrainYOLO, self).__init__()
        self.backbone = BackBoneYOLO(config).conv_layers
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(1024, 1000)
    def forward(self, x):
        x = self.backbone(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class BackBoneYOLO(nn.Module):
    def __init__(self, config):
        self.config = config
        super(BackBoneYOLO, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),nn.LeakyReLU(0.1),
            nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),nn.LeakyReLU(0.1),
            nn.Conv2d(192, 128, kernel_size=1, stride=1, padding=0),nn.LeakyReLU(0.1),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),nn.LeakyReLU(0.1),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),nn.LeakyReLU(0.1),
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),nn.LeakyReLU(0.1),
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),nn.LeakyReLU(0.1),
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),nn.LeakyReLU(0.1),
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),nn.LeakyReLU(0.1),
            nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),nn.LeakyReLU(0.1),
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0),nn.LeakyReLU(0.1),
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0),nn.LeakyReLU(0.1),
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),nn.LeakyReLU(0.1),
        )
    def forward(self, x):
        return self.conv_layers(x)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    import Config
    cfg = Config.Config()
    model = PreTrainYOLO(cfg)
    model.to(device)
    input_tensor = torch.randn(32, 3, 224, 224)
    input_tensor = input_tensor.to(device)
    output = model.forward(input_tensor)
    print(output.shape)
