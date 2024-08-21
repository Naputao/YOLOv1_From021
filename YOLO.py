import torch
import torch.nn as nn

from Grid import Grid


class YOLO(nn.Module):
    def __init__(self, grid=7, bounding_boxes=2, clazz=20):
        self.clazz = clazz
        self.bounding_boxes = bounding_boxes
        self.grid = grid

        super(YOLO, self).__init__()
        self.conv_layers_1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_layers_2 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_layers_3 = nn.Sequential(
            nn.Conv2d(192, 128, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
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
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_layers_5 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1)
        )
        self.conv_layers_6 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1)
        )
        self.flatten = nn.Flatten()
        self.conn_layers_7 = nn.Sequential(
            nn.Linear(7 * 7 * 1024, 4096)
        )
        self.conn_layers_8 = nn.Sequential(
            nn.Linear(4096, 7 * 7 * (5*self.bounding_boxes+self.clazz))
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
        return x


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = YOLO()
    model.to(device)
    input_tensor = torch.randn(1, 3, 448, 448)
    input_tensor = input_tensor.to(device)
    output = model(input_tensor)
    print(output.shape)
