import torch
from torch.onnx.symbolic_opset9 import tensor
class Box:
    def __init__(self, tensor):
        assert isinstance(tensor, torch.Tensor), f"Not a Tensor: {type(tensor)}"
        assert tensor.shape[0] == 5 or tensor.shape[0] ==4 or tensor.shape[0]==7, f"Tensor shape is incorrect: {tensor.shape}"
        self.tensor = tensor

    @property
    def x(self):
        return self.tensor[0]

    @property
    def y(self):
        return self.tensor[1]

    @property
    def w(self):
        return self.tensor[2]

    @property
    def h(self):
        return self.tensor[3]

    @property
    def confidence(self):
        return self.tensor[4]

    def __str__(self):
        return (f"\n["
                f"x: {self.x}, "
                f"y: {self.y}, "
                f"w: {self.w}, "
                f"h: {self.h}, "
                f"confidence: {self.confidence}]")

    def __repr__(self):
        return self.__str__()
    def normalize(self,grid_x,grid_y,cfg):
        x = (self.tensor[0]-cfg.grid_width * grid_x)/cfg.grid_width
        y = (self.tensor[1]-cfg.grid_height * grid_y)/cfg.grid_height
        w = self.tensor[2]/cfg.input_width
        h = self.tensor[3]/cfg.input_height

        return torch.tensor([x,y,w,h])


def t52box(ts,grid_x,grid_y,cfg):
    box_center_x = cfg.grid_width * (ts[0] + grid_x)
    box_center_y = cfg.grid_height * (ts[1] + grid_y)
    box_width = cfg.input_width * ts[2]
    box_height = cfg.input_height * ts[3]
    return Box(torch.tensor([box_center_x, box_center_y, box_width, box_height,ts[4]]))