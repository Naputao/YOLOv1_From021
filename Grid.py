import torch
import Box
from Box import t52box

class Grid:
    def __init__(self, tensor,grid_x,grid_y, cfg):
        assert isinstance(tensor, torch.Tensor), f"Not a Tensor: {type(tensor)}"
        assert tensor.shape[0] == 5 * cfg.bounding_boxes + cfg.clazz, f"Tensor shape is incorrect: {tensor.shape}"
        self.tensor = tensor
        self.config = cfg
        self.grid_x = grid_x
        self.grid_y = grid_y

    @property
    def detections(self):
        return [t52box(self.tensor[i:i + 5],self.grid_x,self.grid_y,self.config) for i in range(self.config.bounding_boxes)]

    @property
    def class_probabilities(self):
        return self.tensor[self.config.bounding_boxes * 5:self.config.bounding_boxes * 5 + self.config.clazz]

    def __str__(self):
        return (f"\n["
                f"detections:{self.detections}\n"
                f"class_probabilities: {self.class_probabilities}]\n")

if __name__ == "__main__":
    from Config import Config
    config = Config()
    print(Grid(torch.rand(30),3,3,config))