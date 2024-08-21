import torch
import Box
class Grid:
    def __init__(self, tensor,grid_x,grid_y, bounding_boxes=2, clazz=20):
        assert isinstance(tensor, torch.Tensor), f"Not a Tensor: {type(tensor)}"
        assert tensor.shape[0] == 5 * bounding_boxes + clazz, f"Tensor shape is incorrect: {tensor.shape}"
        self.tensor = tensor
        self.clazz = clazz
        self.bounding_boxes = bounding_boxes
        self.grid_x = grid_x
        self.grid_y = grid_y

    @property
    def detections(self):
        return [Box.Box(self.tensor[i:i + 5], grid_x=self.grid_x, grid_y=self.grid_y) for i in range(self.bounding_boxes)]

    @property
    def class_probabilities(self):
        return self.tensor[self.bounding_boxes * 5:self.bounding_boxes * 5 + self.clazz]

    def __str__(self):
        return (f"\n["
                f"detections:{self.detections}\n"
                f"class_probabilities: {self.class_probabilities}]\n")
