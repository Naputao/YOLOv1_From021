import torch


class Detection:
    def __init__(self, tensor, grid_x, grid_y):
        assert isinstance(tensor, torch.Tensor), f"Not a Tensor: {type(tensor)}"
        assert tensor.shape[0] == 5, f"Tensor shape is incorrect: {tensor.shape}"
        self.tensor = tensor
        self.grid_x = grid_x
        self.grid_y = grid_y

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
                f"grid_x:{self.grid_x}, "
                f"grid_y:{self.grid_y}, "
                f"x: {self.x}, "
                f"y: {self.y}, "
                f"w: {self.w}, "
                f"h: {self.h}, "
                f"confidence: {self.confidence}]")

    def __repr__(self):
        return self.__str__()
