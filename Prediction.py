import torch
import Grid
class Prediction:
    def __init__(self,ts,cfg):
        assert isinstance(ts, torch.Tensor), f"Not a Tensor: {type(ts)}"
        assert ts.shape == torch.Size([cfg.batch_size,cfg.grid,cfg.grid,(cfg.clazz+5*cfg.bounding_boxes)]), f"Tensor shape is incorrect: {ts.shape} need:{torch.Size([cfg.batch_size,cfg.grid,cfg.grid,(cfg.clazz+5*cfg.bounding_boxes)])}"
        self.ts = ts
        self.cfg = cfg
        self.grids = [[[Grid.Grid(self.ts[batch_id,grid_x,grid_y],grid_x,grid_y,self.cfg)
                       for grid_y in range(self.cfg.grid)]
                      for grid_x in range(self.cfg.grid)]
                    for batch_id in range(cfg.batch_size)]