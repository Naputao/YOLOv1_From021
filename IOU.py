import Box

class IOU:
    def __init__(self, this, that, cfg):
        assert isinstance(this, Box.Box)
        assert isinstance(that, Box.Box)
        self.config = cfg
        self.this = this
        self.that = that
        self.iou = self.compute_iou()

    def box_corners(self, it):
        box_center_x = self.config.grid_width * (it.x + it.grid_x)
        box_center_y = self.config.grid_height * (it.y + it.grid_y)
        box_width = self.config.input_width * it.w
        box_height = self.config.input_height * it.h

        left = box_center_x - box_width / 2
        right = box_center_x + box_width / 2
        top = box_center_y - box_height / 2
        bottom = box_center_y + box_height / 2
        return [left, top, right, bottom]

    def compute_iou(self):
        this_left, this_top, this_right, this_bottom = self.box_corners(self.this)
        that_left, that_top, that_right, that_bottom = self.box_corners(self.that)

        inter_left = max(this_left, that_left)
        inter_top = max(this_top, that_top)
        inter_right = min(this_right, that_right)
        inter_bottom = min(this_bottom, that_bottom)

        this_area = (this_right - this_left) * (this_bottom - this_top)
        that_area = (that_right - that_left) * (that_bottom - that_top)
        inter_area = (inter_right - inter_left) * (inter_bottom - inter_top)

        union_area = this_area + that_area - inter_area
        iou = inter_area / union_area if union_area else 0
        return iou


if __name__ == '__main__':
    import torch
    import Config
    import time
    config = Config.Config()
    start_time = time.time()
    for i in range(10000):
        box1 = Box.Box(torch.rand(5), 0, 0)
        box2 = Box.Box(torch.rand(5), 0, 0)
        IOU(box1, box2, config)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time:.4f} seconds")