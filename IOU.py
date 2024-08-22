import Box

def box_corners(it):
    left = it.x - it.w / 2
    right = it.x + it.w / 2
    top = it.y - it.h / 2
    bottom = it.y + it.h / 2
    return [left, top, right, bottom]

class IOU:
    def __init__(self, this, that):
        assert isinstance(this, Box.Box)
        assert isinstance(that, Box.Box)
        this_left, this_top, this_right, this_bottom = box_corners(this)
        that_left, that_top, that_right, that_bottom = box_corners(that)

        inter_left = max(this_left, that_left)
        inter_top = max(this_top, that_top)
        inter_right = min(this_right, that_right)
        inter_bottom = min(this_bottom, that_bottom)

        this_area = (this_right - this_left) * (this_bottom - this_top)
        that_area = (that_right - that_left) * (that_bottom - that_top)
        inter_area = (inter_right - inter_left) * (inter_bottom - inter_top)

        union_area = this_area + that_area - inter_area
        self.iou = inter_area / union_area if union_area else 0



if __name__ == '__main__':
    import torch
    import time
    start_time = time.time()
    for i in range(10000):
        box1 = Box.Box(torch.rand(5))
        box2 = Box.Box(torch.rand(4))
        IOU(box1, box2)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time:.4f} seconds")