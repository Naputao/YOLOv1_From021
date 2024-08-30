import torch
from numpy.ma.core import argmax, minimum
from requests.packages import target

from Config import Config


class NMS:
    def __init__(self,cfg):
        self.cfg = cfg
    def filter(self,ts):
        batch_size = ts.shape[0]
        bnbox_size =  self.cfg.bounding_boxes
        grid = self.cfg.grid
        bnbox_len = grid * grid * bnbox_size
        minimum_confidence = self.cfg.minimum_confidence
        maximum_iou = self.cfg.maximum_iou
        bnbox = ts[...,0:5*bnbox_size].reshape(batch_size,grid*grid*bnbox_size,5)
        seq1 = torch.arange(7).repeat_interleave(2).repeat(7).repeat(batch_size,1).unsqueeze(2)
        seq2 = torch.arange(7).repeat_interleave(14).repeat(batch_size,1).unsqueeze(2)
        bnbox = torch.cat((bnbox,seq2,seq1),dim=-1)
        bnbox_filtered = []
        for i in range(batch_size):
            batch_bnbox = bnbox[i]
            mask = batch_bnbox[...,4] > minimum_confidence
            batch_bnbox = batch_bnbox[mask]
            batch_top = batch_bnbox[..., 1] + batch_bnbox[..., 6] - 3.5 * batch_bnbox[..., 3]
            batch_bottom = batch_bnbox[..., 1] + batch_bnbox[..., 6] + 3.5 * batch_bnbox[..., 3]
            batch_left = batch_bnbox[..., 0] + batch_bnbox[..., 5] - 3.5 * batch_bnbox[..., 2]
            batch_right = batch_bnbox[..., 0] + batch_bnbox[..., 5] + 3.5 * batch_bnbox[..., 2]
            batch_area = batch_bnbox[..., 2] * batch_bnbox[..., 3] * 49

            batch_bnbox_filtered = []
            while batch_bnbox.shape[0]>0:

                bnbox_len = batch_bnbox.shape[0]

                max_values, max_indices = torch.max(batch_bnbox[..., 4], dim=-1)
                batch_bnbox_filtered.append(batch_bnbox[max_indices])

                max_top = batch_top[max_indices].repeat(bnbox_len)
                max_bottom = batch_bottom[max_indices].repeat(bnbox_len)
                max_left = batch_left[max_indices].repeat(bnbox_len)
                max_right = batch_right[max_indices].repeat(bnbox_len)
                max_area = batch_area[max_indices].repeat(bnbox_len)

                inter_top = torch.max(max_top,batch_top)
                inter_bottom = torch.min(max_bottom,batch_bottom)
                inter_left = torch.max(max_left,batch_left)
                inter_right = torch.min(max_right,batch_right)
                inter_width = inter_right - inter_left
                inter_width = torch.where(inter_width>0,inter_width,torch.tensor(0.0))
                inter_height = inter_bottom - inter_top
                inter_height = torch.where(inter_height>0,inter_height,torch.tensor(0.0))
                inter_area = inter_width*inter_height

                total_area = max_area + batch_area - inter_area
                iou = torch.where(total_area > 0, inter_area / total_area, torch.tensor(0.0))
                mask = iou < maximum_iou
                batch_bnbox = batch_bnbox[mask]
                batch_top = batch_top[mask]
                batch_bottom = batch_bottom[mask]
                batch_left = batch_left[mask]
                batch_right = batch_right[mask]
                batch_area = batch_area[mask]
            bnbox_filtered.append(batch_bnbox_filtered)
        return bnbox_filtered
    def no_filter(self,ts):
        batch_size = ts.shape[0]
        bnbox_size = self.cfg.bounding_boxes
        grid = self.cfg.grid
        bnbox = ts[..., 0:5 * bnbox_size].reshape(batch_size, grid * grid * bnbox_size, 5)
        seq1 = torch.arange(7).repeat_interleave(2).repeat(7).repeat(batch_size, 1).unsqueeze(2)
        seq2 = torch.arange(7).repeat_interleave(14).repeat(batch_size, 1).unsqueeze(2)
        bnbox = torch.cat((bnbox, seq2, seq1), dim=-1)
        bnbox_filtered = []
        for i in bnbox:
            batch_bnbox_filtered = []
            for j in i:
                batch_bnbox_filtered.append(j)
            bnbox_filtered.append(batch_bnbox_filtered)
        return bnbox_filtered
    def filter_max_confident(self,ts):
        batch_size = ts.shape[0]
        bnbox_size = self.cfg.bounding_boxes
        grid = self.cfg.grid
        bnbox = ts[..., 0:5 * bnbox_size].reshape(batch_size, grid * grid * bnbox_size, 5)
        seq1 = torch.arange(7).repeat_interleave(2).repeat(7).repeat(batch_size, 1).unsqueeze(2)
        seq2 = torch.arange(7).repeat_interleave(14).repeat(batch_size, 1).unsqueeze(2)
        bnbox = torch.cat((bnbox, seq2, seq1), dim=-1)
        bnbox_filtered = []
        for i in bnbox:
            batch_bnbox_filtered = []
            top5_values, top5_indices = torch.topk(i[...,4], 3)
            for j in i[top5_indices]:
                batch_bnbox_filtered.append(j)
            bnbox_filtered.append(batch_bnbox_filtered)
        return bnbox_filtered
if __name__ == '__main__':
    cfg = Config()
    nms = NMS(cfg)
    print(nms.filter_max_confident(torch.rand([16,7,7,30]),5))