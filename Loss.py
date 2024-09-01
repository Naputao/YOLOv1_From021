import math
import time

import IOU
import torch
import torch.nn as nn
from Box import Box
from Prediction import Prediction
import torch.nn.functional as F

class Loss(nn.Module):
    def __init__(self,config):
        super(Loss, self).__init__()
        self.config = config

    def forward(self, output, target):
        batch_size = output.shape[0]
        target_size = target.shape[0]
        device = self.config.device
        lamda_coord = self.config.lamda_coord
        lamda_noobj = self.config.lamda_noobj
        lamda_obj = self.config.lamda_obj
        lamda_size = self.config.lamda_size
        bounding_boxes = self.config.bounding_boxes
        gridn = self.config.grid
        clazz = self.config.clazz
        target_id = torch.arange(target_size)
        batch_id =  target[...,-1].int()
        gx = target[...,4].int()
        gy = target[...,5].int()
        grid = output[batch_id,gx,gy]
        x_grid = grid[...,0:bounding_boxes*5:5]
        y_grid = grid[...,1:bounding_boxes*5:5]
        w_grid = grid[...,2:bounding_boxes*5:5]
        h_grid = grid[...,3:bounding_boxes*5:5]
        c_grid = grid[...,4:bounding_boxes*5:5]

        top_grid = y_grid-3.5*h_grid
        bottom_grid = y_grid+3.5*h_grid
        left_grid = x_grid-3.5*w_grid
        right_grid = x_grid+3.5*w_grid

        x_target =target[...,0]
        y_target =target[...,1]
        w_target =target[...,2]
        h_target =target[...,3]

        top_target =(y_target-3.5*h_target).unsqueeze(1).repeat(1,2)
        bottom_target =(y_target+3.5*h_target).unsqueeze(1).repeat(1,2)
        left_target =(x_target-3.5*w_target).unsqueeze(1).repeat(1,2)
        right_target =(x_target+3.5*w_target).unsqueeze(1).repeat(1,2)

        top_inter = torch.max(top_grid, top_target)
        bottom_inter = torch.min(bottom_grid, bottom_target)
        left_inter = torch.max(left_grid, left_target)
        right_inter = torch.min(right_grid, right_target)


        w_inter = right_inter-left_inter
        w_inter = torch.where(w_inter>0,w_inter,torch.tensor(0.0))#set 0 when width is a negative
        h_inter = bottom_inter-top_inter
        h_inter = torch.where(h_inter>0,h_inter,torch.tensor(0.0))#set 0 when width is a negative

        area_inter = w_inter*h_inter
        area_target = (w_target*h_target*49).unsqueeze(1).repeat(1,2)
        area_grid = w_grid*h_grid*49

        area_total = area_target + area_grid - area_inter

        iou = torch.where(area_total > 1e-6, area_inter / area_total, torch.tensor(0.0))

        argmax_iou = torch.argmax(iou,dim=1)

        x_responsible = x_grid[target_id,argmax_iou]
        y_responsible = y_grid[target_id,argmax_iou]
        w_responsible = w_grid[target_id,argmax_iou]
        h_responsible = h_grid[target_id,argmax_iou]
        c_responsible = c_grid[target_id,argmax_iou]

        loss = lamda_coord*(torch.sum((x_target-x_responsible)**2)+
                             torch.sum((y_target-y_responsible)**2))
        w_sqrt_target = torch.sqrt(torch.abs(w_target) + 1e-6) * torch.sign(w_target)
        h_sqrt_target = torch.sqrt(torch.abs(h_target) + 1e-6) * torch.sign(h_target)
        w_sqrt_responsible = torch.sqrt(torch.abs(w_responsible) + 1e-6) * torch.sign(w_responsible)
        h_sqrt_responsible = torch.sqrt(torch.abs(h_responsible) + 1e-6) * torch.sign(h_responsible)
        loss+= lamda_size*(torch.sum((w_sqrt_target-w_sqrt_responsible)**2)+
                            torch.sum((h_sqrt_target-h_sqrt_responsible)**2))

        loss += lamda_obj * torch.sum((c_responsible-1)**2)-lamda_noobj * torch.sum(c_responsible**2)

        c = output[...,4:5*bounding_boxes:5]
        loss += lamda_noobj * torch.sum(c **2)

        classification_grid = grid[..., 5*bounding_boxes:]
        classification_target = target[..., 6].int()
        classification_responsible = classification_grid[target_id,classification_target]
        loss +=(classification_grid**2).sum()
        loss +=((classification_responsible-1)**2-classification_responsible**2).sum()
        if torch.isnan(loss).any():
            with open('model_weights_on_nan.txt', 'w') as f:
                for var_name, var_value in locals().items():
                    f.write(f"{var_name}:\n{var_value}\n\n")
            raise ValueError('Loss is NaN')
        return loss
class PreTrainLoss(nn.Module):
    def __init__(self,config):
        super(PreTrainLoss, self).__init__()
        self.config = config

    def forward(self, output, target):
        output = F.softmax(output, dim=1)
        lambda_confidence = self.config.lambda_confidence
        batch_num = output.shape[0]
        batch_id = torch.arange(batch_num)
        confidence_responsible = output[batch_id,target]
        loss = -torch.log(confidence_responsible+1e-6).sum()
        if torch.isnan(loss).any():
            with open('model_weights_on_nan.txt', 'w') as f:
                for var_name, var_value in locals().items():
                    f.write(f"{var_name}:\n{var_value}\n\n")
            raise ValueError('Loss is NaN')
        return loss
if __name__ == '__main__':
    import Config
    cfg = Config.Config()
    criterion = PreTrainLoss(cfg)
    criterion.cuda()
    criterion.eval()
    output = torch.randn([16,1000],device='cuda')
    from Dataset_ILSVRC import Dataset
    from torch.utils.data import DataLoader
    dataset = Dataset(cfg)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=dataset.collate_fn)
    with torch.no_grad():
        print("start")
        for data_batch, target in dataloader:
            loss = criterion(output, target)
            print(loss)

    # import Config
    # cfg = Config.Config()
    # criterion = Loss(cfg)
    # criterion.cuda()
    # criterion.eval()
    # output = torch.randn([16,7,7,30],device='cuda')
    # from Dataset_VOC2012 import Dataset
    # from torch.utils.data import DataLoader
    # dataset = Dataset(cfg)
    # dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=dataset.collate_fn)
    # with torch.no_grad():
    #     print("start")
    #     for data_batch, target in dataloader:
    #         loss = criterion(output, target)
    #         if torch.isnan(loss):
    #             print(loss)


    # import Config
    # from YOLO import YOLO
    # from ALLZERO import *
    # from Dataset_VOC2012 import Dataset
    # from torch.utils.data import DataLoader
    # import os
    # cfg = Config.Config()
    # dataset = Dataset(cfg)
    # dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=dataset.collate_fn)
    # model = YOLO(cfg).eval()
    # device = cfg.device
    # model.to(device)
    # criterion = Loss(cfg).to(device)
    # saved_model_path = cfg.saved_model_path
    # if saved_model_path is not None:
    #     print(f"loading models on {saved_model_path}")
    #     model.load_state_dict(torch.load(saved_model_path, weights_only=True))
    # print(time.time())
    # loss_total = 0.0
    # from Image import Image
    # img = Image(cfg)
    # with torch.no_grad():
    #     for data_batch, target in dataloader:
    #         output = model(data_batch.to(device))
    #         img.show_with_annotation_and_detection_no_filter(data_batch,target,output)
    #         loss = criterion(output, target)
    #         loss_total += loss
    #         print(loss)
    # print(loss_total)
    # print(time.time())
    # output = torch.rand([cfg.batch_size,cfg.grid,cfg.grid,cfg.bounding_boxes*5+cfg.clazz])
    # target = [[torch.rand(4)*448]*2]*cfg.batch_size
    # print(Loss(cfg).forward(output, target))
