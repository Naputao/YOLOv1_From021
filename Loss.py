import math
import time

import IOU
import torch
import torch.nn as nn
from Box import Box
from Prediction import Prediction


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
        bounding_boxes = self.config.bounding_boxes
        gridn = self.config.grid
        clazz = self.config.clazz
        target_id = torch.arange(target_size)
        batch_id =  target[...,-1].view(-1).int()
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
        h_inter = top_inter-bottom_inter
        h_inter = torch.where(h_inter>0,h_inter,torch.tensor(0.0))#set 0 when width is a negative

        area_inter = w_inter*h_inter
        area_target = (w_target*h_target*49).unsqueeze(1).repeat(1,2)
        area_grid = w_grid*h_grid*49

        area_total = area_target + area_grid - area_inter

        iou = torch.where(area_total > 0, area_inter / area_total, torch.tensor(0.0))

        argmax_iou = torch.argmax(iou,dim=1)

        x_responsible = x_grid[target_id,argmax_iou]
        y_responsible = y_grid[target_id,argmax_iou]
        w_responsible = w_grid[target_id,argmax_iou]
        h_responsible = h_grid[target_id,argmax_iou]
        c_responsible = c_grid[target_id,argmax_iou]

        loss = lamda_coord *torch.sum((x_target-x_responsible)**2+
         (y_target-y_responsible)**2+
         (torch.sqrt(w_target)-torch.sqrt(w_responsible))**2+
         (torch.sqrt(h_target)-torch.sqrt(h_responsible))**2)
        loss += torch.sum((c_responsible-1)**2-lamda_noobj * c_responsible**2)

        c = output[...,4:5*bounding_boxes:5]
        loss += torch.sum(lamda_noobj * c **2)

        classification_grid = grid[..., 5*bounding_boxes:]
        classification_target = target[..., 6].int()
        classification_responsible = classification_grid[target_id,classification_target]
        loss +=(classification_grid**2).sum()
        loss +=((classification_responsible-1)**2-classification_responsible**2).sum()
        return loss/batch_size

    # def forward(self,output, target):
    #     loss = 0.0
    #     prediction = Prediction(output, self.config)
    #     batch_size = self.config.batch_size
    #     grid_width = self.config.grid_width
    #     grid_height = self.config.grid_height
    #     lamda_coord = self.config.lamda_coord
    #     lamda_noobj = self.config.lamda_noobj
    #     bounding_boxes = self.config.bounding_boxes
    #     grid = self.config.grid
    #     clazz = self.config.clazz
    #     device = self.config.device
    #     for batch_id in range(batch_size):
    #         for target_box in target[batch_id]:
    #             target_box = Box(target_box)
    #             grid_x = int(target_box.x)//grid_width
    #             grid_y = int(target_box.y)//grid_height
    #             tensor_1d = torch.tensor(
    #                 [IOU.IOU(target_box, prediction.grids[batch_id][grid_x][grid_y].detections[i]).iou
    #                    for i in range(bounding_boxes)])
    #             #find which box is responsible to predict
    #             max_index = torch.argmax(tensor_1d)
    #             output_box = output[batch_id,grid_x,grid_y][5*max_index:5*max_index+5]
    #             output_classification = output[batch_id,grid_x,grid_y][-clazz:]
    #             target_classification = torch.rand(clazz).to(device)
    #             target_box = target_box.normalize(grid_x,grid_y,self.config)
    #             #localization_loss
    #             loss += ((output_box[0] - target_box[0]) ** 2 +
    #                      (output_box[1] - target_box[1]) ** 2 +
    #                      (math.sqrt(output_box[2]) - math.sqrt(target_box[2])) ** 2 +
    #                      (math.sqrt(output_box[3]) - math.sqrt(target_box[3])) ** 2) * lamda_coord
    #             #confidence_loss
    #             loss += (output_box[4] - 1) ** 2 - (output_box[4] ** 2) * lamda_noobj
    #             #classification_loss
    #             loss += torch.dot(output_classification, target_classification)**2
    #         for grid_y in range(grid):
    #             for grid_x in range(grid):
    #                 for i in range(bounding_boxes):
    #                     output_box = output[batch_id,grid_x,grid_y][5*i:5*i+5]
    #                     loss += (output_box[4] ** 2) * lamda_noobj
    #     return loss

    # def forward(self, output, target):
    #     loss = 0.0
    #     prediction = Prediction(output, self.config)
    #
    #     batch_size = self.config.batch_size
    #     grid_width = self.config.grid_width
    #     grid_height = self.config.grid_height
    #     lamda_coord = self.config.lamda_coord
    #     lamda_noobj = self.config.lamda_noobj
    #     bounding_boxes = self.config.bounding_boxes
    #     grid_size = self.config.grid
    #
    #     # Batch-wise processing
    #     for batch_id in range(batch_size):
    #         # Convert target to boxes and grids only once
    #         target_boxes = [Box(target_box) for target_box in target[batch_id]]
    #         for target_box in target_boxes:
    #             grid_x = int(target_box.x) // grid_width
    #             grid_y = int(target_box.y) // grid_height
    #
    #             iou_scores = torch.tensor([
    #                 IOU.IOU(target_box, prediction.grids[batch_id][grid_x][grid_y].detections[i]).iou
    #                 for i in range(bounding_boxes)
    #             ])
    #             # Identify the responsible box
    #             max_index = torch.argmax(iou_scores)
    #             output_box = output[batch_id, grid_x, grid_y][5 * max_index:5 * max_index + 5]
    #
    #             # Normalize target box
    #             target_box = target_box.normalize(grid_x, grid_y, self.config)
    #
    #             # Localization loss
    #             loss += ((output_box[0] - target_box[0]) ** 2 +
    #                      (output_box[1] - target_box[1]) ** 2 +
    #                      (math.sqrt(output_box[2]) - math.sqrt(target_box[2])) ** 2 +
    #                      (math.sqrt(output_box[3]) - math.sqrt(target_box[3])) ** 2) * lamda_coord
    #
    #             # Confidence loss (object)
    #             loss += (output_box[4] - 1) ** 2 * lamda_noobj
    #
    #         # Confidence loss (no object)
    #         for grid_y in range(grid_size):
    #             for grid_x in range(grid_size):
    #                 for i in range(bounding_boxes):
    #                     output_box = output[batch_id, grid_x, grid_y][5 * i:5 * i + 5]
    #                     loss += (output_box[4] ** 2) * lamda_noobj
    #
    #     return loss
    # def localization_loss(self,output, target):
    #     loss = 0.0
    #     for i in range(self.batch_size):
    #         for grid_x in range(self.grid):
    #             for grid_y in range(self.grid):
    #                 predict = Grid(self.predict[i, grid_x, grid_y], grid_x, grid_y)
    #                 target = Grid(self.target[i, grid_x, grid_y], grid_x, grid_y)
    #                 for j in range(self.bounding_boxes):
    #                     if target.detections[j].confidence == 0: continue
    #                     loss += ((predict.detections[j].x - target.detections[j].x) ** 2 +
    #                              (predict.detections[j].y - target.detections[j].y) ** 2 +
    #                              (math.sqrt(predict.detections[j].w) - math.sqrt(target.detections[j].w)) ** 2 +
    #                              (math.sqrt(predict.detections[j].h) - math.sqrt(target.detections[j].h)) ** 2)
    #     return loss * self.lamda_coord
    #
    # def confidence_loss(self,output, target):
    #     loss = 0.0
    #     for i in range(self.batch_size):
    #         for grid_x in range(self.grid):
    #             for grid_y in range(self.grid):
    #                 predict = Grid(self.predict[i, grid_x, grid_y], grid_x, grid_y)
    #                 target = Grid(self.target[i, grid_x, grid_y], grid_x, grid_y)
    #                 for j in range(self.bounding_boxes):
    #                     if target.detections[j].confidence == 0:
    #                         loss += self.lamda_noobj * (
    #                                     predict.detections[j].confidence - target.detections[j].confidence) ** 2
    #                     else:
    #                         loss += (predict.detections[j].confidence - target.detections[j].confidence) ** 2
    #     return loss
    #
    # def classification_loss(self,output, target):
    #     loss = 0.0
    #     for i in range(self.batch_size):
    #         for grid_x in range(self.grid):
    #             for grid_y in range(self.grid):
    #                 predict = Grid(self.predict[i, grid_x, grid_y], grid_x, grid_y)
    #                 target = Grid(self.target[i, grid_x, grid_y], grid_x, grid_y)
    #                 for j in range(self.clazz):
    #                     loss += (target.class_probabilities[j] - predict.class_probabilities[j]) ** 2
    #     return loss
    # def responsible_box(self,output, target):
    #     responsible_box_batch = []
    #     prediction = Prediction(output,self.config)
    #     for batch_id in range(self.config.batch_size):
    #         responsible_box = []
    #         for target_box in target[batch_id]:
    #             target_box = Box(target_box)
    #             tensor_3d = torch.tensor([[[IOU.IOU(target_box,prediction.get_grid(batch_id, grid_x, grid_y).detections[i]).iou
    #                                     for grid_x in range(self.config.grid)]
    #                                     for grid_y in range(self.config.grid)]
    #                                     for i in range(self.config.bounding_boxes)])
    #             max_index = torch.argmax(tensor_3d)
    #             max_index_3d = torch.unravel_index(max_index, tensor_3d.shape)
    #             responsible_box.append(max_index_3d)
    #         responsible_box_batch.append(responsible_box)
    #     return responsible_box_batch

if __name__ == '__main__':
    import Config
    import YOLO
    from ALLZERO import *
    from Dataset_VOC2012 import Dataset
    from torch.utils.data import DataLoader
    import os
    cfg = Config.Config()
    dataset = Dataset(cfg)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=dataset.collate_fn)
    model = ALLZERO(cfg)
    device = torch.device('cuda')
    model.to(device)
    criterion = Loss(cfg).to(device)
    saved_model_path = cfg.saved_model_path
    if saved_model_path is not None:
        print(f"loading models on {saved_model_path}")
        model.load_state_dict(torch.load(saved_model_path, weights_only=True))
    print(time.time())
    loss_total = 0.0
    with torch.no_grad():
        for data_batch, target in dataloader:
            output = model(data_batch.to(device))
            loss = criterion(output, target)
            loss_total += loss
            print(loss)
    print(loss_total)
    print(time.time())
    # output = torch.rand([cfg.batch_size,cfg.grid,cfg.grid,cfg.bounding_boxes*5+cfg.clazz])
    # target = [[torch.rand(4)*448]*2]*cfg.batch_size
    # print(Loss(cfg).forward(output, target))
