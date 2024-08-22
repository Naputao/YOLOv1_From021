import math
import IOU
import torch
import torch.nn as nn
from Box import Box
from Prediction import Prediction


class Loss(nn.Module):
    def __init__(self,config):
        super(Loss, self).__init__()
        self.config = config
    def forward(self,output, target):
        loss = 0.0
        prediction = Prediction(output, self.config)
        for batch_id in range(self.config.batch_size):
            for target_box in target[batch_id]:
                target_box = Box(target_box)
                grid_x = int(target_box.x)//self.config.grid_width
                grid_y = int(target_box.y)//self.config.grid_height
                tensor_1d = torch.tensor(
                    [IOU.IOU(target_box, prediction.grids[batch_id][grid_x][grid_y].detections[i]).iou
                       for i in range(self.config.bounding_boxes)])
                #find which box is responsible to predict
                max_index = torch.argmax(tensor_1d)
                output_box = output[batch_id,grid_x,grid_y][5*max_index:5*max_index+5]
                target_box = target_box.normalize(grid_x,grid_y,self.config)
                #localization_loss
                loss += ((output_box[0] - target_box[0]) ** 2 +
                         (output_box[1] - target_box[1]) ** 2 +
                         (math.sqrt(output_box[2]) - math.sqrt(target_box[2])) ** 2 +
                         (math.sqrt(output_box[3]) - math.sqrt(target_box[3])) ** 2) * self.config.lamda_coord
                #confidence_loss
                loss += (output_box[4] - 1) ** 2 - (output_box[4] ** 2) * self.config.lamda_noobj
            for grid_y in range(self.config.grid):
                for grid_x in range(self.config.grid):
                    for i in range(self.config.bounding_boxes):
                        output_box = output[batch_id,grid_x,grid_y][5*i:5*i+5]
                        loss += (output_box[4] ** 2) * self.config.lamda_noobj
        return loss
    #
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
    import Dataset
    from torch.utils.data import DataLoader
    cfg = Config.Config()
    dataset = Dataset.Dataset(cfg)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=Dataset.collate_fn)
    model = YOLO.YOLO(cfg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    for data_batch, target in dataloader:
        output = model.forward(data_batch.to(device))
        print(Loss(cfg).forward(output, target))
    # output = torch.rand([cfg.batch_size,cfg.grid,cfg.grid,cfg.bounding_boxes*5+cfg.clazz])
    # target = [[torch.rand(4)*448]*2]*cfg.batch_size
    # print(Loss(cfg).forward(output, target))
