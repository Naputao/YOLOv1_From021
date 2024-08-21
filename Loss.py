import math
import IOU
import torch.nn as nn
from Grid import Grid
from Image import Image, config
import Annotation

class Loss(nn.Module):
    def __init__(self,config):
        super(Loss, self).__init__()
        self.config = config
        self.batch_size = self.predict.shape[0]
        self.image = Image(self.config)
    def forward(self,predict, annotation):
        #equal to localization_loss()+confidence_loss()+classification_loss()
        loss = 0.0
        for i in range(self.batch_size):
            for grid_x in range(self.grid):
                for grid_y in range(self.grid):
                    predict = Grid(self.predict[i, grid_x, grid_y], grid_x, grid_y)
                    target = Grid(self.target[i, grid_x, grid_y], grid_x, grid_y)
                    for j in range(self.bounding_boxes):
                        if target.detections[j].confidence == 0:
                            loss += self.lamda_noobj * (
                                    predict.detections[j].confidence - target.detections[j].confidence) ** 2
                        else:
                            loss += ((predict.detections[j].confidence - target.detections[j].confidence) ** 2 +
                                     self.lamda_coord * ((predict.detections[j].x - target.detections[j].x) ** 2 +
                                                         (predict.detections[j].y - target.detections[j].y) ** 2 +
                                                         (math.sqrt(predict.detections[j].w) - math.sqrt(
                                                             target.detections[j].w)) ** 2 +
                                                         (math.sqrt(predict.detections[j].h) - math.sqrt(
                                                             target.detections[j].h)) ** 2))
                    for j in range(self.clazz):
                        loss += (target.class_probabilities[j] - predict.class_probabilities[j]) ** 2
        return loss

    def localization_loss(self,predict, annotation):
        loss = 0.0
        for i in range(self.batch_size):
            for grid_x in range(self.grid):
                for grid_y in range(self.grid):
                    predict = Grid(self.predict[i, grid_x, grid_y], grid_x, grid_y)
                    target = Grid(self.target[i, grid_x, grid_y], grid_x, grid_y)
                    for j in range(self.bounding_boxes):
                        if target.detections[j].confidence == 0: continue
                        loss += ((predict.detections[j].x - target.detections[j].x) ** 2 +
                                 (predict.detections[j].y - target.detections[j].y) ** 2 +
                                 (math.sqrt(predict.detections[j].w) - math.sqrt(target.detections[j].w)) ** 2 +
                                 (math.sqrt(predict.detections[j].h) - math.sqrt(target.detections[j].h)) ** 2)
        return loss * self.lamda_coord

    def confidence_loss(self,predict, annotation):
        loss = 0.0
        for i in range(self.batch_size):
            for grid_x in range(self.grid):
                for grid_y in range(self.grid):
                    predict = Grid(self.predict[i, grid_x, grid_y], grid_x, grid_y)
                    target = Grid(self.target[i, grid_x, grid_y], grid_x, grid_y)
                    for j in range(self.bounding_boxes):
                        if target.detections[j].confidence == 0:
                            loss += self.lamda_noobj * (
                                        predict.detections[j].confidence - target.detections[j].confidence) ** 2
                        else:
                            loss += (predict.detections[j].confidence - target.detections[j].confidence) ** 2
        return loss

    def classification_loss(self,predict, annotation):
        loss = 0.0
        for i in range(self.batch_size):
            for grid_x in range(self.grid):
                for grid_y in range(self.grid):
                    predict = Grid(self.predict[i, grid_x, grid_y], grid_x, grid_y)
                    target = Grid(self.target[i, grid_x, grid_y], grid_x, grid_y)
                    for j in range(self.clazz):
                        loss += (target.class_probabilities[j] - predict.class_probabilities[j]) ** 2
        return loss
    def responsible_box(self,predict, annotation):


if __name__ == '__main__':
    import Config
    import torch
    import YOLO
    cfg = config.Config()
    predict = torch.randn(32, 1470)
    Loss(cfg).forward()