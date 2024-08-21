import math
import IOU
import torch.nn as nn
from Grid import Grid
from Image import Image


class Loss(nn.Module):
    def __init__(self, predict, target, grid=7, bounding_boxes=2, clazz=20, lamda_coord=5, lamda_noobj=0.5):
        super(Loss, self).__init__()
        assert predict.shape[1] % (
                grid * grid * (5 * bounding_boxes + clazz)) == 0, f"Tensor shape is incorrect: {predict.shape}"
        self.target = target.view(-1, 5 * bounding_boxes + clazz)
        self.predict = predict.view(-1, grid, grid, (5 * bounding_boxes + clazz))
        self.grid = grid
        self.bounding_boxes = bounding_boxes
        self.clazz = clazz
        self.lamda_coord = lamda_coord
        self.lamda_noobj = lamda_noobj
        self.batch_size = self.predict.shape[0]
        self.image = Image(448, 448,7)
    def forward(self):
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

    def localization_loss(self):
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

    def confidence_loss(self):
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

    def classification_loss(self):
        loss = 0.0
        for i in range(self.batch_size):
            for grid_x in range(self.grid):
                for grid_y in range(self.grid):
                    predict = Grid(self.predict[i, grid_x, grid_y], grid_x, grid_y)
                    target = Grid(self.target[i, grid_x, grid_y], grid_x, grid_y)
                    for j in range(self.clazz):
                        loss += (target.class_probabilities[j] - predict.class_probabilities[j]) ** 2
        return loss
    def responsible_box(self):
        for box in self.target.detections:
        IOU()