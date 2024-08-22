import torch
from numpy.ma.extras import vstack
from torchgen.executorch.api.types import tensorT


class Annotation:
    def __init__(self,xml,cfg):
        self.xml = xml.getroot()
        self.cfg = cfg
    @property
    def objects(self):
        return [Object(obj,self.cfg.input_width/self.width,self.cfg.input_height/self.height) for obj in self.xml.findall('.//object')]
    @property
    def folder(self):
        return float(self.xml.find('.//folder').text)
    @property
    def width(self):
        return float(self.xml.find('.//width').text)
    @property
    def height(self):
        return float(self.xml.find('.//height').text)
    def to_list(self):
        ts = []
        for obj in self.objects:
            ts.append(obj.to_tensor())
        return ts
class Object:
    def __init__(self,xml,scale_x,scale_y):
        self.xml = xml
        self.scale_x = scale_x
        self.scale_y = scale_y

    @property
    def bottom(self):
        return float(self.xml.find(".//ymax").text)*self.scale_y
    @property
    def left(self):
        return float(self.xml.find(".//xmin").text)*self.scale_x
    @property
    def top(self):
        return float(self.xml.find(".//ymin").text)*self.scale_y
    @property
    def right(self):
        return float(self.xml.find(".//xmax").text)*self.scale_x
    @property
    def width(self):
        return self.right - self.left
    @property
    def height(self):
        return self.bottom - self.top
    @property
    def x(self):
        return (self.left + self.right) /2
    @property
    def y(self):
        return (self.top + self.bottom) /2
    def to_tensor(self):
        return torch.tensor([self.x, self.y, self.width, self.height])
if __name__ == '__main__':
    from Config import Config
    import xml.etree.ElementTree as ET
    import os
    cfg = Config()
    with open(os.getcwd()+"/Dataset/"+cfg.current_annotations_path) as xml_file:
        xml = Annotation(ET.parse(xml_file),cfg)
        print(xml.to_list())