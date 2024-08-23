import torch
import torch.nn.utils.rnn as rnn_utils
import math

class Annotation:
    def __init__(self,xml,cfg):
        self.xml = xml.getroot()
        self.cfg = cfg
    @property
    def objects(self):
        return [Object(obj,self.width,self.height,self.cfg) for obj in self.xml.findall('.//object')]
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
            ts.append(obj.tensor)
        return ts
class Object:
    def __init__(self,xml,scale_x,scale_y,cfg):
        self.bottom = float(xml.find(".//ymax").text)
        self.top = float(xml.find(".//ymin").text)
        self.left = float(xml.find(".//xmin").text)
        self.right = float(xml.find(".//xmax").text)
        self.x,self.gx = math.modf((self.left + self.right)*cfg.grid/2/scale_x)
        self.y,self.gy = math.modf((self.top + self.bottom)*cfg.grid/2/scale_y)
        self.w = (self.right - self.left)/scale_x
        self.h = (self.bottom - self.top)/scale_y
        self.tensor = torch.tensor([self.x, self.y, self.w, self.h,self.gx,self.gy]).to(cfg.device)
if __name__ == '__main__':
    from Config import Config
    import xml.etree.ElementTree as ET
    import os
    cfg = Config()
    with open(os.getcwd()+"/Dataset/"+cfg.current_annotations_path) as xml_file:
        xml = Annotation(ET.parse(xml_file),cfg)
        print(xml.to_list())