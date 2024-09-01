import torch
import torch.nn.utils.rnn as rnn_utils
import math

class Annotation:
    def __init__(self,xml,cfg):
        self.xml = xml.getroot()
        self.cfg = cfg
    @property
    def objects(self):
        return [Object(obj,self.cfg) for obj in self.xml.findall('.//object')]

    def to_list(self):
        return [self.cfg.loc_synset_mapping[obj.name] for obj in self.objects]
class Object:
    def __init__(self,xml,cfg):
        self.xml = xml
        self.cfg = cfg
        self.name = xml.find('.//name').text
if __name__ == '__main__':
    from Config import Config
    import xml.etree.ElementTree as ET
    import os
    cfg = Config()
    with open(os.getcwd()+"/Dataset/"+cfg.current_annotations_path) as xml_file:
        xml = Annotation(ET.parse(xml_file),cfg)
        print(xml.to_list())