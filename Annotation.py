import xml.etree.ElementTree as ET
class Annotation:
    def __init__(self,path):
        self.path = path
        self.tree = ET.parse(path)
    @property
    def objects(self):
        return [Object(obj) for obj in self.tree.getroot().findall('.//object')]

class Object:
    def __init__(self,tree):
        self.tree = tree
    @property
    def bottom(self):
        return int(self.tree.find(".//ymax").text)
    @property
    def left(self):
        return int(self.tree.find(".//xmin").text)
    @property
    def top(self):
        return int(self.tree.find(".//ymin").text)
    @property
    def right(self):
        return int(self.tree.find(".//xmax").text)
    @property
    def width(self):
        return self.right - self.left
    @property
    def height(self):
        return self.bottom - self.top

if __name__ == '__main__':
    from Config import Config
    cfg = Config()
    annotation = Annotation(cfg.current_annotations_path)
    print(annotation.objects[0].height)