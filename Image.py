import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import Annotation_ILSVRC
class Image:
    def __init__(self, config):
        self.config = config
    def show(self):
        annotation = Annotation.Annotation(self.config.current_annotations_path)
        img = mpimg.imread(self.config.current_images_path)
        fig, ax = plt.subplots()
        ax.imshow(img)
        for obj in annotation.objects:
            rect = patches.Rectangle((obj.left, obj.top), obj.width, obj.height, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
        plt.show()

if __name__ == '__main__':
    import Config
    config = Config.Config()
    Image(config).show()
