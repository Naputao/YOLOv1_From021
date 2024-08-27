import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
from matplotlib import patches

import NMS



class Image:
    def __init__(self, config):
        self.config = config
    def show(self,imgs):
        for img in imgs:
            fig, ax = plt.subplots()
            ax.imshow(img.cpu().permute(1, 2, 0).numpy())
            plt.show()
    def show_with_detection(self,imgs,output):
        imgs = imgs.cpu()
        output = NMS.NMS(self.config).filter(output.cpu())
        for id, img in enumerate(imgs):
            fig, ax = plt.subplots()
            ax.imshow(img.cpu().permute(1, 2, 0).numpy())
            for box in output[id]:
                x = (box[0] + box[5] - 3.5 * box[2]).detach().numpy() * self.config.grid_width
                y = (box[1] + box[6] - 3.5 * box[3]).detach().numpy() * self.config.grid_width
                w = box[2].detach().numpy() * self.config.input_width
                h = box[3].detach().numpy() * self.config.input_height
                rect = patches.Rectangle((x, y), w, h, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
            plt.show()
        return self
    def show_with_annotation(self,imgs,targets):
        imgs = imgs.cpu()
        targets = targets.cpu()
        table = {}
        for box in targets:
            if int(box[7].item()) in table.keys():
                table[int(box[7].item())].append(box)
            else:
                table[int(box[7].item())] = [box]
        for id,img in enumerate(imgs):
            fig, ax = plt.subplots()
            ax.imshow(img.cpu().permute(1, 2, 0).numpy())
            for box in table[id]:
                x = (box[0]+box[4]-3.5*box[2])*self.config.grid_width
                y = (box[1]+box[5]-3.5*box[3])*self.config.grid_width
                w = box[2]*self.config.input_width
                h = (box[3]*self.config.input_height)
                rect = patches.Rectangle((x,y), w, h, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
            plt.show()
        return self
    def show_with_annotation_and_detection(self,imgs,targets,output):
        imgs = imgs.cpu()
        output = NMS.NMS(self.config).filter(output.cpu())
        print(output)
        table = {}
        targets = targets.cpu()
        for box in targets:
            if int(box[7].item()) in table.keys():
                table[int(box[7].item())].append(box)
            else:
                table[int(box[7].item())] = [box]

        for id, img in enumerate(imgs):
            fig, ax = plt.subplots()
            ax.imshow(img.cpu().permute(1, 2, 0).numpy())
            for box in output[id]:
                x = (box[0] + box[5] - 3.5 * box[2]).detach().numpy() * self.config.grid_width
                y = (box[1] + box[6] - 3.5 * box[3]).detach().numpy() * self.config.grid_width
                w = box[2].detach().numpy() * self.config.input_width
                h = box[3].detach().numpy() * self.config.input_height
                rect = patches.Rectangle((x, y), w, h, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
            for box in table[id]:
                x = (box[0]+box[4]-3.5*box[2])*self.config.grid_width
                y = (box[1]+box[5]-3.5*box[3])*self.config.grid_width
                w = box[2]*self.config.input_width
                h = (box[3]*self.config.input_height)
                rect = patches.Rectangle((x,y), w, h, edgecolor='b', facecolor='none')
                ax.add_patch(rect)
            plt.show()
        return self
    def show_with_annotation_and_detection_no_filter(self,imgs,targets,output):
        imgs = imgs.cpu()
        output = NMS.NMS(self.config).no_filter(output.cpu())
        table = {}
        targets = targets.cpu()
        for box in targets:
            if int(box[7].item()) in table.keys():
                table[int(box[7].item())].append(box)
            else:
                table[int(box[7].item())] = [box]

        for id, img in enumerate(imgs):
            fig, ax = plt.subplots()
            ax.imshow(img.cpu().permute(1, 2, 0).numpy())
            ax.set_xlim(0, 448)
            ax.set_ylim(448,0)
            grid_exist = set()

            for box in table[id]:
                x = (box[0]+box[4]-3.5*box[2]).detach().numpy()*self.config.grid_width
                y = (box[1]+box[5]-3.5*box[3]).detach().numpy()*self.config.grid_width
                w = box[2].detach().numpy() * self.config.input_width
                h = box[3].detach().numpy() * self.config.input_height
                rect = patches.Rectangle((x,y), w, h, edgecolor='b', facecolor='none')
                ax.plot((box[0]+box[4]).detach().numpy()*self.config.grid_width, (box[1]+box[5]).detach().numpy()*self.config.grid_width, 'bo')
                ax.add_patch(rect)
                grid_exist.add((int(box[4].item()),int(box[5].item())))
            for box in output[id]:
                if (int(box[5].item()), int(box[6].item())) in grid_exist:
                    x = (box[0] + box[5] - 3.5 * box[2]).detach().numpy() * self.config.grid_width
                    y = (box[1] + box[6] - 3.5 * box[3]).detach().numpy() * self.config.grid_width
                    w = box[2].detach().numpy() * self.config.input_width
                    h = box[3].detach().numpy() * self.config.input_height
                    rect = patches.Rectangle((x, y), w, h, edgecolor='r', facecolor='none')
                    ax.add_patch(rect)
            plt.show()
        return self
if __name__ == '__main__':
    from Dataset_VOC2012 import Dataset
    from torch.utils.data import DataLoader
    import Config
    from YOLO import YOLO
    cfg = Config.Config()
    dataset = Dataset(cfg)
    image = Image(cfg)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=dataset.collate_fn)
    model = YOLO(cfg)
    device = torch.device('cuda')
    model.to(device)
    saved_model_path = cfg.saved_model_path
    if saved_model_path is not None:
        print(f"loading models on {saved_model_path}")
        model.load_state_dict(torch.load(saved_model_path, weights_only=True))
    torch.set_printoptions(threshold=torch.inf)
    for batch_id, (data_batch, target) in enumerate(dataloader):
        output = model(data_batch)
        with open("tensor.log", 'w') as f:
            f.write(str(output))
        image.show_with_annotation_and_detection(data_batch,target,output)
        break