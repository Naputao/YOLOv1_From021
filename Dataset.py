import zipfile
import Config
from PIL import Image
class Dataset:
    def __init__(self, config):
        self.config = config
        self.file=zipfile.ZipFile(config.zip_dataset_path, 'r')
        self.file_list = list({f[33:-4] for f in self.file.namelist() if f.startswith(config.train_annotations_path)} &
                          {f[26:-5] for f in self.file.namelist() if f.startswith(config.train_images_path)})
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.config.train_images_path+self.file_list[idx]+".JPEG"
        target_path = self.config.train_annotations_path+self.file_list[idx]+".xml"
        img = None
        with self.file.open(img_path) as img_file:
            img = self.config.transform(Image.open(img_file).convert('RGB'))

        return (input,target)



if __name__ == '__main__':
    config = Config.Config()

    dataset=Dataset(config)
    print(dataset.file_list[0:6])
    # input_file_list_diff = [f for f in dataset.file.namelist() if f.startswith(config.train_images_path)]
    # #
    # #
    # list1 = input_file_list_diff
    # list2 = dataset.input_file_list
    #
    # diff1 = list(set(list1) - set(list2))
    # print(diff1[0:2])  # 输出: [1, 2, 3]
    #
    # # 差集：在 list2 中但不在 list1 中的元素
    # diff2 = list(set(list2) - set(list1))
    # print(diff2)  # 输出: [8, 6, 7]

    # 对称差：在 list1 或 list2 中，但不在两个列表中的元素
    # sym_diff = list(set(list1).symmetric_difference(set(list2)))
    # print(sym_diff)  # 输出: [1, 2, 3, 6, 7, 8]

