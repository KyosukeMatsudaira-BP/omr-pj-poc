from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import torch

class CustomDataset(Dataset):
    def __init__(self, img_dir, annotation_dir, transform=None):
        self.img_dir = img_dir
        self.annotation_dir = annotation_dir
        # デフォルトの変換にToTensorを追加
        self.transform = transforms.Compose([
            transforms.Resize((576, 576)),
            transforms.ToTensor(),
        ]) if transform is None else transform

        self.img_files = os.listdir(img_dir)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        # 画像を読み込む
        img_path = os.path.join(self.img_dir, self.img_files[index])
        image = Image.open(img_path).convert("RGB")

        # アノテーションファイルを読み込む
        annotation_file = self.img_files[index].replace('.jpg', '.txt').replace('.png', '.txt')
        annotation_dir = os.path.join(self.annotation_dir, annotation_file)

        boxes = []
        with open(annotation_dir, 'r') as file:
            for line in file.readlines():
                class_label, x_center, y_center, width, height = [
                    float(x) for x in line.replace('\n', '').split()
                ]
                boxes.append([index, class_label, x_center, y_center, width, height])

        boxes = torch.tensor(boxes)

        if self.transform:
            image = self.transform(image)

        return image, boxes

def custom_collate_fn(batch):
    images = torch.stack([item[0] for item in batch], dim=0)
    bboxes = torch.cat([item[1] for item in batch], dim=0)

    # 画像のindexを振り直し
    dict_map = {element.item(): i for i, element in enumerate(torch.unique(bboxes[:, 0]))}
    for b in bboxes:
        b[0] = dict_map[b[0].item()]

    return images, bboxes