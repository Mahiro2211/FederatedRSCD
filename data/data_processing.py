import os
import numpy as np
from PIL import Image
from torch.utils import data
from data.data_augment import CDDataAugmentation

IMG_FOLDER_NAME = "A"
IMG_POST_FOLDER_NAME = "B"
LIST_FOLDER_NAME = "list"
ANNOT_FOLDER_NAME = "label"
label_suffix = ".png"


def load_img_name_list(dataset_path):
    img_name_list = np.loadtxt(dataset_path, dtype=str)
    return img_name_list[:, 0] if img_name_list.ndim == 2 else img_name_list


def get_img_path(root_dir, img_name):
    return os.path.join(root_dir, IMG_FOLDER_NAME, img_name)


def get_img_post_path(root_dir, img_name):
    return os.path.join(root_dir, IMG_POST_FOLDER_NAME, img_name)


def get_label_path(root_dir, img_name):
    return os.path.join(root_dir, ANNOT_FOLDER_NAME, img_name.replace(".jpg", label_suffix))


def load_image(img_path):
    return np.asarray(Image.open(img_path).convert("RGB"))


class ImageDataset(data.Dataset):

    def __init__(self, root_dir, split="train", img_size=256, is_train=True, to_tensor=True):
        super().__init__()
        self.root_dir = root_dir
        self.img_size = img_size
        self.split = split
        self.list_path = os.path.join(self.root_dir, LIST_FOLDER_NAME, f"{split}.txt")
        self.img_name_list = load_img_name_list(self.list_path)
        self.A_size = len(self.img_name_list)
        self.to_tensor = to_tensor
        
        augm_params = {"img_size": self.img_size} if not is_train else {
            "img_size": self.img_size,
            "with_random_hflip": True,
            "with_random_vflip": True,
            "with_scale_random_crop": True,
            "with_random_blur": True,
            "random_color_tf": True,
        }
        self.augm = CDDataAugmentation(**augm_params)

    def __getitem__(self, index):
        img_name = self.img_name_list[index]
        img = load_image(get_img_path(self.root_dir, img_name))
        img_B = load_image(get_img_post_path(self.root_dir, img_name))
        [img, img_B], _ = self.augm.transform([img, img_B], [], to_tensor=self.to_tensor)
        return {"A": img, "B": img_B, "name": img_name}

    def __len__(self):
        return self.A_size


class CDDataset(ImageDataset):
    def __init__(self, root_dir, img_size, split="train", is_train=True, label_transform=None, to_tensor=True):
        super().__init__(root_dir, split=split, img_size=img_size, is_train=is_train, to_tensor=to_tensor)
        self.label_transform = label_transform

    def __getitem__(self, index):
        img_name = self.img_name_list[index]
        img = load_image(get_img_path(self.root_dir, img_name))
        img_B = load_image(get_img_post_path(self.root_dir, img_name))
        label = np.array(Image.open(get_label_path(self.root_dir, img_name)), dtype=np.uint8)
        
        if self.label_transform == "norm":
            label = label // 255
        
        [img, img_B], [label] = self.augm.transform([img, img_B], [label], to_tensor=self.to_tensor)
        return {"name": img_name, "A": img, "B": img_B, "L": label}


if __name__ == "__main__":
    pp = "/home/dhm/dataset/levir_cropped"
    dataset = CDDataset(root_dir=pp, img_size=256, is_train=False, split="val")
    print(len(dataset))
    item = dataset[0]
    print(item)
