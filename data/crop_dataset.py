"""
Dataset preprocessing: crop large images into fixed-size patches
"""

import os

from PIL import Image

Image.MAX_IMAGE_PIXELS = 1_000_000_000
import numpy as np


def _write_list_file(list_folder, file_lists):
    """Write train/test/val list files."""
    os.makedirs(list_folder, exist_ok=True)
    for name, items in file_lists.items():
        if items:
            with open(os.path.join(list_folder, f"{name}.txt"), "w") as f:
                f.write("\n".join(items))


def _crop_and_save_patches(
    image, crop_size, img_name, dest_folder, split, list_name, file_lists
):
    """Crop an image into patches, save them, and track filenames in file_lists."""
    h, w = image.shape[:2]
    n_rows = h // crop_size
    n_cols = w // crop_size

    for i in range(n_rows):
        for j in range(n_cols):
            patch = image[
                crop_size * i : crop_size * (i + 1),
                crop_size * j : crop_size * (j + 1),
            ]
            new_name = f"{img_name}_{i}_{j}.png"
            dest_path = os.path.join(dest_folder, new_name)
            os.makedirs(dest_folder, exist_ok=True)
            Image.fromarray(patch).save(dest_path)

            if list_name:
                file_lists[list_name].append(new_name)


def crop_levir(ori_folder, cropped_folder):
    """
    Crop LEVIR-CD dataset to 256x256 patches.

    Input structure:
        LEVIR-CD/{train,test,val}/{A,B,label}/*.png

    Output structure:
        LEVIR-Cropped/{A,B,label}/*.png
        LEVIR-Cropped/list/{train,test,val}.txt
    """
    ori_size = 1024
    crop_size = 256
    file_lists = {"train": [], "test": [], "val": []}

    for split in os.listdir(ori_folder):
        split_path = os.path.join(ori_folder, split)
        if not os.path.isdir(split_path):
            continue

        for folder in os.listdir(split_path):
            folder_path = os.path.join(split_path, folder)
            if not os.path.isdir(folder_path):
                continue

            is_label = folder == "label"
            for img in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img)
                if is_label:
                    original_image = np.array(Image.open(img_path), dtype=np.uint8)
                else:
                    original_image = np.asarray(Image.open(img_path).convert("RGB"))

                img_base = img.split(".png")[0]
                dest_folder = os.path.join(cropped_folder, folder)
                list_name = split if is_label else None

                _crop_and_save_patches(
                    original_image,
                    crop_size,
                    img_base,
                    dest_folder,
                    split,
                    list_name,
                    file_lists,
                )

    _write_list_file(os.path.join(cropped_folder, "list"), file_lists)


def crop_s2looking(ori_folder, cropped_folder):
    """
    Crop S2Looking dataset to 256x256 patches.

    Input structure:
        S2Looking/{train,test,val}/{Image1,Image2,label}/*.png

    Output structure:
        S2Looking-Cropped/{A,B,label}/*.png
        S2Looking-Cropped/list/{train,test,val}.txt
    """
    folder_mapping = {"Image1": "A", "Image2": "B", "label": "label"}
    crop_size = 256
    file_lists = {"train": [], "test": [], "val": []}

    for split in os.listdir(ori_folder):
        split_path = os.path.join(ori_folder, split)
        if not os.path.isdir(split_path):
            continue

        for src_folder, dest_name in folder_mapping.items():
            folder_path = os.path.join(split_path, src_folder)
            if not os.path.isdir(folder_path):
                continue

            is_label = src_folder == "label"
            for img in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img)
                if is_label:
                    original_image = np.array(Image.open(img_path), dtype=np.uint8)
                else:
                    original_image = np.asarray(Image.open(img_path).convert("RGB"))

                img_base = img.split(".png")[0]
                dest_folder = os.path.join(cropped_folder, dest_name)
                list_name = split if is_label else None

                _crop_and_save_patches(
                    original_image,
                    crop_size,
                    img_base,
                    dest_folder,
                    split,
                    list_name,
                    file_lists,
                )

    _write_list_file(os.path.join(cropped_folder, "list"), file_lists)


def crop_whu(ori_folder, cropped_folder):
    """
    Crop WHU-CD dataset to 256x256 patches.

    Input structure:
        WHU/{2012,2016}/{train,test}/*.tif
        WHU/change_label/{train,test}/change_label.tif

    Output structure:
        WHU-Cropped/{A,B,label}/*.png
        WHU-Cropped/list/{train,test}.txt
    """
    full_size_pre_folder = os.path.join(ori_folder, "2012")
    full_size_post_folder = os.path.join(ori_folder, "2016")
    full_size_label_folder = os.path.join(ori_folder, "change_label")
    crop_label_folder = os.path.join(cropped_folder, "label")
    crop_pre_folder = os.path.join(cropped_folder, "A")
    crop_post_folder = os.path.join(cropped_folder, "B")

    for d in [crop_label_folder, crop_pre_folder, crop_post_folder]:
        os.makedirs(d, exist_ok=True)

    train_list = []
    test_list = []
    cH, cW = 256, 256

    for split in ["train", "test"]:
        pre_file_path = os.path.join(full_size_pre_folder, split, f"2012_{split}.tif")
        post_file_path = os.path.join(full_size_post_folder, split, f"2016_{split}.tif")
        label_file_path = os.path.join(
            full_size_label_folder, split, "change_label.tif"
        )

        label = np.array(Image.open(label_file_path), dtype=np.uint8)
        img_pre = np.asarray(Image.open(pre_file_path).convert("RGB"))
        img_post = np.asarray(Image.open(post_file_path).convert("RGB"))

        H, W = label.shape
        total_W = (W // cW + 1) if split == "train" else (W // cW)
        total_H = H // cH + 1

        total = 0

        for i in range(total_H):
            for j in range(total_W):
                is_last_h = i == total_H - 1
                is_last_w = j == total_W - 1

                if is_last_h and is_last_w and split == "train":
                    crop_pre = img_pre[
                        cH * i - 6 : cH * (i + 1), cW * j - 5 : cW * (j + 1)
                    ]
                    crop_post = img_post[
                        cH * i - 6 : cH * (i + 1), cW * j - 5 : cW * (j + 1)
                    ]
                    crop_label = label[
                        cH * i - 6 : cH * (i + 1), cW * j - 5 : cW * (j + 1)
                    ]
                elif is_last_h:
                    crop_pre = img_pre[cH * i - 6 : cH * (i + 1), cW * j : cW * (j + 1)]
                    crop_post = img_post[
                        cH * i - 6 : cH * (i + 1), cW * j : cW * (j + 1)
                    ]
                    crop_label = label[cH * i - 6 : cH * (i + 1), cW * j : cW * (j + 1)]
                elif is_last_w and split == "train":
                    crop_pre = img_pre[cH * i : cH * (i + 1), cW * j - 5 : cW * (j + 1)]
                    crop_post = img_post[
                        cH * i : cH * (i + 1), cW * j - 5 : cW * (j + 1)
                    ]
                    crop_label = label[cH * i : cH * (i + 1), cW * j - 5 : cW * (j + 1)]
                else:
                    crop_pre = img_pre[cH * i : cH * (i + 1), cW * j : cW * (j + 1)]
                    crop_post = img_post[cH * i : cH * (i + 1), cW * j : cW * (j + 1)]
                    crop_label = label[cH * i : cH * (i + 1), cW * j : cW * (j + 1)]

                crop_label[crop_label == 1] = 255

                assert crop_pre.shape[:2] == (256, 256), (
                    f"Unexpected crop size: {crop_pre.shape}"
                )
                assert crop_post.shape[:2] == (256, 256), (
                    f"Unexpected crop size: {crop_post.shape}"
                )
                assert crop_label.shape == (256, 256), (
                    f"Unexpected label size: {crop_label.shape}"
                )

                new_name = f"{split}_{total}.png"

                Image.fromarray(crop_pre).save(os.path.join(crop_pre_folder, new_name))
                Image.fromarray(crop_post).save(
                    os.path.join(crop_post_folder, new_name)
                )
                Image.fromarray(crop_label).save(
                    os.path.join(crop_label_folder, new_name)
                )

                if split == "train":
                    train_list.append(new_name)
                else:
                    test_list.append(new_name)
                total += 1

    _write_list_file(
        os.path.join(cropped_folder, "list"),
        {"train": train_list, "test": test_list},
    )


if __name__ == "__main__":
    ori_folder = "/home/dhm/dataset/S2Looking/S2Looking"
    cropped_folder = "/home/dhm/dataset/S2Looking_cropped"

    # crop_levir(ori_folder=ori_folder, cropped_folder=cropped_folder)
    # crop_s2looking(ori_folder=ori_folder, cropped_folder=cropped_folder)

    whu_ori_folder = "/home/dhm/dataset/whucd"
    whu_cropped_folder = "/home/dhm/dataset/whucd_cropped"
    crop_whu(ori_folder=whu_ori_folder, cropped_folder=whu_cropped_folder)
