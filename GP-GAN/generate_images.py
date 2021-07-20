import argparse
import cv2
import numpy as np
import os, glob
import time
import json
from tqdm import tqdm
import random


# Creating of data path list with "person" class
def path_creater(json_path):
    data_name = '_'.join(json_path.split('/')[-1].split('_')[:-2])
    mask_dir = '/'.join(json_path.split('/')[:-1])
    img_dir = mask_dir.replace('gtFine_trainvaltest/gtFine', 'leftImg8bit_trainvaltest/leftImg8bit')
    mask_path = os.path.join(mask_dir, str(data_name) + '_gtFine_labelIds.png')
    img_path = os.path.join(img_dir, str(data_name) + '_leftImg8bit.png')

    return img_path, mask_path


def path_list_creater():
    data_counter = 0
    class_data_counter = 0

    class_data_list = []
    json_paths = glob.glob(os.path.join(dataset_dir, 'gtFine_trainvaltest', 'gtFine', '*', '*', '*.json'))

    for json_path in tqdm(json_paths, total=len(json_paths)):
        with open(json_path) as json_file:
            json_data = json.load(json_file)
            obj_key = 'objects'
            if obj_key in json_data:
                class_list = [label['label'] for label in json_data[obj_key]]
                if 'person' in class_list:
                    img_path, mask_path = path_creater(json_path)
                    act_data = [img_path, mask_path]
                    class_data_list.append(act_data)
                    class_data_counter += 1

        data_counter += 1

    print('%s images with class "Person" in the dataset' % (class_data_counter))
    print("The cityscapes dataset consists of %s images" % (data_counter))

    return class_data_list


# Dataset creating
def data_name(path):
    name = '_'.join(path[1].split('/')[-1].split('_')[:-2])

    return name


def data_loader(path):
    img_path = path[0];
    mask_path = path[1]

    img = cv2.imread(img_path);
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_path);
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

    return img, mask


def data_saver(data_name, dst, src, id_data):
    dst = cv2.cvtColor(dst, cv2.COLOR_RGB2BGR)
    src = cv2.cvtColor(src, cv2.COLOR_RGB2BGR)

    dst_path = os.path.join(save_dir, 'train_data', 'dst', data_name + '_' + str(id_data) + '.jpg')
    src_path = os.path.join(save_dir, 'train_data', 'src', data_name + '_' + str(id_data) + '.jpg')

    cv2.imwrite(dst_path, dst)
    cv2.imwrite(src_path, src)


# Augmentation
def brightness_contrast(img, hBrightness, lBrightness, hContrast, lContrast):
    brightness = random.randint(lBrightness, hBrightness)
    contrast = random.randint(lContrast, hContrast)
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow

        buf = cv2.addWeighted(img, alpha_b, img, 0, gamma_b)
    else:
        buf = img.copy()

    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)

        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf


def srcImg_creater(img, mask):
    thresh_mask = np.where(mask == person_value, 255, 0).astype(np.uint8)
    gray_mask = cv2.cvtColor(thresh_mask, cv2.COLOR_RGB2GRAY)

    obj_contours, obj_hierarchy = cv2.findContours(gray_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    obj_areas = np.array([cv2.contourArea(obj_contour) for obj_contour in obj_contours]).astype(np.int)

    if any(obj_areas):
        person_ids = np.where(obj_areas > 0)[0].astype(np.int)

        new_img = img.copy()
        for person_id in person_ids:
            obj_mask = np.full((mask.shape[0], mask.shape[1]), 0, np.uint8)

            obj_mask = cv2.drawContours(obj_mask, obj_contours, contourIdx=person_id, color=255, thickness=-1)
            obj_mask = cv2.cvtColor(obj_mask, cv2.COLOR_GRAY2RGB)

            augment_img = brightness_contrast(img,
                                              hBrightness=highBrightness_value,
                                              lBrightness=lowBrightness_value,
                                              hContrast=highContrast_value,
                                              lContrast=lowContrast_value)
            new_img = np.where((obj_mask == 255), augment_img, new_img)

        return new_img
    else:
        return 'None'


def save_folder_creater(save_dir):
    # check if folder structure exists
    folders = [os.path.join(save_dir, 'train_data', 'dst'), os.path.join(save_dir, 'train_data', 'src'), os.path.join(save_dir, 'results', 'img'), os.path.join(save_dir, 'results', 'mask')]
    for f in folders:
        try:
            os.makedirs(f)
            print(f"Created {f}")
        except FileExistsError:
            # folder already exists
            pass


def data_resizer(Img, Mask, shape):
    w = shape
    h = shape
    Img = cv2.resize(Img, (w, h), interpolation=cv2.INTER_AREA)
    Mask = cv2.resize(Mask, (w, h), interpolation=cv2.INTER_NEAREST)

    return Img, Mask


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Source & destination image creating')

    parser.add_argument('--dataset_size', default=51,
                        help='Size of source & destination image dataset')
    parser.add_argument('--img_shape', default=64,
                        help='image shape')
    parser.add_argument('--dataset_dir', default='CITYSCAPES_DATASET',
                        help='Directory of Cityscapes Dataset')
    parser.add_argument('--save_dir', default='DataBase',
                        help='Directory to save images')
    parser.add_argument('--highBrightness_value', default=50,
                        help='High brightness value')
    parser.add_argument('--lowBrightness_value', default=-50,
                        help='Low brightness value')

    parser.add_argument('--highContrast_value', default=30,
                        help='High contrast value')
    parser.add_argument('--lowContrast_value', default=-30,
                        help='Low contrast value')

    args = parser.parse_args()

    dataset_size = int(args.dataset_size)
    img_shape = int(args.img_shape)
    dataset_dir = args.dataset_dir
    save_dir = args.save_dir
    highBrightness_value = int(args.highBrightness_value)
    lowBrightness_value = int(args.lowBrightness_value)
    highContrast_value = int(args.highContrast_value)
    lowContrast_value = int(args.lowContrast_value)


    # https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
    # |    name    |  id  |
    # _____________________
    # |  'person'  | 24   |
    person_value = 24

    id_data = 1

    save_folder_creater(save_dir)
    img_paths = path_list_creater()

    start_time = time.time()
    while (id_data <= dataset_size):
        try:
            currentImg_path = random.choice(img_paths)

            # Data name chosing
            imgName = data_name(currentImg_path)

            # Data loading
            dstImg, dstMask = data_loader(currentImg_path)

            # Data resizing
            dstImg, dstMask = data_resizer(dstImg, dstMask, img_shape)

            # Source image creating
            srcImg = srcImg_creater(dstImg, dstMask)
            if srcImg is 'None':
                continue

            # Data saving
            data_saver(imgName, dstImg, srcImg, id_data)

            print("------------------- %s rest --------------------" % (dataset_size - id_data))
            id_data += 1
        except:
            print("!!! ERROR   %s   ERROR !!!" % currentImg_path)
            continue

    print("----------------- %s seconds ----------------" % (round((time.time() - start_time), 2)))






