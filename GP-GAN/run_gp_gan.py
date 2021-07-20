import argparse
import os

import cv2
import tensorflow as tf

from gp_gan import gp_gan
from model import EncoderDecoder
import shutil

import numpy as np

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU computation

basename = lambda path: os.path.splitext(os.path.basename(path))[0]

"""
    Note: source image, destination image and mask image have the same size.
"""


def combine_fg_bg(fg, bg, alpha):
    foreground = fg / 255
    background = bg
    alpha = alpha / 255

    return ((alpha * foreground + (1 - alpha) * background) * 255).astype(np.uint8)


def save_folder_creater(save_dir):
    # check if folder structure exists
    folders = [os.path.join(save_dir, 'results', 'img'), os.path.join(save_dir, 'results', 'mask')]
    for f in folders:
        try:
            os.makedirs(f)
            print(f"Created {f}")
        except FileExistsError:
            # folder already exists
            pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gaussian-Poisson GAN for high-resolution image blending')
    parser.add_argument('--nef', type=int, default=64, help='# of base filters in encoder')
    parser.add_argument('--ngf', type=int, default=64, help='# of base filters in decoder or G')
    parser.add_argument('--nc', type=int, default=3, help='# of output channels in decoder or G')
    parser.add_argument('--nBottleneck', type=int, default=4000, help='# of output channels in encoder')
    parser.add_argument('--ndf', type=int, default=64, help='# of base filters in D')

    parser.add_argument('--image_size', type=int, default=64, help='The height / width of the input image to network')

    parser.add_argument('--color_weight', type=float, default=1, help='Color weight')
    parser.add_argument('--sigma', type=float, default=0.5,
                        help='Sigma for gaussian smooth of Gaussian-Poisson Equation')
    parser.add_argument('--gradient_kernel', type=str, default='normal', help='Kernel type for calc gradient')
    parser.add_argument('--smooth_sigma', type=float, default=1, help='Sigma for gaussian smooth of Laplacian pyramid')

    parser.add_argument('--generator_path', default=None, help='Path to GAN model checkpoint')

    parser.add_argument('--list_path', default='',
                        help='File for input list in csv format: obj_path;bg_path;mask_path in each line')
    parser.add_argument('--result_folder', default='blending_result', help='Name for folder storing results')
    parser.add_argument('--dataset_dir', default='../basic_approaches/DataBase',
                        help='Directory of Dataset (DataBase) in basic_approaches for GP-GAN')
    parser.add_argument('--save_dir', default='DataBase',
                        help='Directory of Dataset (DataBase) in GP-GAN')

    args = parser.parse_args()

    print('Input arguments:')
    for key, value in vars(args).items():
        print('\t{}: {}'.format(key, value))
    print('')

    save_folder_creater(args.save_dir)

    # Init CNN model
    generator = EncoderDecoder(encoder_filters=args.nef, encoded_dims=args.nBottleneck, output_channels=args.nc,
                               decoder_filters=args.ngf, is_training=False, image_size=args.image_size,
                               scope_name='generator')

    inputdata = tf.placeholder(
        dtype=tf.float32,
        shape=[1, args.image_size, args.image_size, args.nc],
        name='input'
    )

    gan_im_tens = generator(inputdata)

    loader = tf.train.Saver(tf.all_variables())
    sess = tf.Session()

    with sess.as_default():
        loader.restore(sess=sess, save_path=args.generator_path)

    test_list = []
    img_names = os.listdir(os.path.join(args.dataset_dir,'img'))
    for img_name in img_names:
        data_row = [os.path.join(args.dataset_dir,'img', img_name), os.path.join(args.dataset_dir,'mask', img_name.split('.')[0]+'.png'), os.path.join(args.dataset_dir,'gp-gan_predict', 'alpha_mask', img_name.split('.')[0]+'.png')]
        test_list.append(data_row)

    total_size = len(test_list)

    # https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
    # |    name    |  id  |
    # _____________________
    # |  'person'  | 24   |
    person_value = 24

    for idx in range(total_size):
        print('Processing {}/{} ...'.format(idx + 1, total_size))

        # load image
        obj = cv2.cvtColor(cv2.imread(test_list[idx][0], 1), cv2.COLOR_BGR2RGB) / 255
        bg = cv2.cvtColor(cv2.imread(test_list[idx][0], 1), cv2.COLOR_BGR2RGB) / 255
        mask = cv2.imread(test_list[idx][1], 0).astype(obj.dtype)
        mask = np.where(mask == person_value, 1, 0)
        alpha = cv2.imread(test_list[idx][2])

        blended_img = gp_gan(obj, bg, mask, gan_im_tens, inputdata, sess, args.image_size, color_weight=args.color_weight,
                            sigma=args.sigma,
                            gradient_kernel=args.gradient_kernel, smooth_sigma=args.smooth_sigma)
        combine_img = combine_fg_bg(blended_img, bg, alpha)
        img_name = test_list[idx][0].split('/')[-1]
        mask_name = test_list[idx][1].split('/')[-1]

        blended_img_path = os.path.join(args.save_dir, 'results', 'img', img_name)
        mask_path = os.path.join(args.save_dir, 'results', 'mask', mask_name)

        cv2.imwrite(blended_img_path, cv2.cvtColor(combine_img, cv2.COLOR_RGB2BGR))
        shutil.copy(test_list[idx][1], mask_path)