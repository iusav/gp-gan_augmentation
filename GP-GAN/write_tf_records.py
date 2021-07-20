import argparse
import os, glob
import tensorflow as tf
import cv2


def _int64_feature(value):
    """
    Wrapper for inserting int64 features into Example proto.
    :param value:
    :return:
    """
    if not isinstance(value, list):
        value = [value]
    value_tmp = []
    is_int = True
    for val in value:
        if not isinstance(val, int):
            is_int = False
            value_tmp.append(int(float(val)))
    if not is_int:
        value = value_tmp
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    """
    Wrapper for inserting float features into Example proto.
    :param value:
    :return:
    """
    if not isinstance(value, list):
        value = [value]
    value_tmp = []
    is_float = True
    for val in value:
        if not isinstance(val, int):
            is_float = False
            value_tmp.append(float(val))
    if is_float is False:
        value = value_tmp
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    """
    Wrapper for inserting bytes features into Example proto.
    :param value:
    :return:
    """
    if not isinstance(value, bytes):
        if not isinstance(value, list):
            value = value.encode('utf-8')
        else:
            value = [val.encode('utf-8') for val in value]
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def create_tf_example(encoded_image_data_cp, encoded_image_data_bg):

    image_format = b'jpg'

    height, width, _ = encoded_image_data_cp.shape
    encoded_image_data_cp = encoded_image_data_cp.tostring()
    encoded_image_data_bg = encoded_image_data_bg.tostring()

    feature_dict = {
        'image/height': _int64_feature(height),
        'image/width': _int64_feature(width),
        'image/encoded_cp': _bytes_feature(encoded_image_data_cp),
        'image/encoded_bg': _bytes_feature(encoded_image_data_bg),
        'image/format': _bytes_feature(image_format)
    }

    tf_example = tf.train.Example(features=tf.train.Features(feature=feature_dict))

    return tf_example


def create_copy_pastes(src_paths, writer, process_name):
    dataset_size = len(src_paths)
    id_data = 1
    print(" %s start " % process_name)
    for src_path in src_paths:
        src_name = src_path.split('/')[-1]
        dst_path = os.path.join(dataset_dir, 'dst', src_name)

        src = cv2.cvtColor(cv2.imread(src_path), cv2.COLOR_BGR2RGB)
        dst = cv2.cvtColor(cv2.imread(dst_path), cv2.COLOR_BGR2RGB)

        tf_example = create_tf_example(src, dst)
        writer.write(tf_example.SerializeToString())

        print("------------------- %s rest --------------------" % (dataset_size - id_data))
        id_data += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tensorflow records writer')

    parser.add_argument('--dataset_dir', default='DataBase/train_data',
                        help='Directory of created training Dataset AND to save tfrecords')
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    out_path_train = args.dataset_dir+'/train.tfrecords'
    out_path_val = args.dataset_dir+'/val.tfrecords'

    val_ratio = 0.2

    writer_train = tf.python_io.TFRecordWriter(out_path_train)
    writer_val = tf.python_io.TFRecordWriter(out_path_val)

    src_paths = glob.glob(os.path.join(dataset_dir, 'src', '*.jpg'))

    val_end = int(val_ratio * len(src_paths))

    create_copy_pastes(src_paths[val_end:], writer_train, 'Train')
    writer_train.close()

    create_copy_pastes(src_paths[:val_end], writer_val, 'Val')
    writer_val.close()