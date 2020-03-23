from configs import *
import os
import numpy as np
from PIL import Image
import tensorflow as tf

def scan_file(path):
    result = []
    for filename in os.listdir(path):
        if filename.endswith('.png'):
            filename = path + '/' + filename
            result.append(filename)
    return result

def image_to_example(image_path):
    img = Image.open(image_path, 'r')
    # 将图片转化为二进制格式
    img_raw = img.tobytes()
    example = tf.train.Example(
        features=tf.train.Features(feature={
            'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        }))
    return example

def main():
    train_tfrecord_path = TFRECORD_PATH + '/train.tfrecord'
    test_tfrecord_path = TFRECORD_PATH + '/test.tfrecord'
    train_writer = tf.python_io.TFRecordWriter(train_tfrecord_path)
    test_writer = tf.python_io.TFRecordWriter(test_tfrecord_path)

    train_file_list = scan_file(TRAINING_DATA_PATH)
    test_file_list = scan_file(TESTING_DATA_PATH)

    for train_file in train_file_list:
        example = image_to_example(train_file)
        # 序列化为字符串
        train_writer.write(example.SerializeToString())
    train_writer.close()

    for test_file in test_file_list:
        example = image_to_example(test_file)
        # 序列化为字符串
        test_writer.write(example.SerializeToString())
    test_writer.close()


if __name__ == '__main__':
    main()