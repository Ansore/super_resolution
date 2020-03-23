import tensorflow as tf
import numpy as np
import cv2 as cv
from configs import *
import model


def main():
    ckpt_state = tf.train.get_checkpoint_state(CHECKPOINTS_PATH)
    if not ckpt_state or not ckpt_state.model_checkpoint_path:
        print('No check point files are found!')
        return

    ckpt_files = ckpt_state.all_model_checkpoint_paths
    num_ckpt = len(ckpt_files)
    if num_ckpt < 1:
        print('No check point files are found!')
        return

    low_res_holder = tf.placeholder(tf.float32, shape=[1, INPUT_SIZE, INPUT_SIZE, NUM_CHENNELS])
    logits = model.inference(low_res_holder)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # load model
    saver = tf.train.Saver(tf.global_variables())
    saver.restore(sess, ckpt_files[-1])

    low_res_img = cv.imread('./images/small.png')

    output_size = int(logits.get_shape()[1])
    input_size = INPUT_SIZE
    available_size = output_size // SCALE_FACTOR
    margin = (input_size - available_size) // 2

    img_rows = low_res_img.shape[0]
    img_cols = low_res_img.shape[1]
    img_chns = low_res_img.shape[2]

    padded_rows = int(img_rows / available_size + 1) * available_size + margin * 2
    padded_cols = int(img_cols / available_size + 1) * available_size + margin * 2
    padded_low_res_img = np.zeros((padded_rows, padded_cols, img_chns), dtype=np.uint8)
    padded_low_res_img[margin: margin + img_rows, margin: margin + img_cols, ...] = low_res_img
    padded_low_res_img = padded_low_res_img.astype(np.float32)
    padded_low_res_img /= 255
    # padded_low_res_img -= 0.5

    high_res_img = np.zeros((padded_rows * SCALE_FACTOR, padded_cols * SCALE_FACTOR, img_chns), dtype=np.float32)
    low_res_patch = np.zeros((1, input_size, input_size, img_chns), dtype=np.float32)
    for i in range(margin, margin + img_rows, available_size):
        for j in range(margin, margin + img_cols, available_size):
            low_res_patch[0, ...] = padded_low_res_img[i - margin: i - margin + input_size, j - margin: j - margin + input_size, ...]
            high_res_patch = sess.run(logits, feed_dict={low_res_holder: low_res_patch})

            out_rows_begin = (i - margin) * SCALE_FACTOR
            out_rows_end = out_rows_begin + output_size
            out_cols_begin = (j - margin) * SCALE_FACTOR
            out_cols_end = out_cols_begin + output_size
            high_res_img[out_rows_begin: out_rows_end, out_cols_begin: out_cols_end, ...] = high_res_patch[0, ...]

    # high_res_img += 0.5
    high_res_img = tf.image.convert_image_dtype(high_res_img, tf.uint8, True)

    high_res_img = high_res_img[:SCALE_FACTOR * img_rows, :SCALE_FACTOR * img_cols, ...]
    cv.imwrite('./images/enhance.png', high_res_img.eval(session=sess))

    print('Super Resolution Finished!')

if __name__ == '__main__':
    main()
