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
    # we still need to initialize all variables even when we use Saver's restore method.
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver(tf.global_variables())
    saver.restore(sess, ckpt_files[-1])  # load the lateast model
    low_res_patch = np.zeros((1, INPUT_SIZE, INPUT_SIZE, NUM_CHENNELS), dtype=np.float32)
    high_res_patch = sess.run(logits, feed_dict={low_res_holder: low_res_patch})

    print(high_res_patch)

    print('Enhance Finished!')

if __name__ == '__main__':
    main()
