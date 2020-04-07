import numpy as np
import cv2 as cv
import os
import shutil
from configs import *


def clear_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    else:
        old_pairs = os.listdir(path)
        if len(old_pairs) > 0:
            shutil.rmtree(path)
            os.mkdir(path)


def main():
    clear_dir(TRAINING_DATA_PATH)
    clear_dir(TESTING_DATA_PATH)

    assert os.path.exists(ORIGINAL_IMAGES_PATH)
    image_files = os.listdir(ORIGINAL_IMAGES_PATH)
    assert len(image_files) > 0

    random_patch_generate(image_files)

    print('Data generating finished!')


def random_patch_generate(image_files):
    file_id = 0
    for image_file in image_files:
        file_id += 1
        img = cv.imread(os.path.join(ORIGINAL_IMAGES_PATH, image_file))
        if img is None:
            continue
        print('oringinal image: ', image_file)

        rows = img.shape[0]
        cols = img.shape[1]
        # generate training data
        patch_id = 0
        num_patches = max(rows, cols) // PATCH_RAN_GEN_RATIO
        validation_beginning_id = num_patches - 2 * num_patches // 10
        testing_beginning_id = num_patches - num_patches // 10
        while True:
            patch_x = np.random.randint(0, cols - PATCH_SIZE)
            patch_y = np.random.randint(0, rows - PATCH_SIZE)

            high_res_patch = img[patch_y: patch_y + PATCH_SIZE, patch_x: patch_x + PATCH_SIZE, ...]
            patch_std = np.std(high_res_patch, axis=(0, 1), dtype=np.float32)
            if np.max(patch_std) < 3.0:
                continue

            save_name = '%d_%d.png' % (file_id, patch_id)
            # print('saving sub image: ', save_name)

            if patch_id < validation_beginning_id:
                cv.imwrite(os.path.join(TRAINING_DATA_PATH, save_name), high_res_patch)
            elif patch_id < testing_beginning_id:
                cv.imwrite(os.path.join(VALIDATION_DATA_PATH, save_name), high_res_patch)
            else:
                cv.imwrite(os.path.join(TESTING_DATA_PATH, save_name), high_res_patch)

            patch_id += 1
            if patch_id >= num_patches:
                break
