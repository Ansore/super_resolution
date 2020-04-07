import tensorflow as tf
from configs import *

def train_parse_exmp(serial_exmp):
    features = tf.parse_single_example(serial_exmp, features={
        'image_raw': tf.FixedLenFeature([], tf.string)})
    patch = tf.decode_raw(features['image_raw'], tf.uint8)
    patch = tf.reshape(patch, [PATCH_SIZE, PATCH_SIZE, NUM_CHENNELS])
    patch = tf.image.convert_image_dtype(patch, dtype=tf.float32)

    if MAX_RANDOM_BRIGHTNESS > 0:
        patch = tf.image.random_brightness(patch, MAX_RANDOM_BRIGHTNESS)
    if len(RANDOM_CONTRAST_RANGE) == 2:
        patch = tf.image.random_contrast(patch, *RANDOM_CONTRAST_RANGE)
    patch = tf.image.random_flip_left_right(patch)
    high_res_patch = tf.image.random_flip_up_down(patch)

    crop_margin = PATCH_SIZE - LABEL_SIZE
    assert crop_margin >= 0
    if crop_margin > 1:
        high_res_patch = tf.random_crop(patch, [LABEL_SIZE, LABEL_SIZE, NUM_CHENNELS])

    downscale_size = [INPUT_SIZE, INPUT_SIZE]
    resize_nn = lambda: tf.image.resize_nearest_neighbor([high_res_patch], downscale_size, True)
    resize_area = lambda: tf.image.resize_area([high_res_patch], downscale_size, True)
    resize_cubic = lambda: tf.image.resize_bicubic([high_res_patch], downscale_size, True)
    r = tf.random_uniform([], 0, 3, dtype=tf.int32)
    low_res_patch = tf.case({tf.equal(r, 0): resize_nn, tf.equal(r, 1): resize_area}, default=resize_cubic)[0]

    # add jpeg noise to low_res_patch
    if JPEG_NOISE_LEVEL > 0:
        low_res_patch = tf.image.convert_image_dtype(low_res_patch, dtype=tf.uint8, saturate=True)
        jpeg_quality = 100 - 5 * JPEG_NOISE_LEVEL
        jpeg_code = tf.image.encode_jpeg(low_res_patch, quality=jpeg_quality)
        low_res_patch = tf.image.decode_jpeg(jpeg_code)
        low_res_patch = tf.image.convert_image_dtype(low_res_patch, dtype=tf.float32)

    # we must set tensor's shape before doing following processes
    low_res_patch.set_shape([INPUT_SIZE, INPUT_SIZE, NUM_CHENNELS])

    # add noise to low_res_patch
    if GAUSSIAN_NOISE_STD > 0:
        low_res_patch += tf.random_normal(low_res_patch.get_shape(), stddev=GAUSSIAN_NOISE_STD)

    low_res_patch = tf.clip_by_value(low_res_patch, 0, 1.0)
    high_res_patch = tf.clip_by_value(high_res_patch, 0, 1.0)

    return low_res_patch, high_res_patch


def test_parse_exmp(serial_exmp):
    features = tf.parse_single_example(serial_exmp, features={
        'image_raw': tf.FixedLenFeature([], tf.string)})
    patch = tf.decode_raw(features['image_raw'], tf.uint8)
    patch = tf.reshape(patch, [PATCH_SIZE, PATCH_SIZE, NUM_CHENNELS])
    patch = tf.image.convert_image_dtype(patch, dtype=tf.float32)

    crop_margin = PATCH_SIZE - LABEL_SIZE
    offset = tf.convert_to_tensor([crop_margin // 2, crop_margin // 2, 0])
    size = tf.convert_to_tensor([LABEL_SIZE, LABEL_SIZE, NUM_CHENNELS])
    high_res_patch = tf.slice(patch, offset, size)

    downscale_size = [INPUT_SIZE, INPUT_SIZE]
    resize_nn = lambda: tf.image.resize_nearest_neighbor([high_res_patch], downscale_size, True)
    resize_area = lambda: tf.image.resize_area([high_res_patch], downscale_size, True)
    resize_cubic = lambda: tf.image.resize_bicubic([high_res_patch], downscale_size, True)
    r = tf.random_uniform([], 0, 3, dtype=tf.int32)
    low_res_patch = tf.case({tf.equal(r, 0): resize_nn, tf.equal(r, 1): resize_area}, default=resize_cubic)[0]

    # add jpeg noise to low_res_patch
    if JPEG_NOISE_LEVEL > 0:
        low_res_patch = tf.image.convert_image_dtype(low_res_patch, dtype=tf.uint8, saturate=True)
        jpeg_quality = 100 - 5 * JPEG_NOISE_LEVEL
        jpeg_code = tf.image.encode_jpeg(low_res_patch, quality=jpeg_quality)
        low_res_patch = tf.image.decode_jpeg(jpeg_code)
        low_res_patch = tf.image.convert_image_dtype(low_res_patch, dtype=tf.float32)

    # we must set tensor's shape before doing following processes
    low_res_patch.set_shape([INPUT_SIZE, INPUT_SIZE, NUM_CHENNELS])

    # add noise to low_res_patch
    if GAUSSIAN_NOISE_STD > 0:
        low_res_patch += tf.random_normal(low_res_patch.get_shape(), stddev=GAUSSIAN_NOISE_STD)

    low_res_patch = tf.clip_by_value(low_res_patch, 0, 1.0)
    high_res_patch = tf.clip_by_value(high_res_patch, 0, 1.0)

    return low_res_patch, high_res_patch