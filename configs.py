# data path and log path
ORIGINAL_IMAGES_PATH  = 'data/originals'
TRAINING_DATA_PATH    = 'data/train'
TESTING_DATA_PATH     = 'data/test'
TFRECORD_PATH         = 'data/tfrecord'
TRAINING_SUMMARY_PATH = 'train_log'
CHECKPOINTS_PATH      = 'checkpoints'
MAX_CKPT_TO_KEEP      = 50

# patch generation
PATCH_SIZE = 80
PATCH_RAN_GEN_RATIO = 2

BATCH_SIZE   = 32
# the image size input to the network
INPUT_SIZE   = 28
SCALE_FACTOR = 2
LABEL_SIZE   = SCALE_FACTOR * INPUT_SIZE
NUM_CHENNELS = 3

# data queue
MIN_QUEUE_EXAMPLES = 1024
NUM_PROCESS_THREADS = 3
NUM_TRAINING_STEPS = 1000000
NUM_TESTING_STEPS = 600

# data argumentation
MAX_RANDOM_BRIGHTNESS = 0.2
RANDOM_CONTRAST_RANGE = [0.8, 1.2]
GAUSSIAN_NOISE_STD = 0.01  # [0...1] (float)
JPEG_NOISE_LEVEL = 2  # [0...4] (int)
