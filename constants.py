INPUT_SIZE = [512, 640]

NUM_LABELS = 66

DOWNCONV_FILTERS = [(20, [5, 5], 1), (64, [5, 5], 1), (64, [5, 5], 1)]

UPCONV_FILTERS = [(64, [5, 5], 1), (64, [5, 5], 1), (64, [5, 5], 1), (NUM_LABELS, [5, 5], 1)]

VAL_SIZE = 1800

SYLVESTER = False

MB_SIZE = 2

MAX_CROP_MARGIN = 0.25

SAVED_MODEL_PATH = 'tmp/model.ckpt'

LABELS_BY_PROBABILITY = [27, 30, 13, 17, 55, 15, 16, 65, 3, 29, 45, 24, 23, 2, 64, 6, 35, 50, 61, 47, 8, 54, 19, 25, 5, 4, 46, 12, 48, 28, 10, 51, 14, 9, 20, 49, 21, 7, 57, 32, 58, 52, 63, 44, 41, 39, 11, 36, 1, 18, 42, 40, 33, 34, 38, 53, 0, 37, 59, 62, 56, 31, 43, 22, 60, 26]

POOL_SIZE = 4
