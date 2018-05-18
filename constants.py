INPUT_SIZE = [128, 128]

NUM_LABELS = 66

DOWNCONV_FILTERS = [(10, [5, 5], 1), (64, [5, 5], 3), (64, [5, 5], 3)]

UPCONV_FILTERS = [(64, [5, 5], 3), (64, [5, 5], 3), (64, [5, 5], 3), (NUM_LABELS, [5, 5], 1)]

VAL_SIZE = 1800

SYLVESTER = False

MB_SIZE = 8

SAVED_MODEL_PATH = 'tmp/model.ckpt'
