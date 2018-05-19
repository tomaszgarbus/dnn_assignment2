INPUT_SIZE = [256, 256]

NUM_LABELS = 66

DOWNCONV_FILTERS = [(10, [5, 5], 1), (64, [5, 5], 1), (64, [5, 5], 1)]

UPCONV_FILTERS = [(64, [5, 5], 1), (64, [5, 5], 1), (64, [5, 5], 1), (NUM_LABELS, [5, 5], 1)]

VAL_SIZE = 1800

SYLVESTER = False

MB_SIZE = 4

SAVED_MODEL_PATH = 'tmp/model.ckpt'
