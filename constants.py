INPUT_SIZE = [128, 128]

NUM_LABELS = 66

DOWNCONV_FILTERS = [(10, [5, 5]), (64, [5, 5]), (64, [5, 5]), (64, [5, 5])]

UPCONV_FILTERS = [(64, [5, 5]), (64, [5, 5]), (64, [5, 5]), (64, [5, 5]), (NUM_LABELS, [5, 5])]

VAL_SIZE = 1800

SYLVESTER = False

MB_SIZE = 4

SAVED_MODEL_PATH = 'tmp/model.ckpt'
