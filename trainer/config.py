TRAIN_PATH = './train_data'
VAL_PATH = './val_data'
MODEL_PATH = './model'

# Set to an int to cap training shards (e.g. 20 ≈ 3.6M rows). None = all shards.
MAX_TRAIN_SHARDS = None

# Val shards to load (more = smoother val curve, higher RAM). Default 20.
MAX_VAL_SHARDS = 20
