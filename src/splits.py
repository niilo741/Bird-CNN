import hashlib

# Never change these after the first real experiment: doing so moves recordings between splits.
TEST_PCT = 15
VAL_PCT = 15


def split_for(xc_id):
    """Permanently assign a recording to train/val/test based on a hash of its ID.

    The same recording always lands in the same split, no matter how much data is added,
    so test scores stay comparable between dataset versions.
    """
    h = int(hashlib.md5(str(xc_id).encode()).hexdigest(), 16) % 100
    if h < TEST_PCT:
        return "test"
    if h < TEST_PCT + VAL_PCT:
        return "val"
    return "train"
