import os

from matplotlib.dates import AutoDateFormatter

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(ROOT_DIR, 'datasets')

H36_DIR = os.path.join(DATASET_DIR, "H36Pose")

LOG_DIR = os.path.join(ROOT_DIR, 'miscellaneous/logs')

CHECKPOINT_DIR = os.path.join(ROOT_DIR, 'miscellaneous/checkpoints')

STATE_SPACE_JOINTS_DIR = os.path.join(CHECKPOINT_DIR, 'Joint_Based/StateSpace')

AUTOREGRESSIVE_JOINTS_DIR = os.path.join(CHECKPOINT_DIR, "Joint_Based/Autoregressive")

SEQ2SEQ_JOINTS_DIR = os.path.join(CHECKPOINT_DIR, "Joint_Based/Seq2Seq")

LSTM_GAN = os.path.join(CHECKPOINT_DIR, "Joint_Based/LSTM_GAN/")

HM_STATE_SPACE_JOINTS_DIR = os.path.join(CHECKPOINT_DIR, 'HM_Based/StateSpace')

HM_AUTOREGRESSIVE_JOINTS_DIR = os.path.join(CHECKPOINT_DIR, "HM_Based/Autoregressive")

HM_SEQ2SEQ_JOINTS_DIR = os.path.join(CHECKPOINT_DIR, "HM_Based/Seq2Seq")