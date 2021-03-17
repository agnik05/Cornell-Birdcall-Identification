import warnings
warnings.filterwarnings('ignore')

import src.utils as utils
from src.datasets import CornellDataset
from src.models.sed import PANNsCNN14Att


if __name__ == '__main__':

    utils.set_seed(config['seed'])