import random
import numpy as np
from config import SEED

def seed_everything():
    random.seed(SEED)
    np.random.seed(SEED)

import warnings
import pandas as pd
from config import IGNORE_WARNINGS, PANDAS_FLOAT_FORMAT

def setup_environment():
    if IGNORE_WARNINGS:
        warnings.filterwarnings('ignore')
        
    pd.options.display.float_format = PANDAS_FLOAT_FORMAT
