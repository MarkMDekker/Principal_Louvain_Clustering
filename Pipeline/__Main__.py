# ----------------------------------------------------------------- #
# About script
# ----------------------------------------------------------------- #

# ----------------------------------------------------------------- #
# Preambule
# ----------------------------------------------------------------- #

import numpy as np
import matplotlib.pyplot as plt
from Class_Brain import Brain
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# ----------------------------------------------------------------- #
# Run Class
# ----------------------------------------------------------------- #

params_input = {'SUBPART': 'NO'}
Class = Brain(params_input)
Class.preprocess()
Class.determine_residual_series()
#Class.get_from_file()
Class.pca()
#Class.grid_rate_matrix()
#Class.clustering()
Class.save_to_file()
