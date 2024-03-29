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

# for sub in ['279418', '279419', '339295', '339296', '340233', '341319',
#             '341321', '346110', '346111', '347640', '347641']:
for sub in ['279418', '279419', '339295', '339296', '340233', '341319',
            '341321', '346110', '346111', '347640', '347641']:
    for reg in ['HIP', 'PFC', 'PAR']:
        for st in ['TRAINING']:
            params_input = {'SUBPART': 'NO',
                            'REGION': reg,
                            'TYPE': st,
                            'SUBJECT': sub}
            Class = Brain(params_input)
            Class.preprocess()
            Class.determine_residual_series()
            Class.save_to_file()
