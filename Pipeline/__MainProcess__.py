# ----------------------------------------------------------------- #
# About script
# ----------------------------------------------------------------- #

# ----------------------------------------------------------------- #
# Preambule
# ----------------------------------------------------------------- #

import numpy as np
import matplotlib.pyplot as plt
from Class_Process import Brain_process
import warnings
warnings.filterwarnings("ignore")

# ----------------------------------------------------------------- #
# Run Class
# ----------------------------------------------------------------- #

params_input = {}
Class = Brain_process(params_input)
Class.load_files()
Class.select_master_clusters()
#Class.plot_preselection_network()
Class.acquire_locations()
Class.save_processing()
