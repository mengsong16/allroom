import yaml
import os
import logging
import numpy as np

cur_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.join(cur_path[:cur_path.find("/allroom")], "allroom")
runs_path = os.path.join(root_path, "runs")
config_path = os.path.join(root_path, "configs")
figure_path = os.path.join(root_path, "figures")
tensorboard_path = os.path.join(root_path, "tensorboard")