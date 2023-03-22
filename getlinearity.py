from front_locations import *
import math as m
import pandas as pd
from linearity import get_linearity

simulation_folder = 'Z:\\hirst\\bjornssimies\\delta_k\\'
simulation = 'Para_1-6225ft_PHI_48_THETA_134'
print(get_linearity(simulation_folder,simulation))