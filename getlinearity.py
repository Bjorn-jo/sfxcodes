from front_locations import *
import math as m
import pandas as pd
from linearity import get_linearity

simulation_folder = 'E:\\sfx_1_33_targeting\\linearitycomp\\deltak\\'
simulation = 'Para_4-363517ft_PHI_46_THETA_233'
print(get_linearity(simulation_folder,simulation))