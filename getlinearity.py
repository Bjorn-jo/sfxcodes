from front_locations import *
import math as m
import pandas as pd
from linearity import get_linearity

simulation_folder = 'C:\\Users\\Bjorn\\Desktop\\sfx\\1_45targeting\\2frontscurrent\\'
simulation = 'Para_5ft_PHI_54_THETA_340'
print(get_linearity(simulation_folder,simulation))