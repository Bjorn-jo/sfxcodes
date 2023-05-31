from abafrank_lib.Function_Library import *
#from R_Curve import R_curve
from itertools import product
from shutil import copyfile
from abafrank_lib.Variables import *
import sys
import os
import time
import signal
import subprocess
import math
from numpy import genfromtxt
import random

try:
    import PyF3D
    import Vec3D
except Exception as e:
    raise e
os.chdir( 'E:\\mitchrecreation\\manual_prop_crack')
print(os.getcwd())
#from Model_Definition import BONE
def Open_fdb ():
    f3d.OpenFdbModel(file_name= 'orthotropicfranc3d1_full.dtp')
    PRINT('maybe opened')
Open_fdb

def Write_INP ():
    #[name,Simulation_Step,U_I] = args
    f3d.RunAnalysis(
        model_type='ABAQUS',
        file_name=name + '_Stp'  + str( Simulation_Step ) + '_UCI_' + str(U_I) + '.fdb',
        flags=['NO_WRITE_TEMP','TRANSFER_BC','NO_CFACE_TRACT','CFACE_CNTCT','FILE_ONLY','MID_SIDE'],
        merge_tol=0.0001,
        connection_type='CONSTRAIN',
        executable=abaqus_bat_bin,
        command=abaqus_bat_bin + ' job=' + name + '_Stp'  + str( Simulation_Step ) + '_UCI_' + str(U_I) +'_full -interactive -analysis ',
        global_model= name+'_GLOBAL.inp' ,
        merge_surf_labels=[Local_Merge_Set],
        global_surf_labels=[Global_Merge_Set],
        locglob_constraint_adjust=False,
        locglob_constraint_pos_tol=0.001,
        crack_contact_type=1,
        crack_contact_surf_interact='A',
        crack_contact_surf_behavior=1,
        crack_contact_friction=0,
        crack_contact_small_sliding=False,
        crack_contact_tied=False,
        crack_contact_adjust=0.1)
