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


def R_curve(a,K1c_0):
    R = K1c_0 + 1 * K1c_0 * (1 - m.exp(-0.05*a))
    return R
#from Model_Definition import BONE

#mod_vars = importlib.import_module('abafrank_lib.model_definitions.{}'.format(BONE))
#globals().update({v: getattr(mod_vars, v)
#                  for v in mod_vars.__dict__
#                  if not v.startswith("_")})
# Make sure this is the last import for overwriting variables
#if os.path.isfile('Overwrite_Variables.py'):
#    from Overwrite_Variables import *

# Janky logging initiate
logfile = open('templog_propagatecrack', 'w')
def print_and_log(string):
    print_and_log_std_interp(string, logfile)


# Input Array: [ dtp_Step, Current_Step, UCI, int(0), int(1), float(2.), int(0) ]

#dtp_Step = int(sys.argv[-7])
#Current_Step = int(sys.argv[-6])
#UCI = int(sys.argv[-5])

#SKIP_CHECK = bool(int(sys.argv[-4]))
#if SKIP_CHECK == True:
#    print_and_log('##############################                 WARNING: SKIP_CHECK IS TRUE.')

# Set Normal Method and Median Extension
#Normal_Method = int(sys.argv[-3])
#median_extension = float(sys.argv[-2])
#Meshing_Strategy = int(sys.argv[-1])
#try:
#    assert(Normal_Method in [1,2])
#    assert(median_extension > 0.)
#except:
#    print_and_log('INPUT ERROR: Normal Method: ' + str(Normal_Method) + ', Med. Ext.: ', str(median_extension))
#    raise InputError('Values passed into PropagateCrack are incorrect.')

# prop_crack_file = 'p_crack_log_' + str(dtp_Step) + '_' + str(Current_Step) + '_' + str(UCI) + '.propagatecrack'

#if (os.path.isfile('Kink_Extension_Model.py') == False):
#    copyfile( ABAFRANK_LIB_PATH + '/Kink_Extension_Model.py', str(os.getcwd()) + '/Kink_Extension_Model.py' )
#assert(os.path.isfile('Kink_Extension_Model.py'))
Name='Mpas'

def Open_fdb ():
    f3d.OpenFdbModel(file_name= Name +'.fdb' ,orig_mesh_file=Name + '_LOCAL.inp',
        global_file= Name + '_GLOBAL.inp' ,mesh_file= Name + '.inp',
        resp_file= Name + '_full.dtp')


def Write_INP ():
    #[name,Simulation_Step,U_I] = args
    f3d.RunAnalysis(
        model_type='ABAQUS',
        file_name=Name + '_full.fdb',
        flags=['NO_WRITE_TEMP','TRANSFER_BC','NO_CFACE_TRACT','CFACE_CNTCT','FILE_ONLY','MID_SIDE'],
        merge_tol=0.0001,
        connection_type='CONSTRAIN',
        executable=abaqus_bat_bin,
        command=abaqus_bat_bin + ' job=' + Name +'_full -interactive -analysis ',
        global_model= Name + '_GLOBAL.inp' ,
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

filename = Name + '_original.frt'
NumFronts = 1

def writeFrt():
    f3d.ComputeSif(sif_method='DISP_CORR', do_crack_face_contact=False)
    fdata = []
    for frontnum in range(int(NumFronts)):
        front_ = filename + str(frontnum)
        if ( os.path.isfile( front_ ) ) :
            os.remove(front_)
        f3d.WriteSif(file_name= front_ , front_id = frontnum , flags=['COMMA', 'CRD'],
        sif_method='DISP_CORR', do_crack_face_contact=False)
        fdata.append(genfromtxt(front_, delimiter=',', skip_header=1))
    
    if ( os.path.isfile( filename ) ) :
            os.remove(filename)
    
    f = open(filename, 'w')
    f.write('CRACK_PROPAGATION\n{\nVERSION: 1\nNUM_CRACK_FRONTS: '+str(NumFronts)+'\n')
    for frontnum in range(int(NumFronts)):  
        f.write('OLD_FRT_PTS: '+ str(frontnum) + ' {}\n'.format(len(fdata[frontnum])))
        for d in fdata[frontnum]:
            f.write('             {}          {}          {}\n'.format(
                d[1], d[2], d[3]))
        f.write('}\n')
    f.close()

    # Check if the .frt file is generated successfully
    if ( os.path.isfile( filename ) ) :
        print('##############################                 FRANC 3D: Original .frt file generated successfully.\n')
    else:
        print('##############################                 FRANC 3D: Original .frt file generation failed!\n')
        raise Exception('FRANC 3D: Original .frt file generation failed!')
    
    # Read material coordinate system
[VX, VY, VZ, name , COORD] = Read_Initial_Velocity()

# Set filenames
local_inp_file = Name + '_LOCAL.inp'
global_inp_file = Name + '_GLOBAL.inp'
base_name = Name
original_file_crk = base_name + '_original.crk'
original_file_frt = Name + '_original.frt'

# Start an instance of F3D
f3d = PyF3D.F3DApp()

# Open the fdb model for the current step

Open_fdb ()

#figuring out the number of crack fronts before calling writeFrt()

num_fronts= 1

###################### Generating original .frt file ########################
FM,ExTrapo = [],[]
for _ in range(1):
    FM.append(1)
    ExTrapo.append([30,30])

writeFrt()


####################### Generating New Crack Front #####################
# Calculate kink angle based on predictions of F3D
original_front = open(original_file_frt,'r')
lines = list( original_front.readlines() )
original_front.close()
# Read from outer surface nodes to determine normal
F = open( 'OUTER_SURF.json' ,'r')
LOC = json.load(F)
F.close()

# Reopen the model
Open_fdb ()

# Compute SIF of the initial crack
f3d.ComputeSif(sif_method='DISP_CORR', do_crack_face_contact=False)

#TODO stuff below this:(
# Write all the fronts to file
num_pts,Front_Mult,KINK,EXTENSION = [],[],[],[]
for front_ in range(1): # SimDict['Edge']): # Loop through each crack fronts 
    # Compute KIc for this current crack based on current crack length and R curve
    curr_crack_length = .5 #SimDict['crack_length'][ SimDict['Crack_Lookup'][front_] ][-1]
    K1c_parallel = R_curve( curr_crack_length , 2187.23)
    #print(K1c_parallel)
    fronts = [] # Old crack front coordinates
    for _ in range(len(lines)): # Loop through lines in the frt file
        if ('OLD_FRT_PTS: '+str(front_) in lines[_]):
            num_pts.append(int(lines[_][14:])) # Read number of points
            for __ in range(_+1 , _+num_pts[front_]+1): # Loop through all available points
                temp = []
                for item in lines[__].split(' '):
                    try:
                        temp.append(float(item)) # Coordinates of the old crack front points
                    except:
                        pass
                fronts.append( np.array(temp) )
                #print(len(fronts))
            break
               
    # Calculate kink angle using the crack front point with median K_equivalent
    # Read SIF data along the front
    FN =Name + '.sif' #SimDict['NAME'] + '_Front_' + str(front_) + '_Stp' + str(SimDict['Current_Step']+Step_increment-1) + '_UCI ' + str(SimDict['Unstable_iter']) + '.sif'
    f3d.WriteSif(file_name= FN , front_id = front_ , flags=['COMMA', 'KI' , 'KII' , 'KIII' , 'AXES'],
        sif_method='DISP_CORR', do_crack_face_contact=False)
    [K1,K2,K3,Front_Axes] = Parse_SIF (FN)

    #Writes the front location file to be able to compare front locations later
    FrontLocName = 'Location_of_Front_'+ Name +'.sif' #SimDict['NAME'] + 'Location_of_Front_' + str(front_) + '_Stp' + str(SimDict['Current_Step']+Step_increment-1) + '_UCI ' + str(SimDict['Unstable_iter']) + '.sif'
    delete_if_exists([FrontLocName])
    f3d.WriteCOD(file_name = FrontLocName, front_id = int(front_), flags=['COMMA','CRD'])


    # Compute an equivalent K using all 3 modes of SIF (sign comes from the sign of K1)
    beta2 = np.abs( K2 ) / ( np.abs(K2) + np.abs(K3) )
    beta3 = np.abs( K3 ) / ( np.abs(K2) + np.abs(K3) )
    K_equivalent = np.sqrt( K1**2 + beta2 * K2**2 + beta3 * K3**2 )

    # Identify the sense (outward is desired) of the surface normal
    # This can be done by comparing the minimum distance from either end of the old crack front
    # to the nodes on the outer surface of the bone region
    DISTANCE1 , DISTANCE2 = [] , []
    for _ in LOC :
        DISTANCE1.append( -1 * np.linalg.norm( np.array(fronts[0] ) - _ ) )
        DISTANCE2.append( -1 * np.linalg.norm( np.array(fronts[-1] ) - _ ) )

    # Get top point on the extended front
    old_front = fronts[0] - fronts[-1] # The old crack front vector
    mag_old_front = np.linalg.norm( old_front ) # Length of the old crack front

    if ( min(DISTANCE1) < min(DISTANCE2) ): # Meaning that front point 0 is on the top surface
        top_surface_index = 0
        print(' in if part')
    else: # Meaning that front point -1 (last one) is on the top surface
        old_front *= -1 
        top_surface_index = -1

    Top_point = np.array( fronts[top_surface_index] )
    Local_Normal =[0.,0.,1.] #Get_Normal( Top_point , -old_front ) # normal relative to the three point bend thingy


    # Global material toughness axis (Cylinderical), predefined in Set_Impact_ConfiguSimDict['ratio']n.py
    # The polar(z) axis passes through the ossification center, Location of the ossification center
    Material_X,Material_Y,Material_Z,OC = COORD #this OC is wrong and needs ot be changed TODO bjorn fix
    
    # Mapping that transforms the global coordinate system to the material coordinate (2D)
    Transform = np.linalg.inv( np.matrix(np.transpose([Material_X,Material_Y,Material_Z])) )

    Overall_max_K_resolved = []
    Front_Kink_Angle = []

    # On May 23 2019, we decided to perform averaging over the crack front to smooth out the SIFs
    # 20% of the crack front nodes will be averaged, with a minimum number of 3 points to average
    Num_pt_avg = max( int(round(0.2*len(K1))) , 3 )
    print_and_log('##############################                 FRANC 3D: ' + str( Num_pt_avg ) + ' neighboring crack front nodes will be averaged.')

    for kink_node in range(len(K1)): # Loop through all nodes at the crack front
        Local_X = Front_Axes[kink_node][0] # Along the direction of crack advancement 
        Local_Z = Front_Axes[kink_node][2] # Along the crack axis, outward
        # Check sense of the Z axis (it should be outward by default but just to be safe)
        if (np.dot(old_front,Local_Z)<=0):
            Local_Z *= -1 # Original Z axis is inward, flip the sense
        Local_Y = np.cross(Local_Z,Local_X) # Recompute the local Y axis so the local coordinate is right-handed

        # Compute average SIFs at this current node
        Start_Ind = max( 0 , kink_node - Num_pt_avg/2 )
        End_Ind = min( len(K1) , kink_node + Num_pt_avg/2 + 1 )
        if ( Start_Ind == 0 ):
            End_Ind = Num_pt_avg
        if ( End_Ind == len(K1) ):
            Start_Ind = len(K1) - Num_pt_avg

        # Actually convert these into ints for python 3
        # In old version, inconsistent averaging due to using FLOATS for slices!!!!
        Start_Ind = int(math.ceil(Start_Ind))
        End_Ind = int(math.floor(End_Ind))

        K1_avg = np.mean( K1[Start_Ind:End_Ind] )
        K2_avg = np.mean( K2[Start_Ind:End_Ind] )
        K3_avg = np.mean( K3[Start_Ind:End_Ind] )

        # Sweep through angles from -90 to 90 to obtain kink angle
        # Initialize kink angle and max K (resolved)
        t = -np.pi / 2
        MAX_Kdiff_theta = -1*float('inf')
        Kink_angle = 0

        k_delta_max = 0 # BRP: Delta Modification
        while ( t <= np.pi/2 ): #when t = 0, it is checking 0 degrees
            # Compute the resolved K (Driving force)
            # Max. tensile stress criterion:
            K1_resolved = m.cos(t/2) *( K1_avg * m.cos(t/2)**2 - 1.5 * K2_avg * m.sin(t) )
            K_MTS = K1_resolved

            # Max. shear stress criterion:
            K2_resolved = abs(0.5*m.cos(t/2)*( K1_avg * m.sin(t) - K2_avg * (3*m.cos(t) - 1) ))
            K3_resolved = abs(K3_avg*m.cos(t/2))

            # Compute weight of the Mode II and III factors based on their relative magnitudes
            K2_weight = K2_resolved/(K2_resolved+K3_resolved)
            K3_weight = K3_resolved/(K2_resolved+K3_resolved)

            K_MSS = m.sqrt( (K2_weight*K2_resolved)**2 + (K3_weight*K3_resolved)**2 )
            
            # Use the generalized stress criterion for driving force
            K_Dominant = max([K_MTS,K_MSS])

            # Compute the direction of a virtual extension of unit length (In gloabl cartesian coordinate)
            virtual_front = Local_Y*m.sin(t) + Local_X*m.cos(t)
            # print_and_log("\n\nt = " + str(t)) #TODO: DELETE THIS
            # print_and_log("KINK ANGLE = " + str(Kink_angle)) #TODO: DELETE THIS
            # print_and_log("VIRTUAL FRONT DIRECTION (global) = " + str(virtual_front)) #TODO: DELETE THIS

            # Compute the local fiber direction at the crack tip of the main crack front (In gloabl cartesian coordinate)
            local_fiber = Normalize(OC - np.array(fronts[kink_node]))
            # print_and_log("LOCAL FIBER DIRECTION (global) = " + str(local_fiber)) #TODO: DELETE THIS
            
            # Transform these directions to the material axis ( Change of basis )
            # Only the projection onto the X-Y plane is used since the material axis is assumed to be 2D
            Fiber_direction = np.array( Transform * np.matrix(local_fiber.reshape((3,1))) ).reshape((1,3))[0][:2]
            # print_and_log("FIBER DIRECTION (material) = " + str(Fiber_direction))#TODO: DELETE THIS
            Fiber_direction = Normalize(Fiber_direction)
            # print_and_log("FIBER DIRECTION (material) (normalized) = " + str(Fiber_direction))#TODO: DELETE THIS

            # print_and_log("Transform = " + str(Transform)) #TODO: delete this
            Virtual_Extension = np.array( Transform * np.matrix(virtual_front.reshape((3,1))) ).reshape((1,3))[0][:2]
            # print_and_log("VIRTUAL EXTENSION DIRECTION (material) = " + str(Virtual_Extension)) #TODO: DELETE THIS
            Virtual_Extension = Normalize(Virtual_Extension)
            # print_and_log("VIRTUAL EXTENSION DIRECTION (material) (normalized) = " + str(Virtual_Extension)) #TODO: DELETE THIS

            # Calculate the angle between them
            tot_ang = np.arccos( np.dot(Fiber_direction , Virtual_Extension) )

            K1c_MTS = K1c_parallel *  (math.cos(tot_ang )**2) + 10 * K1c_parallel * (math.sin(tot_ang )**2) #TODO: bjorn apperently ratip = 10
            K1c_MSS = K1c_MTS * np.sqrt(0.75) #TODO: trying k_diff new code
            
            K_diff = np.array([K_MSS, K_MTS]) - np.array([K1c_MSS, K1c_MTS]) #TODO: trying k_diff new code

            K_diff_dominant = K_diff[np.argmax(K_diff)] #TODO: trying k_diff new code
            K1c_Final = K1c_MTS if ( np.argmax(K_diff) == 1 ) else K1c_MSS #TODO: trying k_diff new code
            K_Dominant = K_MTS if ( np.argmax(K_diff) == 1 ) else K_MSS #TODO: trying k_diff new code
            
            # Calculate the fracture toughness at this kink angle (Resistance)
            #K1c_theta = K1c_parallel *  (m.cos(tot_ang )**2) + SimDict['ratio'] * K1c_parallel * (m.sin(tot_ang )**2) #TODO: trying k_diff commmented code
            #if(abs(t) < 0.01):
            # print_and_log("PROPAGATE CRACK t = " + str(t)) #TODO: DELETE THIS
            # print_and_log("k_diff array = " + str(K_diff)) #TODO: DELETE THIS 
            # print_and_log("max k_diff = " + str(K_diff_dominant)) #TODO: DELETE THIS
            #print_and_log("K1c_theta = " + str(K1c_theta))#TODO: DELETE THIS


            # Use KI_c to compute KII_c if crack grows in under shear stress
            # if ( np.argmax([K_MTS,K_MSS]) == 1 ):
            #     K1c_theta *= np.sqrt(0.75)

            # k_delta = K_Dominant - K1c_theta # BRP: Delta Modification #TODO: trying k_diff commented code
            k_delta = K_diff_dominant #TODO: trying k_diff new code

            # If K1 is max and local K1 is greater than local resistance, this is the kink angle
            #if ( K_Dominant >= MAX_Kdiff_theta and K_Dominant >= K1c_theta ):
            if (k_delta >= 0 and k_delta > k_delta_max): # BRP: Delta Modification
                k_delta_max = k_delta  # BRP: Delta Modification
                MAX_Kdiff_theta = K_Dominant
                Kink_angle = t
            t += 0.0087 # Using 0.25 degrees increment

        Overall_max_K_resolved.append(MAX_Kdiff_theta)
        Front_Kink_Angle.append(Kink_angle)

    if ( max(Overall_max_K_resolved) != -1*float('inf') ): # Meaning a kink angle is found, crack growth will occur
        node_index = np.argmax(Overall_max_K_resolved) # Use the nodal location where the resolved K1 is max
        kink = Local_Y*m.sin(Front_Kink_Angle[node_index]) + Local_X*m.cos(Front_Kink_Angle[node_index])
        Kink_angle = Front_Kink_Angle[node_index]
        #print_and_log("KINK ANGLE NODE CHOSEN = " + str(node_index)) #TODO: trying k_diff new code
        #print_and_log("KINK ANGLE K_DIFF CHOSEN = " + str(np.max(Overall_max_K_resolved))) #TODO: trying k_diff new code
        #print_and_log("MAX K_DIFF FOR ALL KINK ANGLES = " + str(Overall_max_K_resolved)) #TODO: trying k_diff new code
        #print_and_log('##############################                 FRANC 3D: ' + 'On initiation site ' + str(SimDict['Crack_Lookup'][front_]) +' front ' + str(front_) + ': ')
        #print_and_log('##############################                 FRANC 3D: Kink angle on step '+str(SimDict['Current_Step']+Step_increment-1)+ ' UCI ' + str(SimDict['Unstable_iter']) + ' is: '+str(np.degrees(Kink_angle))+' degrees.\n')

        Top_point = fronts[top_surface_index] + 1.* kink #median_extension * kink #TODO BJORN extension length

        #if ( Normal_Method == 1 ):
        Local_Normal = Normalize([0.,0.,1.])    #Get_Normal( Top_point , old_front ) # Compute surface normal on this point
        #else:
        #    Local_Normal = Normalize(old_front)

        # Calculate crack front extensions
        Extension,Kink_angle = [] , []
        for N in range( num_pts[front_] ):
            Local_Y = Front_Axes[N][1] # Along the direction of crack advancement 
            Local_Z = Front_Axes[N][2] # Along the crack axis, outward
            Lambda = np.dot( (np.array(fronts[N])-np.array(Top_point)) , Local_Z ) / np.dot( Local_Normal , Local_Z )
            v = np.array(Top_point)-np.array(fronts[N]) + Lambda * Local_Normal
            Extension.append( np.linalg.norm(v) )
            Kink_angle.append( np.pi/2.0 - np.arccos( np.dot(Normalize(v) , Local_Y ) )  )

        # Update crack length history
        #SimDict['crack_length'][ SimDict['Crack_Lookup'][front_] ].append( curr_crack_length + np.mean(Extension) )

        # Assign front multiplier
        Front_Mult.append(1) # 1 means to propagate this front

    else: # Meaning crack growth on this front is not going to occur in this step
        ##print_and_log('##############################                 FRANC 3D: On initiation site ' + str(SimDict['Crack_Lookup'][front_]) + ' front ' + str(front_) + ', no crack growth occurred in this step.')
        ## Assign kink angle
        #Kink_angle = np.zeros(num_pts[front_])
        ## Assign extension
        #Extension = np.zeros(num_pts[front_])
        ## Assign front multiplier
        ##Front_Mult.append(0) # 0 means do not propagate this front
        #Front_Mult.append(1) # 1
        node_index = np.argmax(Overall_max_K_resolved) # Use the nodal location where the resolved K1 is max
        kink = Local_Y*m.sin(Front_Kink_Angle[node_index]) + Local_X*m.cos(Front_Kink_Angle[node_index])
        Kink_angle = Front_Kink_Angle[node_index]
        #print_and_log("KINK ANGLE NODE CHOSEN = " + str(node_index)) #TODO: trying k_diff new code
        #print_and_log("KINK ANGLE K_DIFF CHOSEN = " + str(np.max(Overall_max_K_resolved))) #TODO: trying k_diff new code
        #print_and_log("MAX K_DIFF FOR ALL KINK ANGLES = " + str(Overall_max_K_resolved)) #TODO: trying k_diff new code
        #print_and_log('##############################                 FRANC 3D: ' + 'On initiation site ' + str(SimDict['Crack_Lookup'][front_]) +' front ' + str(front_) + ': ')
        #print_and_log('##############################                 FRANC 3D: Kink angle on step '+str(SimDict['Current_Step']+Step_increment-1)+ ' UCI ' + str(SimDict['Unstable_iter']) + ' is: '+str(np.degrees(Kink_angle))+' degrees.\n')

        Top_point = fronts[top_surface_index] + 1.* kink #median_extension * kink #TODO BJORN extension length

        #if ( Normal_Method == 1 ):
        Local_Normal = Normalize([0.,0.,1.])    #Get_Normal( Top_point , old_front ) # Compute surface normal on this point
        #else:
        #    Local_Normal = Normalize(old_front)

        # Calculate crack front extensions
        Extension,Kink_angle = [] , []
        for N in range( num_pts[front_] ):
            Local_Y = Front_Axes[N][1] # Along the direction of crack advancement 
            Local_Z = Front_Axes[N][2] # Along the crack axis, outward
            Lambda = np.dot( (np.array(fronts[N])-np.array(Top_point)) , Local_Z ) / np.dot( Local_Normal , Local_Z )
            v = np.array(Top_point)-np.array(fronts[N]) + Lambda * Local_Normal
            Extension.append( np.linalg.norm(v) )
            Kink_angle.append( np.pi/2.0 - np.arccos( np.dot(Normalize(v) , Local_Y ) )  )

        # Update crack length history
        #SimDict['crack_length'][ SimDict['Crack_Lookup'][front_] ].append( curr_crack_length + np.mean(Extension) )

        # Assign front multiplier
        Front_Mult.append(1) # 1 means to propagate this front



    # Store kink angle and extension information
    KINK.append(Kink_angle)
    EXTENSION.append( Extension )

#TODO BJORN BELOW THIS:)
# Write the front files for all fronts
new_name = Name + '_MAYYBE_DONE'
for numbering in range(1):
    delete_if_exists(['Front_'+str(numbering)+'.txt'])
    F=open('Front_'+str(numbering)+'.txt','w')
    F2=open( new_name + '_Front_'+str(numbering)+'.txt','w')
    for _ in range(len(EXTENSION[numbering])):
        LL = str(EXTENSION[numbering][_]) + ',' + str(KINK[numbering][_]) + '\n'
        F.write( LL )
        F2.write( LL )
    F.close()
    F2.close()


# below ithis is get new input file
from abafrank_lib.Function_Library import *
import json
import sys
from shutil import copyfile
from abafrank_lib.Variables import *
import os
from shutil import copyfile

try:
    import PyF3D
    import Vec3D
except Exception as e:
    raise e

#from Model_Definition import BONE
#mod_vars = importlib.import_module('abafrank_lib.model_definitions.{}'.format(BONE))
#globals().update({v: getattr(mod_vars, v)
#                  for v in mod_vars.__dict__
#                  if not v.startswith("_")})
# Make sure this is the last import for overwriting variables
if os.path.isfile('Overwrite_Variables.py'):
    from Overwrite_Variables import *



input_hash = str(sys.argv[-1])
print(input_hash)
# Read simulation inputs
#SimDict = Read_Simulation_Settings( None )

# Read meshing inputs
#F = open('Meshing_Parameters.json','r')
#meshing_params = json.load(F)
#smoothing_method , extrapo , rad , Mult,Step_increment, Meshing_Strategy, mesh_hash = meshing_params
#F.close()
#assert(input_hash == mesh_hash)
#copyfile('Meshing_Parameters.json', 'Meshing_Parameters_' + str(SimDict['Current_Step']+Step_increment-1) + '_UCI_' + str(SimDict['Unstable_iter']) + '.json')

F = open('Meshing_Parameters.json','r')
meshing_params = json.load(F)
smoothing_method , extrapo , rad , Mult,Step_increment, Meshing_Strategy, mesh_hash = meshing_params #thiss will need to be checked
F.close()

# Set filenames
local_inp_file = Name + '_LOCAL.inp'
global_inp_file = Name + '_GLOBAL.inp'
base_name = Name
new_file_crk = base_name + '_Modified.crk'

# Start an instance of F3D
f3d = PyF3D.F3DApp()

# Open current step model
f3d.OpenFdbModel(
    file_name= Name + '.fdb',
    orig_mesh_file=local_inp_file,
    global_file=global_inp_file,
    mesh_file= Name + '.inp',
    resp_file= Name + '_full.dtp')
# Change F3D settings

# Using Abaqus to mesh the modified crack
#if Meshing_Strategy == 0:
f3d.SetMeshingParameters(volume_meshing_method = 'ABAQUS', do_not_coarsen_uncracked=True,
    surf_mesh_density_decay_ratio=1.15)
#elif Meshing_Strategy == 1:
#    f3d.SetMeshingParameters(volume_meshing_method = 'ABAQUS')
#else:
#    f3d.SetMeshingParameters(volume_meshing_method = 'ABAQUS', do_not_coarsen_uncracked=True,
#       surf_mesh_density_decay_ratio=1.15)

# Use the user-defined kink angle model
f3d.SetUserExtensionsFile(
    file_name='Kink_Extension_Model.py', #  TODO this appears to be uniform across diff simulations
    flags=['USER_NEW_POINT','USER_START_STEP'])
SM,EXT=[],[]
for nf in range(1):
    if ( Mult[nf] == 1 ):
        SM.append( smoothing_method )
    else:
        SM.append( 'FIXED_ORDER_POLY' )
    EXT.append( [ extrapo , extrapo ] )
f3d.ComputeSif(sif_method='DISP_CORR', do_crack_face_contact=False)
f3d.SetGrowthParams(
    growth_type="USER_POINTS",
    load_step_map=["VERSION: 1","MAX_SUB: 1","0 0","LABELS: 1","1 Load Step 1"],
    kink_angle_strategy="MTS",
    median_step=1,
    front_mult=Mult)
f3d.GrowCrack(
    median_step=1,
    front_mult=Mult,
    temp_radius_type="ABSOLUTE",
    temp_radius= rad ,
    smoothing_method= SM ,
    extrapolate=EXT,
    file_name=new_file_crk)

# Save the meshed model
f3d.SaveFdbModel(file_name = Name + '_completed.fdb',
    mesh_file_name = Name + '_mesh_completed.cdb',
    rslt_file_name = Name + '_rslt_completed.dtp',
    analysis_code = 'ABAQUS')
