# IMPORT PACKAGES
#system packages
import os
import sys
#Computational packages
import numpy as np
import pandas as pd
import scipy
import tables
from scipy.io import loadmat
from pandas.core.frame import DataFrame
import numba
from numba import jit, njit, vectorize
from numba.extending import overload
import itertools
np.seterr(divide='ignore', invalid='ignore')

# GLOBAL VARIABLES
TIME_INTERVAL = 10/60
PIXEL_TO_MICRON  = 1.29 #pixel/micron

# FUNCTIONS
def main():
    #Initalize variables
    dic_directionality = {}
    list_directionality = []
    dic_speed = {}
    list_speed = []
    dic_msd = {}
    list_msd = []
    folder_list = []
    subcolumn_count = 0

    #get file path name (directory)
    directory_pathway = input("Please Input FULL Parent Diretory. \nExample:" r"C:\Data\parent:  ")
    directory_check(directory_pathway)

    for folder_path, _, files in os.walk(directory_pathway, topdown=False):
        folder_name = os.path.basename(folder_path)
        folder_list.append(folder_name)
        list_directionality = []
        list_speed = []
        list_msd = []
        for file in files:
            dir_out, speed_out, msd_out = mat_process(os.path.join(folder_path,file))
            list_directionality.append(dir_out)
            list_speed.append(speed_out)
            list_msd.append(msd_out)
        dic_directionality[folder_name] = pd.DataFrame(list_directionality)
        dic_speed[folder_name] = pd.DataFrame(list_speed)
        dic_msd[folder_name] = pd.DataFrame(list_msd)
    print('All folders have been processed')
    #Convert dictionaries to dataframes with the time for x axis (far left) and output to excel. If possible, directly output to GraphPad Prism
    df_dic = pd.concat(dic_directionality.values(), keys=dic_directionality.keys())
    df_speed = pd.concat(dic_speed.values(), keys=dic_speed.keys())
    df_msd = pd.concat(dic_msd.values(), keys=dic_msd.keys())
    df_dic.to_csv(os.path.join(directory_pathway,"Directionality-output.csv"))
    df_speed.to_csv(os.path.join(directory_pathway,"Speed-output.csv"))
    df_msd.to_csv(os.path.join(directory_pathway,"MSD-output.csv"))

    
def directory_check(directory_pathway):
    #INPUT PARENT DIRECTORY
    #starting parent directory 
    #CHECKPOINT: Is this specified parent directory valid
    if len(os.listdir(directory_pathway)) == 0:
        print("\nThis is an empty directory. Exiting code now.")
        sys.exit(1)
    elif os.path.isdir(directory_pathway):
        print("\nIt is a directory. New WD is set to specified path.")
        #change current working directory to specificed directory
        os.chdir(directory_pathway)
    else:
        print("\nThis is not a directory. Exiting code now.")
        sys.exit(1)
    #CHECKPOINT: Make sure path will work for all operating systems
    directory_pathway = os.path.normpath(directory_pathway)

def mat_process(file):
    '''
    
    '''
    mat_file = loadmat(file)
    u = [[element for element in upperElement] for upperElement in mat_file['u_filtered']]
    v = [[element for element in upperElement] for upperElement in mat_file['v_filtered']]
    vel = [[element for element in upperElement] for upperElement in mat_file['velocity_magnitude']]
    u_component, v_component, velocity_matrix = reshape_vec(u, v, vel)
    dir_out, sped_out, msd_out  = vector_calculations(u_component, v_component, velocity_matrix)
    return dir_out, sped_out, msd_out

@overload(np.fliplr)
def reshape_vec(u_comp, v_comp, velocity_comp):
    ''' '''
    u_filter = np.fliplr(np.rot90(u_comp))
    v_filter = -1*(np.fliplr(np.rot90(v_comp)))
    vel_mag = np.fliplr(np.rot90(velocity_comp))
    u_matrix = np.reshape(np.ravel(u_filter), (len(u_comp), -1))
    v_matrix = np.reshape(np.ravel(v_filter), (len(v_comp), -1))
    velocity_matrix = np.reshape(np.ravel(vel_mag), (len(velocity_comp), -1))
    return u_matrix, v_matrix, velocity_matrix

def vector_calculations(u, v, vel):
    '''
    First break up the u and v matrices into temporary position vectors. This should be done for each time point. 
    Then take those temp values and calcualte the directionality of the vectors for that time point, store them, and repeat.
    '''
    # Make position 3d matrix, this will contain the u v component of a vector, for all vectors of a time point.
    position_matrix = np.stack((u,v),axis = -1)
    #use position matrix (combined components) for calucalti8ng desired variables.
    dir_corr = directionality(position_matrix,vel)
    speed_out = speed(position_matrix, vel)
    msd_pos_out = msd_position(u,v)
    msd_out = msd(position_matrix,vel)
    return dir_corr,speed_out,msd_pos_out

def directionality(position, vel):
    dir_cor = []
    dir_corr_avg = []
    unit = np.array([1,0]).transpose()
    time_count = 0
    for element in position:
        dir_cor_temp = np.divide(np.dot(element,unit),vel[time_count,:])
        time_count += 1
        dir_cor.append(dir_cor_temp)
        dir_corr_avg.append(np.nanmean(dir_cor_temp))
    return dir_corr_avg

def speed(position,vel):
    speed_list = []
    speed_avg = []
    for element in vel:
        speed_temp = element/(TIME_INTERVAL/PIXEL_TO_MICRON)
        speed_list.append(speed_temp)
        speed_avg.append(np.nanmean(speed_temp))
    return speed_avg

def msd(position, vel):
    msd_temp = []
    msd_avg = []
    vel_converted = np.divide(vel,PIXEL_TO_MICRON)
    vel_averaged = np.nanmean(vel_converted, axis=1)
    for window in range(2,len(vel)):
        sliding_window = np.lib.stride_tricks.sliding_window_view(vel_averaged, window)
        for element in sliding_window:
            msd_temp.append((element[-1]-element[0])**2)
        msd_avg.append(np.mean(msd_temp))
    return msd_avg

def msd_position(u,v):
    msd_avg = []
    u = np.divide(u,PIXEL_TO_MICRON)
    v = np.divide(v,PIXEL_TO_MICRON)
    reference_u = u[0]
    reference_v = v[0]
    #MSD = average(|position of ith particle at t time - reference position of ith particle|^2)
    for window_size in range(2,len(u)):
        u_comp = u[window_size]-reference_u
        v_comp = v[window_size]-reference_v
        square_disp = np.add(np.square(u_comp), np.square(v_comp))
        msd_avg.append(np.nanmean(square_disp))
    return msd_avg

if __name__ == '__main__':

    main()