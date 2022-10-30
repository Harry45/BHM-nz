"""
Author: Arrykrishna Mootoovaloo
Date: October 2022
Code: Some utility functions to save/load data.
Project: Inferring the tomographic redshift distribution of the KiDS-1000
catalogue.
"""
import os 
import numpy as np 

def load_arrays(folder_name, file_name):
    '''
    Given a folder name and file name, we will load
    the array

    :param: folder_name (str) - the name of the folder

    :param: file_name (str) - name of the file

    :return: matrix (np.ndarray) - array
    '''

    matrix = np.load(folder_name + '/' + file_name + '.npz')['arr_0']

    return matrix


def store_arrays(array, folder_name, file_name):
    '''
    Given an array, folder name and file name, we will store the
    array in a compressed format.

    :param: array (np.ndarray) - array which we want to store

    :param: folder_name (str) - the name of the folder

    :param: file_name (str) - name of the file
    '''

    # create the folder if it does not exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # use compressed format to store data
    np.savez_compressed(folder_name + '/' + file_name + '.npz', array)
    
