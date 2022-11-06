"""
Author: Arrykrishna Mootoovaloo
Date: October 2022
Code: Some utility functions to save/load data.
Project: Inferring the tomographic redshift distribution of the KiDS-1000
catalogue.
"""
import os
import pickle
import numpy as np
import pandas as pd


def load_arrays(folder: str, fname: str) -> np.ndarray:
    """Load numpy arrays

    Args:
        folder (str): folder where the file is stored
        fname (str): name of the file

    Returns:
        np.ndarray: the file
    """
    path = os.path.join(folder, fname)
    matrix = np.load(path + '.npz')['arr_0']
    return matrix


def store_arrays(array: np.ndarray, folder: str, fname: str):
    """Store a numpy array

    Args:
        array (np.ndarray): the array to be stored
        folder (str): name of the folder
        file (str): name of the file
    """

    # create the folder if it does not exist
    os.makedirs(folder, exist_ok=True)

    # use compressed format to store data
    path = os.path.join(folder, fname)
    np.savez_compressed(path + '.npz', array)


def pickle_save(file: list, folder: str, fname: str) -> None:
    """Stores a list in a folder.
    Args:
        list_to_store (list): The list to store.
        folder_name (str): The name of the folder.
        file_name (str): The name of the file.
    """

    # create the folder if it does not exist
    os.makedirs(folder, exist_ok=True)

    # use compressed format to store data
    path = os.path.join(folder, fname)
    with open(path + ".pkl", "wb") as dummy:
        pickle.dump(file, dummy)


def pickle_load(folder: str, fname: str) -> pd.DataFrame:
    """Reads a list from a folder.
    Args:
        folder_name (str): The name of the folder.
        file_name (str): The name of the file.
    Returns:
        pd.DataFrame: a pandas dataframe
    """
    path = os.path.join(folder, fname)
    with open(path + ".pkl", "rb") as dummy:
        file = pickle.load(dummy)
    return file
