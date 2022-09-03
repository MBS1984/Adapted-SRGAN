"""
Function to open a dataset from a pickle file
"""
import pickle as pkl

def open_data(path):

    with open(path, 'rb') as f:
        data = pkl.load(f)
        
    return data
