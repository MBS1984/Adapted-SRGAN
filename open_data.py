import pickle as pkl

def open_data(path):
    """
Function to open a dataset from a .pkl file
    """
        
    with open(path, 'rb') as f:
        data = pkl.load(f)
        
    return data
