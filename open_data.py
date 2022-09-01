import pickle as pkl

def open_data(path):
    """
    Fonction pour ouvrir un jeu de données à partir d'un fichier .pkl
    in : path = le chemin du fichier
    out : data - les données contenues
    """
        
    with open(path, 'rb') as f:
        data = pkl.load(f)
        
    return data