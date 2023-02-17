import os
import torch
from torch.nn.functional import one_hot
import pandas as pd

base_dir = "./data"
file_name = "test_full.csv"

def read_unique_labels(base_dir, file_name, unique_labels = set()):
    print("Reading unique labels")
    with open( os.path.join( base_dir, file_name ), 'rt' ) as f:
        lines = f.readlines()
        train_x = []
        train_y = []

        for i, line in enumerate( lines ):
            #if i % 500 == 0:
            #    print(i)
            if i == 0:
                continue
            elem = line.split( ',' )
            train_x.append( elem[ 2 ] )
            labels = elem[ 3 ].split( ';' )
            unique_labels = unique_labels.union(set(labels))
            train_y.append( labels )
        #unique_labels = list(unique_labels)
    
    return unique_labels

def prepare_data(base_dir, file_name, unique_labels = None):
    files = ["train_full.csv", "test_full.csv", "dev_full.csv"]
    import numpy as np
    if not unique_labels:
        unique_labels = set()
        for f in files:
            unique_labels = read_unique_labels(base_dir, f, unique_labels)
    
    unique_labels = list(unique_labels)
    with open( os.path.join( base_dir, file_name ), 'rt' ) as f:
        lines = f.readlines()
        train_x = []
        train_y = []

        for i, line in enumerate( lines ):
            if i == 0:
                continue
            elem = line.split( ',' )
            train_x.append( elem[ 2 ] )
            labels = elem[ 3 ].split( ';' )
            train_y.append( labels )
        
    train_y_binary = np.zeros( ( len( train_y ), len( unique_labels ) ), dtype= 'int32' )
    for i, y in enumerate( train_y ):
        for yy in y:
            j = unique_labels.index( yy )
            train_y_binary[ i, j ] = 1

    data_dict = {
            "text" : train_x,
            "label" : train_y_binary
    }
    
    return data_dict, unique_labels

