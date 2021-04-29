import argparse
import pickle
import numpy as np 
import pandas as pd
from etl.etl_utilities import reduce_mem_usage_sd
import math
from geopy import distance
from others.constant import COLUMN_TO_USE, CATEGORICAL, TO_DROP
import os

def str2bool(value):
    return value.lower == 'true'

if __name__ == '__main__':
    
    pd.options.mode.chained_assignment = None
    
    parser = argparse.ArgumentParser()

    #add duomo location
    parser.add_argument("-duomo_location", default = [9.191383, 45.464211], type = list)

    #min price
    parser.add_argument("-min_price", default = 30000, type = float)

    #min observation per nil
    parser.add_argument("-min_obs", default = 5, type = float)

    #path out
    parser.add_argument("-path_data_modeling", default = "data/data_modeling", type = str)

    #path import
    parser.add_argument("-path_data", default = "data/data", type = str)

    args = parser.parse_args()
    
    with open(os.path.join(args.path_data, 'data_final.pkl'), 'rb') as file:
        data = pickle.load(file)

    ##CLEAN DATA
    data = data[(data['Prezzo'].notna()) & (data['Prezzo'] > args.min_price)].reset_index(drop = True)

    data['distance_duomo'] = data.apply(
        lambda row: distance.distance(
                        tuple(row[['LONG_WGS84', 'LAT_WGS84']]),
                        args.duomo_location
                    ).kilometers, 
                    axis = 1
       )
    
    #categorical 
    for col in CATEGORICAL:
        count_rows = data.groupby('NIL')['stato'].size()
        to_del = count_rows[count_rows < args.min_obs].index
        data = data[~data[col].isin(to_del)].reset_index(drop = True)

    with open(os.path.join(args.path_data_modeling, 'data_processed.pkl'), 'wb') as file:
        pickle.dump(data, file)
        
    target = np.log1p(data['Prezzo']) 
    train = data[COLUMN_TO_USE]

    #categorical encoding
    dictionary_inverse_encoder = {}
    for cat in CATEGORICAL:
        mapping = {x: i for i, x in enumerate(train[cat].unique())}
        dictionary_inverse_encoder[cat] = {i: x for i, x in enumerate(train[cat].unique())}
                    
        train[cat] = train[cat].map(mapping)

    #change all data type and after downcast
    for col in train.columns:
        train[col] = train[col].copy().astype(float)
    
    train = reduce_mem_usage_sd(train)

    with open(os.path.join(args.path_data_modeling, 'data_model.pkl'), 'wb') as file:
        pickle.dump(train, file)

    with open(os.path.join(args.path_data_modeling, 'target.pkl'), 'wb') as file:
        pickle.dump(target, file)
