import argparse
import pickle
import time
from etl.etl_utilities import etl_1, etl_2

def str2bool(value):
    return value.lower == 'true'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #execute first step of etl1
    parser.add_argument("-etl1", default = True, type = str2bool)

    #execute second step of et2
    parser.add_argument("-etl2", default = True, type = str2bool)

    #treshold for intersection scorer to clean indirizzi
    parser.add_argument("-treshold", default = 0.6, type = float)

    #path data of data.pkl
    parser.add_argument("-path_data", default = "data/immobiliare/data.pkl", type = str)

    #path open for open dataset
    parser.add_argument("-path_datasetMilano", default = "data/dataset_Milano/", type = str)

    #path open for scraped dataset
    parser.add_argument("-path_openMilano", default = "data/openMilano/", type = str)


    #path data of data_url.pkl
    parser.add_argument("-path_data_url", default = "data/immobiliare/data_url.pkl", type = str)

    #saving path  of output
    parser.add_argument("-path_etl_out", default = "data/data/", type = str)

    #logs path for etl2-scraped dataset
    parser.add_argument("-path_etl2_log_data", default = 'logs/etl_2_log_data.txt', type = str)

    #logs path for etl2-aler home
    parser.add_argument("-path_etl2_log_aler", default = 'logs/etl_2_log_aler.txt', type = str)

    #supermercati considered during geo_etl
    parser.add_argument("-supermercati", default = ['esselunga', 'carrefour', 'conad', 'coop', 'lidl'], type = list)

    args = parser.parse_args()
            
    #PHASE -1 HOME SCRAPE PAGE 
    if args.etl1:
        print('Beginning first step: etl1\n')

        try:
            #open data file
            with open(args.path_data, 'rb') as file:
                data = pickle.load(file)
            
            #open url data file
            with open(args.path_data_url, 'rb') as file:
                url_ = pickle.load(file)

            #etl pipeline
            data = etl_1(data, url_)

            #save file
            with open(args.path_etl_out + 'data_etl1.pkl', 'wb') as file:
                pickle.dump(data, file)

        except:
            print('Missing file or wrong path')

    if args.etl2:

        print('Beginning Second step: etl2\n')

        try:

            #open data file
            with open(args.path_etl_out + 'data_etl1.pkl', 'rb') as file:
                data = pickle.load(file)
            
            #etl pipeline
            data, aler_home_corrected = etl_2(args, data)

            #save file
            with open(args.path_etl_out + 'data_etl2.pkl', 'wb') as file:
                pickle.dump(data, file)

            with open(args.path_etl_out + 'data_aler_corrected.pkl', 'wb') as file:
                pickle.dump(aler_home_corrected, file)

        except:
            print('Missing file or wrong path')
