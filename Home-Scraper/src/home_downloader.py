import argparse
from scrape_utilities.home_scrape import scraper, expand_grid
import pickle
import time
from scrape_utilities.url_scraper import scraper_single_url

def str2bool(value):
    return value.lower == 'true'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #scrape each home from zero or just single home
    parser.add_argument("-scrape_all", default = True, type = str2bool)

    #Waiting time between each requests
    parser.add_argument("-time_w8", default = 1.5, type = float)

    #Waiting time if not get 400
    parser.add_argument("-time_implicit_w8", default = 30, type = float)

    #timeout for response
    parser.add_argument("-timeout", default = 30, type = float)
    
    #Minimum/Maximum superficie
    parser.add_argument("-superfice_min", default = 35, type = float)
    parser.add_argument("-superfice_max", default = 200, type = float)

    #Base url of immobiliare
    parser.add_argument("-base_immobiliare_url", 
        default = "https://www.immobiliare.it/vendita-case/milano/con-ascensore/?criterio=rilevanza&superficieMinima=", type = str
    )

    #saving path 
    parser.add_argument("-path_immobiliare", default = "data/immobiliare/", type = str)

    #chrome path
    parser.add_argument("-chrome_path", default = 'chromedriver/chromedriver.exe',
                        type = str)

    #header of session
    parser.add_argument("-header",
     default = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.130 Safari/537.36',
                        type = str)


    args = parser.parse_args()
        
    #define url query
    query = {'proprieta': 'tipoProprieta=', 'stato': 'stato=', 'classe_energetica': 'classeEnergetica=',
             'piano': 'fasciaPiano[]='}
    
    
    #different query - list
    parameter_dic = {'proprieta': [1], 'stato': [1,6,2,5], 'classe_energetica': [2,5,8],
                     'piano': [10,20,30]}

    #get variable for immobiliare scraping
    parameter_df = expand_grid(parameter_dic)
    
    meaning_df = parameter_df.replace({'proprieta': {1: 'intera'},
                                       'stato': {1: 'nuovo_costruzione',6: 'ottimo_ristrutturato',2: 'buono_abitabile',5: 'da_ristrutturare'},
                                       'classe_energetica': {2: 'alta',5: 'media',8: 'bassa'},
                                       'piano': {10: 'piano_terra',20: 'piani_intermedi',30: 'ultimo_piano'}})
    meaning_df['ascensore'] = 'si'
    
    #PHASE -1 HOME SCRAPE PAGE 
    if args.scrape_all:
        print('Beginning first scraping')
        data, error_list = scraper(args = args, query = query,
                                parameter_df = parameter_df, meaning_df = meaning_df)
        
        with open(args.path_immobiliare + 'data.pkl', 'wb') as file:
            pickle.dump(data, file)
        
        with open(args.path_immobiliare + 'error.pkl', 'wb') as file:
            pickle.dump(error_list, file)
        
        print('Ended first scraping.... waiting 2 minute\n')
        time.sleep(120)
    
    else:
        with open(args.path_immobiliare + 'data.pkl', 'rb') as file:
            data = pickle.load(file)

    print(f'Beginning last scraping...\n{data.shape[0]} Total row to parse')

    # #PHASE -2 URL SCRAPE
    df_url = scraper_single_url(args, data)
    
    with open(args.path_immobiliare + 'data_url.pkl', 'wb') as file:
        pickle.dump(df_url, file)