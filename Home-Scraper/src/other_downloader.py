import argparse
from scrape_utilities.aler_mtoday_scraper import aler_scraper, milanotd_scraper
import pickle
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #Waiting time between each requests
    parser.add_argument("-time_w8", default = 1, type = float)

    #Waiting time if not get 400
    parser.add_argument("-time_implicit_w8", default = 10, type = float)

    #timeout for response
    parser.add_argument("-timeout", default = 10, type = float)
    
    #saving path 
    parser.add_argument("-path_open", default = "data/openMilano/")

    #chrome path
    parser.add_argument("-chrome_path", default = 'chromedriver/chromedriver.exe',
                        type = str)

    #aler url
    parser.add_argument("-aler_url", default = 'https://aler.mi.it/ricerca-uog/', type = str)

    #MilanoTD url
    parser.add_argument("-milanotd_url", default = "http://www.milanotoday.it/mappa/tipo/", type = str)

    #parameter for MilanoTD
    parser.add_argument("-last_year_mtdy", default = '/data/ultimo-anno/', type = str)

    #parameter for kind of incident to scrape from milano today
    parser.add_argument("-incidente",
        default = ['incidenti', 'violenza-sessuale', 'rapine', 'furti', 'omicidi', 'prostituzione', 'droga', 'violenza', 'campi-nomadi'],
        type = list
    )

    #script to extract information from MilanoToday map
    parser.add_argument("-script_mtdy", 
        default = "return aItems;", type = str
    )

    #header of session
    parser.add_argument("-header",
     default = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.130 Safari/537.36',
                        type = str)


    args = parser.parse_args()
                    
    # #PHASE -3 ALER/MILANO TODAY SCRAPE
    #Aler
    print('Beginning Aler')
    aler_home = aler_scraper(args = args)

    print('Beginning Milano Today')
    #Milano Today
    milanotd_data = milanotd_scraper(args = args, map_url = args.milanotd_url, ultimo_anno = args.last_year_mtdy,
        incidente = args.incidente 
    )
    
    pickle.dump(aler_home, open(args.path_open + 'data_case_popolari.pkl', 'wb'))
    pickle.dump(milanotd_data, open(args.path_open + 'criminality_info.pkl', 'wb'))