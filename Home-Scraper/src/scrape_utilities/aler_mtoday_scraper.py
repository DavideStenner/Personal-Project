import pandas as pd 
import numpy as np
from scrape_utilities.scrape_fun import get_pages, selenium_proxy_connector
from tqdm import tqdm
import time 

#scraper for Aler Home
def aler_scraper(args):

    def parse_home(x):
        try:
            #find casa
            city = casa.find('td', {'class':'column-2'}).get_text()

            #find address
            address = casa.find('td', {'class':'column-3'}).get_text()

            #find number
            number = casa.find('td', {'class':'column-4'}).get_text()

            #return dataframe format
            return(pd.DataFrame({'city': [city], 'address': [address], 'number_address': [number]}))

        except:
            #return na dataframe
            return(pd.DataFrame({'city': [pd.NA], 'address': [pd.NA], 'number_address': [pd.NA]}))
    
    #get page
    soup = get_pages(url = args.aler_url, args = args)
    
    #find table

    table = soup.find_all('table',{'class': 'tablepress tablepress-id-8 tablepress-responsive'})[0]

    #find all home
    lista_case_popolari = table.find_all('tr')
    
    #cycle over each home and append
    for _, casa in enumerate(lista_case_popolari):

        if _ == 0:
            data = parse_home(casa)
        else:
            data = data.append(parse_home(casa), ignore_index = True)
    
    #drop missing row
    data = data.dropna().reset_index(drop = True)

    #lower all city text
    data['city'] = data.city.apply(lambda x: x.lower())

    #return only Milano
    data = data.loc[data.city == 'milano'].reset_index(drop = True)

    return(data)
    
def milanotd_scraper(args, map_url, ultimo_anno, incidente):

    #create data structure
    data = {x: 'nulla' for x in incidente}
        
    for tipologia in tqdm(incidente):

            #create url
            url = map_url + tipologia + ultimo_anno

            data[tipologia] = selenium_proxy_connector(url = url, args = args, script = args.script_mtdy)

            time.sleep(args.time_w8)

    return(data)




