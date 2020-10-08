import pandas as pd 
import numpy as np
from tqdm import tqdm
import time 
from itertools import product
import re
from scrape_utilities.scrape_fun import get_pages
import gc

def get_num_pages(soup):
    
    #Function which finds number of total pages
    try:
        pag_number_soup = soup.find("ul", {'class': 'pagination pagination__number'})
        num_pages = int((pag_number_soup.find_all('li'))[-1].get_text())
    except:
        num_pages = 1
        
    return(num_pages)

def expand_grid(dictionary):
    #function to create a grid
    return pd.DataFrame([row for row in product(*dictionary.values())],
                        columns=dictionary.keys())

def construct_url(param, query, superfice_min, superfice_max, base_immobiliare_url):
    #Create base url with max/min superficie
    url_base = base_immobiliare_url + str(superfice_min) + "&superficieMassima=" + str(superfice_max) 

    #iterate over param to add filter and create final url
    for col, value in param.iteritems():

        #if not nan add parameter
        if not np.isnan(value.values):
            url_base += '&' + query[col] + str(int(value))

    return(url_base)
    
#substitute lot of space and \n
def cleaner(text):
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\t", " ", text)
    text = re.sub(r"\s", " ", text)
    text = re.sub(r"\xa0", " ", text)
    text = re.sub(" ", "", text)
    return(text)

def find_price(box):
    #find all price, new price if exists and sconto
    try:
        old_price = cleaner(box.find('span', {'class': 'features__price-old--price'}).get_text())
        price = cleaner(box.find('li', {'class': 'lif__item lif__pricing'}).find('div').get_text())
        sconto = cleaner(box.find('span', {'class': 'features__price-old--discount'}).get_text())
        df = pd.DataFrame([{'Prezzo': price,'Prezzo_Vecchio': old_price, 'Sconto': sconto}])
    except:
        try:
            price = cleaner(box.find('li', {'class': 'lif__item lif__pricing'}).get_text())
            df = pd.DataFrame([{'Prezzo': price,'Prezzo_Vecchio': np.nan, 'Sconto': np.nan}])
        except:
            df = pd.DataFrame([{'Prezzo': np.nan,'Prezzo_Vecchio': np.nan, 'Sconto': np.nan}])
    return(df)

def find_room(box):
    #want exact match to not confuse with lif__item lif__pricing
    
    _ = box.find_all(lambda tag: tag.name =='li'and tag.get('class') == ['lif__item'])
    _ += box.find_all('li', {'class': 'lif__item hidden-xs'}) #adding piano

    room, metri, bath, piano = np.nan, np.nan, np.nan, np.nan
    
    for sub_box in _:

        label_ = cleaner(sub_box.find('div', {'class': 'lif__text lif--muted'}).get_text())
        text_ = cleaner(sub_box.find('div', {'class': 'lif__data'}).get_text())

        if label_ == 'locali':
            room = text_

        if label_ == 'superficie':
            metri = text_

        if label_ == 'bagni':
            bath = text_

        if label_ == 'piano':
            piano = text_

    return(pd.DataFrame([[room, metri, bath, piano]], columns = ['Stanze', 'Metri', 'Bagni', 'Piano']))
    
def find_garantito(box):
    try:
        garantito = cleaner(box.find('div', {'class': 'lif__text text-bold text-danger'}).get_text())
    except:
        garantito = np.nan
    return(pd.DataFrame({'Garantito': [garantito]}))

def photo_finder(box):
    try:
        url = box.find('div', {'class': 'showcase__item showcase__item--active'}).find('img')['src']
    except:
        url = np.nan
    return(pd.DataFrame({'Immagine': [url]}))

def boxer_finder(soup, meaning):
    df = pd.DataFrame([{'Prezzo': '', 'Prezzo_Vecchio': '', 'Sconto': '','Stanze': '', 'Metri': '',
                       'Bagni': '', 'Piano': '', 'Garantito': '', 'Posizione': '', 'Url': '', 'Immagine': ''}])
    
    for large in ['wide', 'medium', 'tiny']:

        box_list = soup.find_all('li',{'class': 'listing-item listing-item--' + large + ' js-row-detail'})

        if len(box_list)>0:
            for litag in box_list:

                #price
                price_df = find_price(litag)

                #room
                room_df = find_room(litag)

                #garantito
                garantito_df = find_garantito(litag)

                #Position
                posizione_df = pd.DataFrame({'Posizione': [litag.find('p', {'class': 'titolo text-primary'}).get_text().strip()]})

                #url 
                url_df = pd.DataFrame({'Url': [litag.find('p', {'class': 'titolo text-primary'}).find('a', href = True)['href']]})

                #image
                image_df = photo_finder(litag)

                #create new record
                df = df.append(pd.concat([price_df, room_df, garantito_df, posizione_df, url_df, image_df], axis = 1), ignore_index = True)
                
    df = df.iloc[1:,:].reset_index(drop = True)
    df = supplementar_info(df, meaning)
    return(df)

def supplementar_info(df, meaning):
    n = df.shape[0]
    add = meaning.loc[[0]*n].reset_index(drop = True)
    df = pd.concat([df, add], axis = 1)
    return(df)


def scraper(args, query, parameter_df, meaning_df):
    
    error_list = []

    for section in tqdm(range(parameter_df.shape[0])):
        
        #get first page and num_pages
        parameter = parameter_df.loc[[section]].reset_index(drop = True)
        meaning = meaning_df.loc[[section]].reset_index(drop = True)

        base_url = construct_url(param = parameter, query = query,
         superfice_min = args.superfice_min, superfice_max = args.superfice_max,
         base_immobiliare_url = args.base_immobiliare_url
         )
        
        soup = get_pages(url = base_url, args = args)
        
        #get number of pages
        num_pages = get_num_pages(soup)

        if section == 0:
            df = boxer_finder(soup, meaning)
        else:
            df = pd.concat([df, boxer_finder(soup, meaning)], axis = 0)

        if num_pages > 1:
            for i in np.arange(2, num_pages + 1):
                                    
                #waiting time between each query
                time.sleep(args.time_w8)

                #get url
                temp_url = base_url+ '&pag=' + str(i)

                #get soup
                try:
                    soup = get_pages(url = temp_url, args = args)

                    new_df = boxer_finder(soup, meaning)
                    df = pd.concat([df, new_df], axis = 0)
                except:
                    try:
                        #wait five minutes
                        time.sleep(args.time_implicit_w8)
                        
                        soup = get_pages(url = temp_url, args = args)

                        new_df = boxer_finder(soup, meaning)
                        df = pd.concat([df, new_df], axis = 0)
                    except:
                        error_list += [{'url': temp_url, 'parameter': parameter, 'meaning': meaning}]

    return(df, error_list)
