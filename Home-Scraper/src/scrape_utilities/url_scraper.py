import pandas as pd 
from tqdm import tqdm
import time 
import gc
from scrape_utilities.scrape_fun import get_pages


def scraper_single_url(args, data):
    
    for i, url in tqdm(enumerate(data.Url)):

        if (i % 1000 == 0) & (i > 0):
            #wait 5 minutes
            time.sleep(300)

        time.sleep(args.time_w8)

        try:
            #get pages
            soup = get_pages(url = url, args = args)
            append = True

        except:

            #wait five minutes
            time.sleep(args.time_implicit_w8)

            try:
                soup = get_pages(url = url, args = args)
                append = True

            except:
                temp = pd.DataFrame({'Tipologia': [pd.NA], 'Anno_Costruzione': [pd.NA], 'Descrizione_Casa': [pd.NA]})
                append = False

        #append each new info
        if (i == 0) & append:
            df = get_url_info(soup)

        elif append:
            temp = get_url_info(soup)
            df = pd.concat([df, temp], axis = 0)

        else:
            df = pd.concat([df, temp], axis = 0)
            
    df = df.reset_index(drop = True)

    return(df)

def get_url_info(soup):
    
    try:
        #get tipologia
        box_ = soup.find_all('dt', {'class': 'col-xs-12 col-sm-4'})

        #cycle over each element when find tipologia get text
        for i, x in enumerate(box_):
            if x.get_text() =='Tipologia':
                pos = i
        
        tipologia = soup.find_all('dd', {'class': 'col-xs-12 col-sm-8'})[pos].get_text()
        del pos

    except:
        #assign na otherwise
        tipologia = pd.NA
        
    try:
        #find anno di costruzione by iterating over each element
        box_ = soup.find_all('dt', {'class': 'col-xs-12 col-sm-8'})

        for i, x in enumerate(box_):
            if x.get_text() =='Anno di costruzione':
                pos = i

        #assign anno
        anno_ = soup.find_all('dd', {'class': 'col-xs-12 col-sm-4'})[pos].get_text()
        del pos
    except:
        #assign na otherwise
        anno_ = pd.NA

    try:
        #get description
        text_ = soup.find('div', {'class': 'col-xs-12 description-text text-compressed'}).get_text()
    
    except:
        #assign na otherwise
        text_ = pd.NA
    
    #return data frame
    df = pd.DataFrame({'Tipologia': [tipologia], 'Anno_Costruzione': anno_, 'Descrizione_Casa': text_})

    return(df)