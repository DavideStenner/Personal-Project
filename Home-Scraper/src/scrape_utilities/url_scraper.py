import pandas as pd 
from tqdm import tqdm
import time 
import gc
from scrape_utilities.scrape_fun import get_pages


def scraper_single_url(args, data):
    for i, url in tqdm(enumerate(data.Url[235:])):
		
        if (i % 1000 == 0) & (i > 0):
            #wait 5 minutes
            time.sleep(300)

        time.sleep(args.time_w8)

        try:
            #get pages
            soup = get_pages(url = url, args = args)
            temp = get_url_info(soup)

        except:

            #wait five minutes
            time.sleep(args.time_implicit_w8)

            try:
                soup = get_pages(url = url, args = args)
                temp = get_url_info(soup)
                
            except:
                temp = pd.DataFrame(
                    {
                        'Tipologia': [pd.NA], 'Anno_Costruzione': [pd.NA], 'Descrizione_Casa': [pd.NA],
                        'Stato': [pd.NA], 'Totale Piani': [pd.NA], 'Contratto': [pd.NA], 
                        'disponibilità': [pd.NA]
                    }
                )

        #append each new info
        if (i == 0):
            df = temp

        else:
            df = pd.concat([df, temp], axis = 0)
            
    df = df.reset_index(drop = True)

    return(df)

def get_url_info(soup):
    descrizione, anno, stato, totale_piani, contratto, disponibilità, tipologia = [pd.NA] * 7
    
    box_ = soup.find('dl', {'class': 'im-features__list'})
    
    title_ = box_.find_all('dt', {'class': 'im-features__title'})
    element_ = box_.find_all('dd', {'class': 'im-features__value'})
    descrizione = soup.find('div', {'class': 'im-description__text js-readAllText'}).get_text().lower().strip()

    
    try:
        efficienza = box_.find('span', {'class': 'im-features__energy'}).get_text().lower().strip()
    except:
        efficienza = pd.NA

    for i in range(len(title_)):
        
        if title_[i].get_text().lower().strip() == 'anno di costruzione':
            anno = element_[i].get_text().lower().strip()

        elif title_[i].get_text().lower().strip() == 'stato':
            stato = element_[i].get_text().lower().strip()

        elif title_[i].get_text().lower().strip() == 'totale piani edificio':
            totale_piani = element_[i].get_text().lower().strip()

        elif title_[i].get_text().lower().strip() == 'contratto':
            contratto = element_[i].get_text().lower().strip()

        elif title_[i].get_text().lower().strip() == 'disponibilità':
            disponibilità = element_[i].get_text().lower().strip()

    #return data frame
    df = pd.DataFrame(
        {
            'Tipologia': [tipologia], 'Anno_Costruzione': [anno], 'Descrizione_Casa': [descrizione],
            'Stato': [stato], 'Totale Piani': [totale_piani], 'Contratto': [contratto], 
            'disponibilità': [disponibilità]
        }
    )

    return(df)