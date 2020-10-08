import pandas as pd
import numpy as np
import re
from nltk import word_tokenize
import nltk
from others.logging_utils import init_logger
from itertools import chain
import geojson
import json
from geopy import distance
from pandas.io.json import json_normalize
from tqdm import tqdm

def etl_1(data, url_):
    #function which return anno in number othwerwise null
    def Anno_cleaner(x):
        try:
            return(float(x))
        except:
            return(np.nan)
    
    #check if price has da inside price and return --> "Asta" otherwise "no_asta"
    def asta(x):
        asta = 'no_asta'
        try:
            if 'da' in x:
                asta = 'asta'
        except:

            return(asta)
        
        return(asta)
    
    #Clean price from.. (Da, Symbol, .)
    def clean_price(text):
        try:
            text = re.sub("da", "", text)
            text = re.sub("€", "", text)
            text = re.sub(r'\.', '', text)

        except:
            
            return(text)
        
        return(text)
    
    #Function which clean sconto by taking out parenthesis, %, -
    def clean_sconto(text):
        try:
            text = re.sub(r"\(", "", text)
            text = re.sub(r"\)", "", text)
            text = re.sub(r'%', '', text)
            text = re.sub(r'-', '', text)
        
        except:

            return(text)
        return(text)
    
    #Function which clean metri by taking out m2
    def clean_metri(text):

        try:
            text = re.sub(r'm2','', text)
        
        except:
            return(text)
        
        return(text)
    
    #function which fill NA with mancante
    # def missing_filler(data, char, label = 'mancante'):

    #     for col in char:
    #         data[col] = data[col].fillna('mancante')
        
    #     return(data)
    
    #Clean out from every special character in special_list
    def clean_special(x):
        special_list = [r'\:', r'\.', r'\-', r'\_', r'\;', r'\,', r'\'']

        for symbol in special_list:
            x = re.sub(symbol, ' ', x)

        return(x)
    
    #find position from description
    def position_cleaner(x):
        
        def cl1(x):
            x = re.sub(r'\,', '', x)
            x = re.sub(r' +', ' ', x)
            return(x)

        x = re.sub(r'(\,) +\d+', lambda s: cl1(s.group()), x)
        return(x)
    
    #clean string
    def formatter(x):
        x = x.strip()
        x = re.sub(r'\s+', ' ', x)
        return(x)
    
    #Clean error from short name
    def error_cleaner(x):
        x = re.sub(r'v\.le', 'viale', x)    
        return(x)
    
    #
    def address_exctractor(x):
        termini_ = ['via privata', 'via', 'viale', 'piazzetta', 'foro', 'cavalcavia',
                    'giardino', 'vicolo', 'passaggio', 'sito', 'parco', 'sottopasso',
                    'piazza', 'piazzale', 'largo', 'corso', 'alzaia', 'strada', 'ripa',
                    'galleria', 'foro', 'bastioni']
        
        x = x.lower()

        #find position
        x = position_cleaner(x)

        #clean error
        x = error_cleaner(x)

        #find address after termini_
        address = ''
        for lab_ in termini_: 

            #search for match
            temp = re.search(r'\b%s\b' %lab_, x)

            #find address by matching
            if (temp is not None):
                
                temp = re.search(r'%s (.*?)\,' %lab_, x)
                address = clean_special(temp.group(0))        
                continue
        #clean ending string    
        address = formatter(address)   
        return(address)
    
    #take out number from address to get nome via
    def nome_via(x):
        return(formatter(re.sub(r'\d+', '', x)))
    
    #take out text and keep number
    def numero_via(x):
        x = x.lower()
        x = re.sub('via 8 ottobre 2001', '', x) #via 8 ottobre exception
        digit = re.search(r'\d+', x)
        try:
            x = digit.group()
        except:
            return('')
        
        return(re.sub(r'\s+', '', x))
    
    # char = ['Stanze', 'Bagni', 'Piano', 'Garantito', 'stato', 'classe_energetica', 'piano']
    
    data = data.reset_index(drop = True)
    url_ = url_.reset_index(drop = True)
    
    #Clean Anno
    url_['Anno_Costruzione'] = url_['Anno_Costruzione'].apply(lambda x: Anno_cleaner(x))
    url_['Anno_Costruzione'] = url_['Anno_Costruzione'].convert_dtypes()
    
    data = pd.concat([data, url_], axis = 1)
    
    #Clean Prezzo
    data['asta'] = data['Prezzo'].apply(lambda s: asta(s))
    data['Prezzo'] = data['Prezzo'].apply(lambda s: clean_price(s)).astype(float)
    data['Prezzo_Vecchio'] = data['Prezzo_Vecchio'].apply(lambda s: clean_price(s)).astype(float)
    data['Sconto'] = data['Sconto'].apply(lambda s: clean_sconto(s)).astype(float)

    #Clean Metri
    data['Metri'] = data['Metri'].apply(lambda s: clean_metri(s)).astype(float)
    data['Prezzo_al_mq'] = data['Prezzo']/data['Metri']

    #Clean Piano
    data['Piano'] = data['Piano'].replace({'T': 'Terra', 'R': 'Piano Rialzato', 'S': 'Seminterrato', 'A': 'Ultimo'})
    # data = missing_filler(data, char)
    
    #extract Indirizzo, Nome Via and numero via
    data['indirizzo'] = data['Posizione'].apply(lambda x: address_exctractor(x))
    data['nome_via'] = data.indirizzo.apply(lambda s: nome_via(s))
    data['numero_via'] = data.indirizzo.apply(lambda s: numero_via(s))

    return(data)

def etl_2(args, data):
    
    #Function which calculate intersection score betweem 
    def scorer(segment_1, segment_2, missing_pos, indirizzo, original, logger):
        vec = []
        
        #cycle over each missing position
        for m_1 in missing_pos:

            
            vec_2 = np.zeros(indirizzo.shape[0])

            #calculate intersection between segment_1, segment_1 to normalize
            intersection_top = segment_1[m_1] & segment_1[m_1]

            #calculate score of intersection to normalize
            top_ = score_intersection(intersection_top)

            #iterate over each indirizzo to calculate score of intersection
            for m_2 in range(indirizzo.shape[0]):

                #calculate intersection set
                intersection_try = segment_1[m_1] & segment_2[m_2]

                #calculate score
                vec_2[m_2] = score_intersection(intersection_try)

            #find max
            max_ = np.max(vec_2)
            
            #count how many are equal to max score
            len_max = np.sum(vec_2 == max_)

            #if normalize score assign new indirizzo
            if max_/top_ > args.treshold:


                if len_max>1:
                    #in case of ties take indirizzo with nearest number address
                    number_ = number_intersection(segment_1[m_1], segment_2[vec_2 == max_].values)

                    #find which address is selected
                    pos = (np.where(vec_2 == max_)[0])[number_]

                    #add indirizzo
                    vec += [indirizzo[pos]]

                    #print correction with score
                    logger.info('Segmento errore: {}; via scelta: {}; Match: {}'.format(original[m_1], indirizzo[pos], max_/top_))
                    
                else:
                    #assign indirizzo with max score
                    vec += [indirizzo[np.argmax(vec_2)]]

                    logger.info('Via originale: {}; Via scelta: {}; Match: {}'.format(original[m_1],
                                                                        indirizzo[np.argmax(vec_2)], max_/top_))
        
            else:
                vec += [np.nan]
                
                logger.info('errore no match, score {} -- via originale: {}; Matched: {}'.format(max_/top_, original[m_1],
                                                                    indirizzo[np.argmax(vec_2)]))        
                
        #this home didn't find any real address to match up
        logger.info('\n\n''{} of home deleted cause error in address typing\n\n'.format(np.sum([pd.isna(x) for x in vec])))

        return(vec)
    
    #replace special character with space
    def special_delete(x, punctuations = '@#!?+&*[]-%.:/();$=><|{}^' + "'`"):
        
        for p in punctuations:
            x = x.replace(p, ' ')
    
        x = x.replace('è', 'e')
        x = x.replace('é', 'e')
        x = x.replace('ù', 'u')
        x = x.replace('à', 'a')
        x = x.replace('ò', 'o')
        x = x.replace('ì', 'i')
        
        x = re.sub(r"([0-9]+(\.[0-9]+)?)",r" \1 ", x)
        return(x)
    
    #extract number
    def exctract_number(x):
        try:
            return(re.search(r'\b\d+\b', x).group(0))
        except:
            return('')
    
    #clean number of nil
    def number_nil_clean(x):
        x = re.sub(r'\\+\d+', '', x)
        x = re.sub(r'\/+\d+', '', x)
        x = re.sub(r'[a-zA-Z]+\d+', '', x)
        x = re.sub(r'[a-zA-Z]+', '', x)
        return(x)
    
    #replace special punctuations and accented letters
    def special_space(x, punctuations = '@#!?+&*[]-%.:/();$=><|{}^' + "'`"):
        
        for p in punctuations:
            x = x.replace(p, f' {p} ')
    
        x = x.replace('è', 'e')
        x = x.replace('é', 'e')
        x = x.replace('ù', 'u')
        x = x.replace('à', 'a')
        x = x.replace('ò', 'o')
        x = x.replace('ì', 'i')
        
        x = re.sub(r"([0-9]+(\.[0-9]+)?)",r" \1 ", x)
    
        return(x)
    
    #little clean for f.lli
    def abbreviazioni_replace(x):
        x = x.replace('f.lli', 'fratelli')
        return(x)
    
    #aler cleaner --> to calculate intersection
    def aler_formatter(x):
        x = x.lower()
        x = word_tokenize(x)
        x = sorted(x)
        x = ' '.join(x)
        return(x)
    
    #Function which give 0.5 for each common digit and 1 for each common word
    def score_intersection(intersection):
        number = 0
        word = 0

        for x in intersection:
            if x.isdigit():
                number += 1
            else:
                word += 1
        
        return(number*.5 + word)
    
    #calculate number of intersection in case of ties for same indirizzo
    def number_intersection(fake, possibilities):
        number_list = []

        #cycle over each possible indirizzo
        for x in possibilities:
            
            #take out everything apart form number
            try:
                number_list += [float(re.search(r'\d+', ' '.join(x)).group())]

            #if no number then np.inf
            except:

                number_list += [np.inf]
        #take out everything apart form number
        try:
            number_fake = float(re.search(r'\d+', ' '.join(fake)).group())
        
        #if it has no number assign the median of each indirizzo in possibilities
        except:

            #calculate median over each number of address
            mode = median_modded(number_list)

            #find correct address
            mask = [x == mode for x in number_list]
            pos = np.where(mask)[0]
            
            #take indirizzo text
            if len(pos)>0:
                return(pos[0].item())
            else:
                return(pos.item())
        
        #take indirizzo nearest to the one provided in fake
        result = [abs(x - number_fake) for x in number_list]

        #calculate final indirizzo
        final_indirizzo = np.argmin(result)

        return(final_indirizzo)
        
    #calculate median of number list
    def median_modded(lst):
        #sort
        sortedLst = sorted(lst)
        lstLen = len(lst)
        
        #take median element
        index = (lstLen - 1) // 2


        return sortedLst[index]
    
    logger_aler = init_logger(log_file = args.path_etl2_log_aler)

    #filter out home without any indirizzo
    data = data.loc[data.indirizzo != ''].reset_index(drop = True)
    
    #read open dataset
    #nil
    nil = pd.read_csv(args.path_datasetMilano + 'ds634_civici_quartieri_zone_nil_cap_vie_20180518-unicode.csv', delimiter =';')
    
    #aler list
    aler_home = pd.read_pickle(args.path_openMilano + 'data_case_popolari.pkl')
    
    #nil geo
    nil_geo = pd.read_csv(args.path_datasetMilano + 'ds634_civici_coordinategeografiche_20200203.csv', delimiter =';')
    
    
    #extract number address 
    aler_home['number_address'] = aler_home.number_address.apply(lambda x: exctract_number(x))

    #clean indirizzo by concatenate address and number address
    aler_home['indirizzo'] = aler_home.address + ' ' + aler_home.number_address

    #special aler formatter --> for later to calculate score of intersection with address lists
    aler_home['indirizzo'] = aler_home.indirizzo.apply(lambda x: aler_formatter(x))

    #calculate mean for long and lat because we have more long lat for each address
    nil_geo_mean = nil_geo.loc[:, ['NUMEROCOMPLETO', 'DENOMINAZIONE', 'LONG_WGS84', 'LAT_WGS84']].reset_index(drop = True).groupby(['NUMEROCOMPLETO', 'DENOMINAZIONE']).mean()
    nil_geo = pd.DataFrame(nil_geo_mean).reset_index()
    
    #change numero completo to str
    nil_geo['NUMEROCOMPLETO'] = nil_geo['NUMEROCOMPLETO'].astype(str)
    
    #merge back to nil new lat, long information
    nil = nil.merge(nil_geo, how = 'left',
                    left_on = ['DENOMINAZIONE VIA', 'NUMERO CIVICO'],
                    right_on = ['DENOMINAZIONE', 'NUMEROCOMPLETO'])

    #take out row with long null (lat will be null also)
    nil = nil[~nil.LONG_WGS84.isnull()].reset_index(drop = True)
    
    #little clean of DUG
    nil.DUG = nil.DUG.apply(lambda s: s.lower())

    #little clean and extraction of numero civico
    nil['NUMERO CIVICO'] = nil['NUMERO CIVICO'].apply(lambda s: number_nil_clean(s))

    #little clean of denominazione via
    nil['DENOMINAZIONE VIA'] = nil['DENOMINAZIONE VIA'].apply(lambda s: s.lower())

    #Addedd indirizzo by concatenation of dug, denominazione via and numero civico
    nil['indirizzo'] = nil['DUG'] + ' ' + nil['DENOMINAZIONE VIA'] + ' ' + nil['NUMERO CIVICO']

    #drop each duplicates
    nil = nil.loc[nil[['DENOMINAZIONE VIA', 'indirizzo', 'NUMERO CIVICO']].drop_duplicates().index].reset_index(drop = True)

    #apply special space to add space to each special character --> word_tokenize --> sort --> join with ' ' to create join key
    data['join_key'] = data.indirizzo.apply(lambda s: ' '.join(sorted(word_tokenize(special_space(s)))))
    nil['join_key'] = nil.indirizzo.apply(lambda s: ' '.join(sorted(word_tokenize(special_space(abbreviazioni_replace(s))))))

    
    ################# ALER CHECKER
    #join with nil to add to aler LONG, LAT
    temp = aler_home.merge(nil[['join_key', 'LONG_WGS84', 'LAT_WGS84']],
               how = 'left', left_on = 'indirizzo', right_on = 'join_key')['LONG_WGS84']
    
    #find which LONG is missing
    missing_pos = np.where(temp.isnull())[0].tolist()
    
    #special cleaning for aler wich deletes special characters --> word_tokenize --> create set
    segment_1 = aler_home.indirizzo.apply(lambda s: set(word_tokenize(special_delete(s))))
    segment_2 = nil.indirizzo.apply(lambda s: set(word_tokenize(special_delete(abbreviazioni_replace(s)))))
    
    #calculate corrected indirizzo by checking which indirizzo from nil have higher score with aler_home indirizzo by checking intersection of word/number
    logger_aler.info('*'*100 + '\n\nBeginning scorer for aler\n\n')
    aler_home.loc[missing_pos, 'indirizzo'] = scorer(segment_1 = segment_1,
                                                     segment_2 = segment_2,
                                                     indirizzo = nil.indirizzo,
                                                     original = aler_home.indirizzo,
                                                     missing_pos = missing_pos,
                                                     logger = logger_aler)
    
    #take out every row with missing address after correction 
    mask_aler = data.indirizzo.isnull()
    aler_home = aler_home.loc[~mask_aler].reset_index(drop = True)
    
    #little clean and join with nil after correction of address
    aler_home['indirizzo'] = aler_home.indirizzo.apply(lambda x: aler_formatter(x))

    #join with nil
    aler_home = aler_home.merge(nil[['LONG_WGS84', 'LAT_WGS84', 'join_key']], how = 'left',
                    left_on = ['indirizzo'],
                    right_on = ['join_key'])
    
    #drop join key and save
    aler_home = aler_home.drop('join_key', axis = 1)

    ######################### SINGLE ERROR CHECK
    #calculate set over join_key for scraped dataset and nil
    fakeword = set(word_tokenize(' '.join(data.join_key)))
    realword = set(word_tokenize(' '.join(nil.join_key)))

    #list of real word
    realword_list = list(realword)
    
    #check to correct mispel from scraped address
    
    #find word which are inside fakeword but not in realword
    mispell = fakeword ^ realword & fakeword
    
    #it's a misple if it's a word of 3 or more caracters
    mispell = [x for x in mispell if len(x)>3]
    
    #find which words to delete
    to_del = []

    logger_data = init_logger(log_file = args.path_etl2_log_data)

    logger_data.info('*'*100 + '\n\nBeginning Mispell Correction\n\n')

    #cycle over each mispel and calculate edit_distance with nltk
    for mis in mispell:

        #calculate edit_distance with each real_word
        dist_list = [nltk.edit_distance(x, mis) for x in realword_list]

        #take min correction--> in case of ties select the first one
        correct = realword_list[np.argmin(dist_list)]

        #if mispel has distance equal to 1 correct
        if np.min(dist_list)==1:

            #print Mispel and correction
            logger_data.info('Mispell: {}, Correct: {}'.format(mis, correct))

            #if corrected cycle over each word and replace mispel
            for r in range(data.shape[0]):

                #replace mispel with correction
                data.loc[r, 'indirizzo'] = data.loc[r, 'indirizzo'].replace(f'{mis}', f'{correct}')

            #add mispel corrected to list
            to_del += [mis]  
    
    #take out row with uncorrected mispel
    row_with_mispell = [x for x in mispell if x not in to_del]

    data = data[[np.sum([y in x for y in row_with_mispell])==0 for x in data.indirizzo]].reset_index(drop = True)
    
    #special cleaning to create join_key
    data['join_key'] = data.indirizzo.apply(lambda s: ' '.join(sorted(word_tokenize(special_space(s)))))
    
    #check if there are new word wich doesn't match with real list
    joined_set = set(word_tokenize(' '.join(data['join_key'])))
    joined_error = (joined_set ^ realword) & joined_set

    for x in joined_error:
        if len(x)>4:
            print(f'Problem: {x}')
    
    ###########################

    #merge with nil to get NIL
    temp = data.merge(nil[['join_key','QUARTIERE', 'MUNICIPIO', 'ID_NIL', 'NIL']],
           how = 'left', left_on = 'join_key', right_on = 'join_key')['NIL']

    #take out NIL position
    missing_pos = np.where(temp.isnull())[0].tolist()

    #calculate (after mispel correction) set to calculate score of intersection
    segment_1 = data.indirizzo.apply(lambda s: set(word_tokenize(special_delete(s))))
    segment_2 = nil.indirizzo.apply(lambda s: set(word_tokenize(special_delete(abbreviazioni_replace(s)))))
    
    #calculate score of intersection
    logger_data.info('*'*100 + 'Beginning scorer for scraped dataset\n\n')
    data.loc[missing_pos, 'indirizzo'] = scorer(segment_1 = segment_1,
                                                 segment_2 = segment_2,
                                                 indirizzo = nil.indirizzo,
                                                 original = data.indirizzo,
                                                 missing_pos = missing_pos,
                                                 logger = logger_data)

    #take out null address
    data = data[~data.indirizzo.isnull()].reset_index(drop = True)
    
    #create join_key
    data['join_key'] = data.indirizzo.apply(lambda s: ' '.join(sorted(word_tokenize(special_space(s)))))

    #merge with nil
    data = data.merge(nil[['join_key','QUARTIERE', 'MUNICIPIO', 'ID_NIL', 'NIL', 'LONG_WGS84', 'LAT_WGS84']],
           how = 'left', left_on = 'join_key', right_on = 'join_key')
    
    return(data, aler_home)


def etl_geo(args, data):

    #retain only corrected store... Esselunga abc --> Esselunga
    def correct_store(store, supermercati):
        for sup in supermercati:
            if sup in store:
                return(sup)
        return(store)

    def seconda_linea(x):
        if len(x) == 1:
            return('')
        else:
            return(re.sub(r'.*\,', '', x))
    
    #take first element
    def take_first(x):
        while True:
            dim = list(np.array(x).shape)
            if len(dim) == 2:
                return(x)
            x = x[0]
    

    #calculate lowest distance to given element
    def distanza_home_element(casa, element, long_label, lat_label):

        #calculate long, lat of home
        long_casa, lat_casa = casa.LONG_WGS84, casa.LAT_WGS84
        
        vec = []
        
        #calculate each distance to every store of same categories
        for _, row in element.iterrows():

            long_element, lat_element = row[long_label], row[lat_label]
            dist = distance.distance((lat_casa, long_casa), (lat_element, long_element)).kilometers

            vec += [dist]
        
        result = np.min(vec)

        return(result)

    #calculate nearest distance vector
    def dist_df(data, element, filter_var = None, label_filter = None, long_label = 'LONG', lat_label = 'LAT'):
        vec = []

        #keep only element after filter
        if (filter_var is not None) & (label_filter is not None):
            element = element.loc[element[filter_var] == label_filter].reset_index(drop = True)
        
        if (filter_var is not None) ^ (label_filter is not None):
            raise ValueError("filter or label filter missing")

        for _, row in tqdm(data.iterrows()):
            vec += [distanza_home_element(row, element, long_label, lat_label)]

        return(vec)

    #count how many stores are inside the selected radius
    def radius_df(data, element, radius, filter_var = None, filter_label = None, long_label = 'LONG', lat_label = 'LAT'):
        vec = []

        #keep only element after filter
        if (filter_var is not None) & (filter_label is not None):
            element = element.loc[element[filter_var] == filter_label].reset_index(drop = True)
        
        if (filter_var is not None) ^ (filter_label is not None):
            raise ValueError("filter or label filter missing")

        for _, row in tqdm(data.iterrows()):
            vec += [sum_inside_radius_df(row, element, radius, long_label, lat_label)]

        return(vec) 


    #calculate how many supermercati are inside radius
    def sum_inside_radius_df(casa, element, radius, long_label, lat_label):
        long_casa, lat_casa = casa.LONG_WGS84, casa.LAT_WGS84
                
        vec = []
        #find distance of each home-store
        for _, row in element.iterrows():

            long_store, lat_store = row[long_label], row[lat_label]

            vec += [distance.distance((lat_casa, long_casa), (lat_store, long_store)).kilometers]

        #find how many store are nearest than radius
        vec = [x <= radius for x in vec]

        result = np.sum(vec)

        return(result)

    #calculate number of reati inside selected radius of a selected reato
    def radius_json(data, element, radius, filter_label = None):
        vec = []

        #keep only element after filter
        if (filter_label is not None):
            element = element[filter_label]

        for _, row in tqdm(data.iterrows()):
            vec += [sum_inside_radius_json(row, element, radius)]
        return(vec)


    #calculate how many reati were done inside selected radius from the seleted home of a selected reato
    def sum_inside_radius_json(casa, element, radius):
        long_casa, lat_casa = casa.LONG_WGS84, casa.LAT_WGS84
        
        vec = [distance.distance((lat_casa, long_casa), (lat_store, long_store)).kilometers < radius for lat_store, long_store in element]
        
        result = np.sum(vec)
        return(result)


    #drop missing lat, long
    mask = (data.LONG_WGS84.isnull()) | (data.LONG_WGS84.isnull())
    data = data[~mask].reset_index(drop = True)

    try:
        
        missing_file = 'economia_media_grande_distribuzione_coord.csv'
        negozi = pd.read_csv(args.path_datasetMilano + 'economia_media_grande_distribuzione_coord.csv', delimiter =',')

        missing_file = 'ds634_civici_coordinategeografiche.csv'
        nil_geo = pd.read_csv(args.path_datasetMilano + 'ds634_civici_coordinategeografiche.csv', delimiter =',')

        missing_file = 'tpl_metrofermate.geojson'
        with open(args.path_datasetMilano + 'tpl_metrofermate.geojson') as f:
            fermate_json = json.load(f)

        missing_file = 'parchi.geojson'
        with open(args.path_datasetMilano + 'parchi.geojson') as f:
            parchi_json = json.load(f)

        missing_file = 'scuoleinfanzia.geojson'
        with open(args.path_datasetMilano + 'scuoleinfanzia.geojson') as f:
            scuole_infanzia_json = json.load(f)

        missing_file = 'scuoleprimarie.geojson'
        with open(args.path_datasetMilano + 'scuoleprimarie.geojson') as f:
            scuole_primarie_json = json.load(f)

        missing_file = 'scuolesecondarie1grado.geojson'
        with open(args.path_datasetMilano + 'scuolesecondarie1grado.geojson') as f:
            scuole_secondarie_json = json.load(f)

        missing_file = 'scuolesecondarie2grado.geojson'
        with open(args.path_datasetMilano + 'scuolesecondarie2grado.geojson') as f:
            scuole_secondarie_2_json = json.load(f)

        missing_file = 'criminality_info.pkl'
        criminality = pd.read_pickle(args.path_openMilano + 'criminality_info.pkl')

        del missing_file
    except:
        print(f'Missing file: {missing_file}\n')

    #create dictionary of news: gpslocation
    criminality = {x: [y['gps'] for y in criminality[x] if y['gps'] is not None] for x in criminality.keys()}

    #unlist all criminalty dictionary to create a list of all coordinate
    criminality_all = list(chain(*[x for _, x in criminality.items()]))

    #drop join_key
    data = data.drop('join_key', axis = 1)
    #NEGOZI

    #lowe cleaning 
    negozi['settore_merceologico'] = negozi['settore_merceologico'].apply(lambda x: str(x).lower())
    negozi['insegna'] = negozi['insegna'].apply(lambda x: str(x).lower())
    negozi['DescrizioneVia'] = negozi['DescrizioneVia'].apply(lambda x: str(x).lower())

    #correct store depending on supermercati
    negozi['insegna_corretta'] = negozi['insegna'].apply(lambda x: correct_store(x, args.supermercati))

    #keep only supermercati inside supermercati list
    negozi = negozi[[x in args.supermercati for x in negozi['insegna_corretta']]]

    #cleaning of columns and create mean of lat, long by description --> have only one value
    nil_geo['DESCRIZIONE'] = (nil_geo.TIPO + ' ' + nil_geo.DENOMINAZIONE).apply(lambda x: str(x).lower())
    nil_geo = nil_geo[['DESCRIZIONE', 'LONG_WGS84', 'LAT_WGS84']].groupby('DESCRIZIONE').mean().reset_index()

    #take out null rows
    temp = negozi[negozi.LAT_WGS84.isnull()].copy()

    #merge negozi with nil_geo to take coordinate
    new_value = temp.merge(nil_geo, how = 'left', left_on = 'DescrizioneVia', right_on = 'DESCRIZIONE').iloc[:,-2:].values

    #assign new lat, long value
    negozi.loc[negozi.LAT_WGS84.isnull(),'LONG_WGS84'] = new_value[:,0]
    negozi.loc[negozi.LAT_WGS84.isnull(),'LAT_WGS84'] = new_value[:,1]

    #filter null row
    negozi = negozi[~negozi['LONG_WGS84'].isnull()]

    #check distance to store
    print('Beginning Supermercati\n')
    
    for store in args.supermercati:
        print(f'\nStore: {store}\n')
        data[store+'_distanza'] = dist_df(data, negozi, filter_var = 'insegna', label_filter = store, long_label = "LONG_WGS84", lat_label = "LAT_WGS84")


    #count how many stores are in radius of radius_list
    radius_list = [.25, .5, 1]
    
    print('Beginning Supermercati radius\n')

    for kilometer in radius_list:
        print(f'\nRadius: {kilometer}\n')
        data['store_radius_' + str(kilometer) + "_km"] = radius_df(data, negozi, kilometer, long_label = "LONG_WGS84", lat_label = "LAT_WGS84")

    #find minimum distance of each store to a selected home
    mask_column = [x for x in data.columns if re.search('distanza', x)]
    data['distanza_minima_supermercato'] = data[mask_column].min(axis = 1)

    #find supermercato più vicino
    mask_column = [x for x in data.columns if re.search('distanza', x)]
    data['supermercato_vicino'] =  data[mask].idxmin(axis = 1).apply(lambda x: re.sub('_distanza', '', x))

    #clean fermate_json
    fermate = json_normalize(fermate_json['features']).drop(['type', 'properties.id', 'geometry.type'], axis = 1)

    #rename columns
    fermate = fermate.rename(columns = {'properties.nome': 'nome', 'properties.linee': 'linee', 'geometry.coordinates': 'coordinates'})

    #clean linee1
    fermate['linee1'] = fermate.linee.apply(lambda x: re.sub(r'\,.*', '', x))
    fermate['linee2'] = fermate.linee.apply(lambda x: seconda_linea(x))

    #take lat, long from coordinates
    fermate['LONGITUDE'] = fermate.coordinates.apply(lambda x: x[0])
    fermate['LATITUDE'] = fermate.coordinates.apply(lambda x: x[1])

    #calculate distance to metro and concatenate
    print('Beginning Metro\n')

    fermate['distanza_metro'] = dist_df(data, fermate, filter_var = None, label_filter = None, long_label = "LONGITUDE", lat_label = "LATITUDE")
    fermate = fermate[['nome', 'linee', 'linee1', 'linee2', 'distanza_metro']]
    data = pd.concat([data, temp], axis = 1)

    #json normalize parchi dataset
    parchi = json_normalize(parchi_json['features'])
    parchi = parchi.drop(['type', 'properties.ZONA', 'geometry.type', 'properties.PERIM_M', 'properties.AREA'], axis = 1)

    #rename dataset
    parchi = parchi.rename(columns = {'properties.AREA_MQ': 'area_parco',
                            'properties.PARCO': 'parco', 'geometry.coordinates': 'coordinates'})


    parchi.coordinates = parchi.coordinates.apply(lambda x: np.array(take_first(x)).mean(axis = 0))

    #calculate longitude/latitude
    parchi['LONGITUDE'] = parchi.coordinates.apply(lambda x: x[0])
    parchi['LATITUDE'] = parchi.coordinates.apply(lambda x: x[1])

    #drop small park
    parchi = parchi[parchi.area_parco>10000].reset_index(drop = True)

    #drop everything without PARCO in the name
    parchi = parchi[['PARCO' in x for x in parchi['parco']]].reset_index(drop = True)
    
    #drop duplicates
    parchi = parchi.loc[parchi['parco'].drop_duplicates().index].reset_index(drop = True)

    #calculate the distance of each home to nearest parco and append
    print('Beginning park\n')

    parchi['distanza_parco'] = dist_df(data, parchi)
    parchi = parchi[['distanza_parco', 'area_parco']].reset_index(drop = True)

    data = pd.concat([data, parchi], axis = 1)

    #clean scuole
    scuole_infanzia = json_normalize(scuole_infanzia_json['features'])
    scuole_primarie = json_normalize(scuole_primarie_json['features'])

    #clean scuole secondarie
    scuole_secondarie = json_normalize(scuole_secondarie_json['features'])
    scuole_secondarie_2 = json_normalize(scuole_secondarie_2_json['features'])
    
    #keep useful column
    scuole_infanzia = scuole_infanzia[['properties.GRADO', 'properties.TIPO', 'geometry.coordinates']]
    scuole_primarie = scuole_primarie[['properties.GRADO', 'properties.TIPO', 'geometry.coordinates']]
    scuole_secondarie = scuole_secondarie[['properties.GRADO', 'properties.TIPO', 'geometry.coordinates']]
    scuole_secondarie_2 = scuole_secondarie_2[['properties.POSIZGIURI', 'geometry.coordinates']]

    #rename columns
    scuole_secondarie_2 = scuole_secondarie_2.rename(columns = {'properties.POSIZGIURI': 'properties.TIPO',
                                                            'geometry.coordinates': 'geometry.coordinates'})

    #add Grado
    scuole_secondarie_2['properties.GRADO'] = 'Scuola secondaria di secondo grado'
    scuole_secondarie_2 = scuole_secondarie_2[['properties.GRADO', 'properties.TIPO', 'geometry.coordinates']]

    #concatenate all
    scuole = pd.concat([scuole_infanzia, scuole_primarie, scuole_secondarie, scuole_secondarie_2], axis = 0, ignore_index = True)

    #keep only statal school
    scuole = scuole[['STATALE' ==x for x in scuole['properties.TIPO']]].reset_index(drop = True)
    scuole = scuole.drop('properties.TIPO', axis = 1)

    #rename columns
    scuole = scuole.rename(columns = {'properties.GRADO': 'grado_scuola', 'geometry.coordinates': 'coordinates'})

    #calculate long, lat
    scuole['LONGITUDE'] = scuole.coordinates.apply(lambda x: x[0])
    scuole['LATITUDE'] = scuole.coordinates.apply(lambda x: x[1])

    #unique grade of school list
    school_list = scuole['grado_scuola'].unique()

    print('Beginning school\n')

    #calculate nearest distance to selected grade school
    for school in school_list:
        print(f'\nSchool: {school}\n')
        data[school+'_distanza'] = dist_df(data, negozi, filter_var = 'grado_scuola', label_filter = school)

    print('Beginning criminality\n')

    #calculate number of reati 1km of distance
    for reato in ['violenza-sessuale', 'rapine', 'furti', 'omicidi', 'droga', 'campi-nomadi']:
        print(f'\nReato: {reato}\n')
        data[reato + '_distanza_1k'] = radius_json(data, criminality, radius = 1, filter_label = reato)

    print('Beginning criminality radius\n')

    #calculate number of each reato with 1km of distance
    data['reati_all_distanza_1k'] = radius_json(data, criminality_all, radius = 1)

    return(data)