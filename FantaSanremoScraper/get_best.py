#%%
import json
import pandas as pd
import numpy as np

with open('data/results_4999.json') as config_file:
    results = json.load(config_file)

remapper_dict = {
    'artists/elodie.png': 'Elodie',
    'artists/anna-oxa.png': 'Anna Oxa',
    'artists/ariete.png': 'Ariete',
    'artists/collazio.png': 'Colla Zio',
    'artists/gianluca-grignani.png': 'Gianluca Grignani',
    'artists/lazza.png': 'Lazza',
    'artists/rosa-chemical.png': 'Rosa Chemical',
    'artists/madame.png': 'Madame',
    'artists/mr.png': 'Mr. Rain',
    'artists/moda.png': 'Modà',
    'artists/ultimo.png': 'Ultimo',
    'artists/will.png': 'Will',
    'artists/levante.png': 'Levante',
    'artists/marco-mengoni.png': 'Marco Mengoni',
    'artists/leo-gassmann.png': 'Leo Gassmann',
    'artists/shari.png': 'Shari',
    'artists/i-cugini-di-campagna.png': 'I Cugini di Campagna',
    'artists/sethu.png': 'Sethu',
    'artists/tananai.png': 'Tananai',
    'artists/articolo-31.png': 'Articolo 31',
    'artists/lda.png': 'LDA',
    'artists/olly.png': 'OLLY',
    'artists/gianmaria.png': 'gIANMARIA',
    'artists/paola-e-chiara.png': 'Paola & Chiara',
    'artists/colapesce-dimartino.png': 'Colapesce Dimartino',
    'artists/giorgia.png': 'Giorgia',
    'artists/mara-sattei.png': 'Mara Sattei',
    'artists/coma-cose.png': 'Coma_Cose'
}

quotazioni = {
    'Anna Oxa': 18,
    'Ariete': 21,
    'Articolo 31': 21,
    'Colapesce Dimartino': 20,
    'Colla Zio': 16,
    'Coma_Cose': 20,
    'Elodie': 24,
    'Gianluca Grignani': 20,
    'gIANMARIA': 17,
    'Giorgia': 25,
    'I Cugini di Campagna': 21,
    'Lazza': 22,
    'LDA': 21,
    'Leo Gassmann': 18,
    'Levante': 20,
    'Madame': 22,
    'Mara Sattei': 21,
    'Marco Mengoni': 26,
    'Modà': 18,
    'Mr. Rain': 19,
    'OLLY': 16,
    'Paola & Chiara': 22,
    'Rosa Chemical': 19,
    'Sethu': 16,
    'Shari': 17,
    'Tananai': 22,
    'Ultimo': 27,
    'Will': 16
}
# %%
df = pd.DataFrame([[x, y] for x, y in results.items()])
df.columns = ['artista', 'frequenza']
df['artista'] = df['artista'].map(remapper_dict)
df['quotazione'] = df['artista'].map(quotazioni)
df['score'] = df['frequenza']/df['quotazione']
# %%
df.sort_values('score', ascending=False)
