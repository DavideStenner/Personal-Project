import urllib.request
import time
import json
import pandas as pd
import os
from pandas.io.json import json_normalize

limit_max = 999999999

url_block ={
    (r'economia_media_grande_distribuzione_coord.csv', f'https://dati.comune.milano.it/api/3/action/datastore_search?resource_id=cbce9044-f7e6-45c9-a53d-e753adbcd63c&limit={limit_max}'),
    (r'ds634_civici_coordinategeografiche.csv', f'https://dati.comune.milano.it/api/3/action/datastore_search?resource_id=533b4e63-3d78-4bb5-aeb4-6c5f648f7f21&limit={limit_max}'),
    (r'nilzone.geosjon', r'https://dati.comune.milano.it/dataset/e8e765fc-d882-40b8-95d8-16ff3d39eb7c/resource/9c4e0776-56fc-4f3d-8a90-f4992a3be426/download/ds964_nil_wm.geojson'),
    (r'parchi.geojson', r'https://dati.comune.milano.it/dataset/8920bbea-fe1a-4061-8c6e-0746d95316c1/resource/7cba11e7-c236-49a7-aef4-4db6329eec5d/download/parchi.geojson'),
    (r'scuoleinfanzia.geojson', r'https://dati.comune.milano.it/dataset/cd640520-73cc-47d6-ab8b-f8255ea3b596/resource/06aad6bd-30f1-4517-ba30-3839826ed028/download/scuoleinfanzia_2012_2013.geojson'),
    (r'scuoleprimarie.geojson', r'https://dati.comune.milano.it/dataset/3393bb79-d737-47f3-8c2c-2e7ff69b58c2/resource/fa93e2be-a914-4936-8b86-e49ee6d323da/download/scuoleprimarie_2012_2013.geojson'),
    (r'scuolesecondarie1grado.geojson', r'https://dati.comune.milano.it/dataset/f66566e5-9095-4059-8a9b-18e014ef04ce/resource/e037a4b3-1f99-4fce-b511-df3d33e5766f/download/scuolesecondarie1grado_2012_2013.geojson'),
    (r'scuolesecondarie2grado.geojson', r'https://dati.comune.milano.it/dataset/5b4aee8b-8e80-447b-ac91-623544e1c654/resource/a1fa4ea2-31bb-4725-9ca3-89ef0f03a8c8/download/scuolesecondariesecondogrado.geojson'),
    (r'tpl_metrofermate.geojson', r'https://dati.comune.milano.it/dataset/b7344a8f-0ef5-424b-a902-f7f06e32dd67/resource/dd6a770a-b321-44f0-b58c-9725d84409bb/download/tpl_metrofermate.geojson'),
    (r'tpl_fermate.geojson', r'https://dati.comune.milano.it/dataset/ac494f5d-acd3-4fd3-8cfc-ed24f5c3d923/resource/7d21bd77-3ad1-4235-9a40-8a8cdfeb65a0/download/tpl_fermate.geojson')
}
path_save = r'C:\Users\stenner\Documents\GitHub\Personal-Project\Home-Scraper\data'

def open_loader(path_save):
    for name, url in url_block:
        time.sleep(2)

        extension = name.split('.')[-1].lower()
        save_path = os.path.join(path_save, name)

        with urllib.request.urlopen(url) as file:
            obj = json.loads(file.read())

        if extension == 'csv':
            
            obj = obj['result']['records']
            obj = pd.DataFrame.from_records(obj)

            obj.to_csv(save_path, index = False)

        else:
            with open(save_path, 'w') as outfile:
                json.dump(obj, outfile)

open_loader(path_save)