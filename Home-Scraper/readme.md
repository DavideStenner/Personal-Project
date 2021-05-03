The main task of this analysis is to download information of different homes in Milan, create a model which is able to forecast the price and find the most under priced home.

This project scrapes data from 4 different sources:

- Immobiliare (Price information)
- Milano Today (News about criminality)
- Data open Milano (location of parks, schools, supermarket, ...)
- Aler 

Step to extract the data:
- src/home_downloader.py
- src/opendata_loader.py
- src/other_downloader.py
- src/etl_pipeline.py
- src/preprocess_pipeline.py

The notebook train the model.
