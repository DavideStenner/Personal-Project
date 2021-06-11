import pandas as pd

from configuration.constant import PATH_DATA, PATH_METADATA

def import_dataset():
    try:
        data = pd.read_csv(PATH_DATA)

    except OSError:
        return None

    return data


def import_metadata(get_type_colname = False):
    try:
        metadata = pd.read_excel(PATH_METADATA)
    except OSError:
        return (None, None)

    type_col = {}

    if get_type_colname:
        

        type_col['numeric'] = metadata.loc[metadata['type'] == 'number', 'col_name'].tolist()
        type_col['string'] = metadata.loc[~metadata['type'].isin(['number', 'date', 'id']), 'col_name'].tolist()
        type_col['date'] = metadata.loc[metadata['type'] == 'date', 'col_name'].tolist()
        type_col['id'] = metadata.loc[metadata['type'] == 'id', 'col_name'].tolist()

    return (metadata, type_col)