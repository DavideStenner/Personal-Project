# %%
import json
import pandas as pd
import numpy as np
import cvxpy as cp

with open('mapping_artists.json') as mapping_file:
    mapping = json.load(mapping_file)

artists_mapping = mapping['artists_mapping']
quotazioni = mapping['quotazioni']


def get_artists_composition(
    df: pd.DataFrame, 
    num_baudi: int = 100, 
    num_selection: int =5, solver=cp.ECOS_BB
):
    number_suggestion = df.shape[0]

    selection = cp.Variable(number_suggestion, boolean=True)

    score = cp.Constant(df['score'].values.tolist())
    quotazione = cp.Constant(df['quotazione'].values.tolist())

    problem_knapsack = sum(cp.multiply(score, selection))

    constraint_list = [
        sum(cp.multiply(selection, quotazione)) <= num_baudi,
        sum(selection) == num_selection,
    ]
    fanta_problem = cp.Problem(
        cp.Maximize(problem_knapsack), constraint_list
    )

    score = fanta_problem.solve(solver=cp.ECOS_BB)
    selection_results = np.round(selection.value).astype(int)
    results = df.loc[
        selection_results==1
    ]
    assert results['quotazione'].sum() <= num_baudi

    return results

#%%
with open('data/Pandora/results_1572.json') as config_file:
    results = json.load(config_file)

df = pd.DataFrame([[x, y] for x, y in results['frequency'].items()])

df.columns = ['artista', 'frequenza']
df['artista'] = df['artista'].map(artists_mapping)
df['quotazione'] = df['artista'].map(quotazioni)
df['norm_freq'] = df.groupby('quotazione')['frequenza'].transform('sum')

df['score'] = df['frequenza']/df['norm_freq']

#%%
get_artists_composition(df)