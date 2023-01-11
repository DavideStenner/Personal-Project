# %%
import json
import pandas as pd
import numpy as np
import cvxpy as cp

with open('mapping_artists.json') as mapping_file:
    mapping = json.load(mapping_file)

artists_mapping = mapping['artists_mapping']
quotazioni = mapping['quotazioni']

with open('config.json') as config_file:
    config = json.load(config_file)


def get_artists_composition(
    df: pd.DataFrame, 
    num_baudi: int = config['num_baudi'], 
    num_selection: int = config['num_selection'], solver=cp.ECOS_BB
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

    return results, score

#%%
with open('data/TicketOne/results.json') as config_file:
    results = json.load(config_file)

df = pd.DataFrame(
    [[x, y] for x, y in results['frequency'].items()],
    columns = ['artista', 'frequenza']
)
df = df.merge(
    pd.DataFrame(
        [[x, y] for x, y in results['weight'].items()],
        columns = ['artista', 'weight']
    ),
)

df['artista'] = df['artista'].map(artists_mapping)
df['quotazione'] = df['artista'].map(quotazioni)
# df['norm_freq'] = df.groupby('quotazione')['frequenza'].transform('sum')

df['score'] = (
    #rank medio
    (df['weight']/(df['frequenza'] * 5))# + \
    # (df['frequenza']/(df['norm_freq']))
)
#%%
composition, score = get_artists_composition(df)
print(score)
composition.sort_values('score', ascending=False)