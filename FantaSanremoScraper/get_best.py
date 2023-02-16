# %%
import json
import pandas as pd
import numpy as np
import cvxpy as cp
import argparse
import os

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

def get_best_team(path_save: str, league: str, print_all: bool, use_leaderboard: bool):
    save_folder = os.path.join(path_save, league)

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    with open('mapping_artists.json') as mapping_file:
        mapping = json.load(mapping_file)

    artists_mapping = mapping['artists_mapping']
    quotazioni = mapping['quotazioni']

    with open(f'data/{league}/results.json') as result_file:
        results = json.load(result_file)

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

    #rank medio
    if use_leaderboard:
        leaderboard_results = mapping['classifica_artisti']
        df['score'] = df['artista'].map(leaderboard_results)

        #inspect
        df['score_quotazione'] = df['score']/df['quotazione']

    else:
        df['score'] = (
            (df['weight']/(df['frequenza'] * 5)) 
        )

    composition, score = get_artists_composition(df)
    composition = composition.sort_values('score', ascending=False)\
        .reset_index(drop=True)
    if print_all:
        print('\n\n')
        print(df.sort_values('score', ascending=False).to_markdown())
        print('\n\n')

    print(f'\n\nScore of after optimization: {score}\n\n')
    print(composition.to_markdown())
    
    composition.to_csv(
        os.path.join(save_folder, league+'.csv'), 
        index=False
    )
        

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--path_save', type=str, default="results")
    parser.add_argument('--league', type=str, default="Campionato Mondiale")
    parser.add_argument('--get_all', action='store_true')
    parser.add_argument('--print_all', action='store_true')
    parser.add_argument('--use_leaderboard', action='store_true')

    args = parser.parse_args()
    
    if args.use_leaderboard:
        args.path_save =  'leaderboard_result'

    if args.get_all:
        with open('config.json') as config_file:
            league_dict = json.load(config_file)['league_dict']
        
        for league_name, _ in league_dict.items():
            print(f'\n\nStarting {league_name}')
            get_best_team(args.path_save, league_name, args.print_all, args.use_leaderboard)
    else:
        get_best_team(args.path_save, args.league, args.print_all, args.use_leaderboard)