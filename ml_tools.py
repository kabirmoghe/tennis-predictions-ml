
import math
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,accuracy_score,confusion_matrix
import joblib
from datetime import datetime, timedelta
from rapidfuzz import fuzz, process

#### Match Retrieval ###

# Gets a player's list of matches
def get_player_matches(all_matches,player):
    matches_won = all_matches[all_matches['winner_name'] == player]    
    matches_lost = all_matches[all_matches['loser_name'] == player]

    return pd.concat([matches_won, matches_lost]).sort_values(by = "tourney_date").reset_index(drop=True)

# Gets a player's list of matches after a certain date (the last 'span' days)
def get_recent_matches(matches,span):
    cutoff = int((datetime.now()-timedelta(span)).strftime("%Y%m%d"))
    
    return matches[matches['tourney_date']>cutoff]

### Stats for Individual Matches ###

# Get's a player's ranking from last match played 
def get_player_ranking(last_match, player):

    if last_match["winner_name"] == player:
        return last_match["winner_rank"]
    else:
        return last_match["loser_rank"]

# Gets the split of sets, games, and breakes given a scoreline
def calc_set_game_tallies(scoreline):

    # By default, assumes neither set nor game info exists
    win_tallies = {'set':float("NaN"), 'game':float("NaN"), 'tb':{'w_tbs_won':0, 'l_tbs_won':0}}
    
    # Checks if the scoreline is Null
    if (type(scoreline)==float):
        return win_tallies
    # Valid score
    else:
        # Dicts to hold the total # of sets and games and the split between the loser and winner
        player_sets = {'tot_sets': 0, 'w_sets': 0, 'l_sets': 0}
        player_games = {'tot_games': 0, 'w_games': 0, 'l_games':0}
        sets = scoreline.split()

        # Iterates through sets and increments above values
        for set in sets:
            game_split = set.split('[')[0].split('(')[0].split('-')

            # If the set doesn't end in a default
            if len(game_split) == 2:
                
                # Games
                player_games['w_games'] += int(game_split[0])
                player_games['l_games'] += int(game_split[1])

                # Sets
                player_sets['tot_sets'] += 1
                if int(game_split[0]) > int(game_split[1]):
                    player_sets['w_sets'] += 1
                else:
                    player_sets['l_sets'] += 1

                # Tiebreakers
                if int(game_split[0]) == 7 and int(game_split[1]) == 6:
                    win_tallies["tb"]['w_tbs_won'] += 1
                elif int(game_split[1]) == 7 and int(game_split[0]) == 6:
                    win_tallies["tb"]['l_tbs_won'] += 1
                
        player_games['tot_games'] = player_games['w_games'] + player_games['l_games']

        # Final pcts        
        if player_games['tot_games'] != 0:
            win_tallies['set'] = player_sets
            win_tallies['game'] = player_games

        return win_tallies

# Calculates serving metrics for a match, given that it's a win or loss for the player at hand
def calc_serve_metrics(match_info, w_or_l):
    
    serve_metrics = {'1st_in_pct':float("NaN"),
                    '2nd_in_pct':float("NaN"),
                    '1st_won_pct':float("NaN"),
                    '2nd_won_pct':float("NaN")}

    # Calculating each of the service-related metrics, ensuring that each of the used features aren't null
    if w_or_l == 'w':
        if not np.isnan(match_info['w_svpt']) and match_info['w_svpt'] > 0:
            if not np.isnan(match_info['w_1stIn']):
                # Number of second serves hit
                num_2nd = match_info['w_svpt']-match_info['w_1stIn']
                serve_metrics['1st_in_pct'] = match_info['w_1stIn']/match_info['w_svpt']

                if not np.isnan(match_info['w_df']) and num_2nd > 0:
                    serve_metrics['2nd_in_pct'] = (num_2nd-match_info['w_df'])/num_2nd

                if not np.isnan(match_info['w_1stWon']) and match_info['w_1stIn'] > 0:
                    serve_metrics['1st_won_pct'] = match_info['w_1stWon']/match_info['w_1stIn']

                if not np.isnan(match_info['w_2ndWon']) and num_2nd > 0:
                    serve_metrics['2nd_won_pct'] = match_info['w_2ndWon']/num_2nd

    else:
        if not np.isnan(match_info['l_svpt']) and match_info['l_svpt'] > 0:
            if not np.isnan(match_info['l_1stIn']):
                # Number of second serves hit
                num_2nd = match_info['l_svpt']-match_info['l_1stIn']
                serve_metrics['1st_in_pct'] = match_info['l_1stIn']/match_info['l_svpt']

                if not np.isnan(match_info['l_df']) and num_2nd > 0:
                    serve_metrics['2nd_in_pct'] = (num_2nd-match_info['l_df'])/num_2nd

                if not np.isnan(match_info['l_1stWon']) and match_info['l_1stIn'] > 0:
                    serve_metrics['1st_won_pct'] = match_info['l_1stWon']/match_info['l_1stIn']

                if not np.isnan(match_info['l_2ndWon']) and num_2nd > 0:
                    serve_metrics['2nd_won_pct'] = match_info['l_2ndWon']/num_2nd

    return serve_metrics

# Calculates breaking metrics for a match, given that it's a win or loss for the player at hand
def calc_breaking_metrics(match_info, w_or_l):

    # Break: % of opponent's service games broken; Break Saved: % of break points saved; Break Conv.: % of break points converted
    breaking_metrics = {'break':float("NaN"),
                        'break_svd':float("NaN"),
                        'break_conv':float("NaN")}

    # Calculating each of the breaking-related metrics, ensuring that each of the used features aren't null    
    if w_or_l == 'w':
        if not np.isnan(match_info['l_bpFaced']) and not np.isnan(match_info['l_bpSaved']) and not np.isnan(match_info['l_SvGms']) and match_info['l_SvGms'] > 0 and match_info['l_bpFaced'] > 0:
            breaking_metrics['break'] = (match_info['l_bpFaced']-match_info['l_bpSaved'])/match_info['l_SvGms']
            breaking_metrics['break_conv'] = (match_info['l_bpFaced']-match_info['l_bpSaved'])/match_info['l_bpFaced']
        if not np.isnan(match_info['w_bpSaved']) and not np.isnan(match_info['w_bpFaced']) and match_info['w_bpFaced'] > 0:
            breaking_metrics['break_svd'] = match_info['w_bpSaved']/match_info['w_bpFaced']
    else:
        if not np.isnan(match_info['w_bpFaced']) and not np.isnan(match_info['w_bpSaved']) and not np.isnan(match_info['w_SvGms']) and match_info['w_SvGms'] > 0 and match_info['w_bpFaced'] > 0:
            breaking_metrics['break'] = (match_info['w_bpFaced']-match_info['w_bpSaved'])/match_info['w_SvGms']
            breaking_metrics['break_conv'] = (match_info['w_bpFaced']-match_info['w_bpSaved'])/match_info['w_bpFaced']
        if not np.isnan(match_info['l_bpSaved']) and not np.isnan(match_info['l_bpFaced']) and match_info['l_bpFaced'] > 0:
            breaking_metrics['break_svd'] = match_info['l_bpSaved']/match_info['l_bpFaced']

    return breaking_metrics

# Calculates return metrics for a match, given that it's a win or loss for the player at hand
def calc_return_metrics(match_info, w_or_l):

    return_metrics = {'pts_won_v_1st':float("NaN"),
                      'pts_won_v_2nd':float("NaN")}

    # Calculating both of the return-related metrics, ensuring that each of the used features aren't null
    if w_or_l == 'w':
        if not np.isnan(match_info['l_1stIn']) and not np.isnan(match_info['l_1stWon']) and match_info['l_1stIn'] > 0:
            return_metrics['pts_won_v_1st'] = (match_info['l_1stIn']-match_info['l_1stWon'])/match_info['l_1stIn']
        if not np.isnan(match_info['l_svpt']) and not np.isnan(match_info['l_1stIn']) and not np.isnan(match_info['l_2ndWon']) and (match_info['l_svpt']-match_info['l_1stIn']) > 0:
            return_metrics['pts_won_v_2nd'] = (match_info['l_svpt']-match_info['l_1stIn']-match_info['l_2ndWon'])/(match_info['l_svpt']-match_info['l_1stIn'])
    else:
        if not np.isnan(match_info['w_1stIn']) and not np.isnan(match_info['w_1stWon']) and match_info['w_1stIn'] > 0:
            return_metrics['pts_won_v_1st'] = (match_info['w_1stIn']-match_info['w_1stWon'])/match_info['w_1stIn']
        if not np.isnan(match_info['w_svpt']) and not np.isnan(match_info['w_1stIn']) and not np.isnan(match_info['w_2ndWon']) and (match_info['w_svpt']-match_info['w_1stIn']) > 0:
            return_metrics['pts_won_v_2nd'] = (match_info['w_svpt']-match_info['w_1stIn']-match_info['w_2ndWon'])/(match_info['w_svpt']-match_info['w_1stIn'])

    return return_metrics


# Calculates tournament level performance
def calc_level_performance(player_matches,lvls,player):
    win_pcts_by_lvl = {}
    
    for lvl in lvls:
        matches_at_lvl = player_matches[player_matches["tourney_level"] == lvl]
        num_matches_at_lvl = len(matches_at_lvl)
    
        if num_matches_at_lvl > 0:
            win_pcts_by_lvl['win_pct '+lvl] = len(matches_at_lvl[matches_at_lvl['winner_name'] == player])/num_matches_at_lvl
        else:
            win_pcts_by_lvl['win_pct '+lvl] = float("NaN")

    return win_pcts_by_lvl

# Calculate the activity of the player over a certain period of time
def calculate_player_activity(player_matches,span):
    return len(player_matches)/span

# Calculates all stats for individual player and a certain match
def calc_player_stats(match_info, player):

    # List of fractional features for ensuring that metrics aren't > 1 for faulty data
    pct_stats = ['set', 'game', 'hold', '1st_in_pct', '2nd_in_pct', '1st_won_pct', '2nd_won_pct', 'break',
             'break_svd', 'break_conv', 'pts_won_v_1st', 'pts_won_v_2nd']
    

    # If the player won or lost
    w_or_l = "w"
    if match_info["winner_name"] != player:
        w_or_l = "l"        

    # Gets the split in the score for the match (for sets, games, and breakers)
    set_game_tallies = calc_set_game_tallies(match_info['score'])
    player_sets, player_games, tb_info = set_game_tallies['set'], set_game_tallies['game'], set_game_tallies['tb']

    # Calculate stats
    player_stats = {'set':float("NaN"),
                    'game':float("NaN"),
                    'hold':float("NaN")}

    # Gets opponent ranking points
    opponent_pts = match_info["loser_rank_points"]
    
    if w_or_l == "w":
        # Proportions of sets and games 
        if type(player_sets) == dict:
            player_stats['set'] = player_sets['w_sets']/player_sets['tot_sets']
        if type(player_games) == dict:
            player_stats['game'] = player_games['w_games']/player_games['tot_games']
            
            # Proportion of service games held
            if not np.isnan(match_info['l_bpFaced']) and not np.isnan(match_info['l_bpSaved']) and not np.isnan(match_info['w_SvGms']) and match_info['w_SvGms'] > 0:
                player_stats['hold'] = (player_games['w_games']-tb_info['w_tbs_won']-(match_info['l_bpFaced']-match_info['l_bpSaved']))/match_info['w_SvGms']

    else:
        # Updates ranking points
        opponent_pts = match_info["winner_rank_points"]
        
        # Proportions of sets and games won
        if type(player_sets) == dict:
            player_stats['set'] = player_sets['l_sets']/player_sets['tot_sets']
        if type(player_games) == dict:
            player_stats['game'] = player_games['l_games']/player_games['tot_games']
        
            # Proportion of service games held
            if not np.isnan(match_info['w_bpFaced']) and not np.isnan(match_info['w_bpSaved']) and not np.isnan(match_info['l_SvGms']) and match_info['l_SvGms'] > 0:
                player_stats['hold'] = (player_games['l_games']-tb_info['l_tbs_won']-(match_info['w_bpFaced']-match_info['w_bpSaved']))/match_info['l_SvGms']

    aggr_stats = {**player_stats, **calc_serve_metrics(match_info, w_or_l), **calc_breaking_metrics(match_info, w_or_l), **calc_return_metrics(match_info, w_or_l), "Oppononet Pts":opponent_pts}

    for stat in pct_stats:
        if aggr_stats[stat] > 1:
            aggr_stats[stat] = float("NaN")

    return aggr_stats

### Mean Metrics ###
# For metrics across a set of matches (either entire set of matches across the year, matches across a surface, etc.)
def produce_mean_metrics(match_list,player,prefix=''): # Prefix is if it is for a surface (i.e. 'Grass ')

    lvls = match_list['tourney_level'].unique() # Number of unique levels

    # Hardcoding the number of days to include for overall and recent data
    overall_span = 730 
    recency_span = 90

    # Calculate stat columns
    calculated_stats = ['Set W%','Game W%','Hold %','1st Serve In %','2nd Serve In %','1st Serve Won %','2nd Serve Won %',
                        'Breaking %','Break Pts Saved %','Break Pts Conv. %','Pts Won v 1st Serve %','Pts Won v 2nd Serve %','Opponent Quality']
    
    match_list[calculated_stats] = match_list.apply(lambda row: pd.Series(calc_player_stats(row, player)), axis=1)

    # Career stats
    mean_career_stats = {}
    
    # Recent stats
    recent_matches = get_recent_matches(match_list,recency_span)
    mean_recent_stats = {}

    # Gets mean info 
    for category in calculated_stats:
        mean_career_stats[category] = match_list[category].mean()
        mean_recent_stats["3mo " + category] = recent_matches[category].mean()

    # Adds activity
    mean_career_stats.update({'Activity': calculate_player_activity(match_list,overall_span)})
    mean_recent_stats.update({'3mo Activity': calculate_player_activity(recent_matches,recency_span)})

    # Calculate performance across various tournament levels
    player_lvl_performance = calc_level_performance(match_list,lvls,player)
    recent_player_lvl_performance = calc_level_performance(recent_matches,lvls,player)
    recent_player_lvl_performance = {'3mo ' + lvl: pct for lvl, pct in recent_player_lvl_performance.items()}

    # Aggregates stats together and adds the prefix 
    aggr_stats = {**mean_career_stats,**mean_recent_stats}
    aggr_stats_prefix = {prefix + stat: value for stat, value in aggr_stats.items()}

    return aggr_stats_prefix

# Gets a dictionary of stats for each of the players found across match data
def get_cumulative_stats(all_matches):

    # Hardcoding the number of days to include for recent data
    recency_span = 90

    # List of unique players and dictionary to store each player's stats
    all_players = list(set(all_matches['winner_name']).union(all_matches['loser_name']))
    all_players.sort()
    player_stats = {}

    # Unique tournament levels
    lvls = all_matches['tourney_level'].unique()

    # List of surfaces (hard, grass, clay, etc.)
    surfaces = list(all_matches['surface'].unique()) 
    
    # Iterate through each player
    for player in all_players:
        
        # Get all of their matches
        player_matches = get_player_matches(all_matches,player)

        # Store the stats in the dictionary
        produced_metrics = {"Last Rank":get_player_ranking(player_matches.iloc[-1], player)}
        produced_metrics.update(produce_mean_metrics(player_matches, player))

        # Iterates through surfaces and gets their surface-specific match stats
        for s in surfaces:
            surface_matches = (player_matches[player_matches['surface'] == s]).copy(deep=True)
            if len(surface_matches) > 0:
                produced_metrics.update(produce_mean_metrics(surface_matches, player, s+' ')) # +' ' necessary for space

        player_stats[player] = produced_metrics
    
    return player_stats

def get_matches_surface_stats(r_match_data,stat_data):

    # List of surface-specific columns
    surface_columns = {'Hard':[],'Clay':[],'Grass':[],'None':[]}
    
    for col in stat_data.columns[30:]:
        for surface in surface_columns.keys():
            if surface in col:
                surface_columns[surface].append(col)
    
    # Creates matches with surface stats based on the surface of each match
    surface_stats_data = pd.DataFrame()

    for surface in surface_columns:
            
        # Matches & stats for specific surface
        surface_specific_matches = r_match_data[r_match_data['surface'] == surface]
        surface_specific_stats = stat_data[['Player']+surface_columns[surface]]
    
        # -- Player A --
        
        # Renames surface columns (i.e. Hard Set W%) to generic name to allow for vertical concatenation for cross-surface stats
        surface_rename_dict = {}
        
        for col in surface_columns[surface]:
            surface_rename_dict[col] = 'player_A Surface' + col.split(surface)[1]
        
        # Adds stats by surface to each match
        surface_matches_with_stats = pd.merge(surface_specific_matches, surface_specific_stats, left_on='player_A', right_on='Player', how='left')
        surface_matches_with_stats.rename(columns=surface_rename_dict,inplace=True)
        surface_matches_with_stats.drop(['Player'],axis=1,inplace=True)
        
        # -- Player B -- 
        
        # Renames surface columns (i.e. Hard Set W%) to generic name to allow for vertical concatenation for cross-surface stats
        for col in surface_columns[surface]:
            surface_rename_dict[col] = 'player_B Surface' + col.split(surface)[1]
    
        # Adds stats by surface to each match 
        surface_matches_with_stats = pd.merge(surface_matches_with_stats, surface_specific_stats, left_on='player_B', right_on='Player', how='left')
        surface_matches_with_stats.rename(columns=surface_rename_dict,inplace=True)
        surface_matches_with_stats.drop(['Player'],axis=1,inplace=True)
       
        # Concatenates matches across all surfaces
        surface_stats_data = pd.concat([surface_stats_data,surface_matches_with_stats],axis=0)
    
    surface_stats_data.reset_index(drop=True,inplace=True)

    return surface_stats_data

### Final Data Production ###

# Randomizes the matchups between players (i.e. randomizes the order for players A and B) across a set of matches
def randomize_matches(match_data):

    # Copies original matches
    matches_copy = match_data.copy(deep=True)

    # Randomizes match order
    matches_copy['player_A'] = np.where(np.random.rand(len(matches_copy)) > 0.5, matches_copy['winner_name'], matches_copy['loser_name'])
    matches_copy['player_B'] = np.where(matches_copy['player_A'] == matches_copy['winner_name'], matches_copy['loser_name'], matches_copy['winner_name'])
    matches_copy['outcome'] = np.where(matches_copy['player_A'] == matches_copy['winner_name'], 0, 1)

    return matches_copy[['player_A', 'player_B', 'surface', 'tourney_level', 'outcome']]

# Adds stats to match data
def get_matches_with_stats(match_data):

    # Gets final player stats & randomizes players A and B
    player_stats = get_cumulative_stats(match_data)
    stat_data = pd.DataFrame(player_stats).transpose().reset_index().rename(columns={"index":"Player"})   
    r_match_data = randomize_matches(match_data)
    
    # Overall stat columns and data
    overall_calculated_stats = stat_data.columns[1:30]
    overall_stat_data = stat_data[stat_data.columns[:30]]

    # Gets surface-specific data for players
    r_matches_surface_data = get_matches_surface_stats(r_match_data,stat_data)
            
    # Merge player A's stats
    merged_data = pd.merge(r_matches_surface_data, overall_stat_data, left_on='player_A', right_on='Player', how='left') # overall
    
    # Rename columns for player A
    rename_dict = {}

    for col in overall_calculated_stats:
        rename_dict[col] = f"player_A {col}"
        
    merged_data.rename(columns=rename_dict, inplace=True)
    merged_data.drop(columns=['Player'], inplace=True)
    
    # Merge player B's stats
    merged_data = pd.merge(merged_data, overall_stat_data, left_on='player_B', right_on='Player', how='left')
    
    # Rename columns for player B
    for col in overall_calculated_stats:
        rename_dict[col] = f"player_B {col}"
        
    merged_data.rename(columns=rename_dict, inplace=True)
    merged_data.drop(columns=['Player','surface','tourney_level'], inplace=True)

    return merged_data

### ML Training ###

# Stores trained models
class MatchPredict:
    def __init__(self,X_train,y_train):
        self.X_train, self.y_train = X_train, y_train
        self.model_types = {'rf': RandomForestClassifier(),
                       'lr': LogisticRegression(max_iter=3000)}
        self.trained_models = {}

    def add_model_type(self,model_type,model_declaration):
        self.model_types[model_type] = model_declaration

    def train(self,model_type,model_ref):
        if model_type not in self.model_types.keys():
            raise Exception("Not a currently supported model type.")
        else:
            model = self.model_types[model_type]
            model.fit(self.X_train, self.y_train)
            self.trained_models[model_ref] = {'model':model}

    def predict(self,model_ref,X_test,y_test):
        if model_ref not in self.trained_models.keys():
            raise Exception("Not an existing trained model.")
        else:
            trained_model = self.trained_models[model_ref]['model']
            predictions = trained_model.predict(X_test)

            self.trained_models[model_ref]['X_test'] = X_test
            self.trained_models[model_ref]['y_test'] = y_test
            self.trained_models[model_ref]['y_pred'] = predictions

            return predictions

    def evaluate(self,model_ref):
        if model_ref not in self.trained_models.keys():
            raise Exception("Not an existing trained model.")
        elif 'X_test' not in self.trained_models[model_ref].keys():
            raise Exception("Predictions for this model have not been made.")
        else:
            y_test,y_pred = self.trained_models[model_ref]['y_test'],self.trained_models[model_ref]['y_pred']
            metrics = {'confusion_matrix':confusion_matrix(y_test,y_pred),
                       'f1_score':f1_score(y_test,y_pred),
                       'accuracy_score':accuracy_score(y_test,y_pred)}
            
            return metrics

    def export(self,model_ref):
        joblib.dump(self.trained_models[model_ref]['model'], f'{model_ref}.pkl')        

# Predicts an individual match outcome
def predict_winner(player_A,player_B,surface,stat_data,model):
    
    if player_A not in stat_data['Player'].values:
        tentative_player_A, score, _ = process.extractOne(player_A, stat_data['Player'].values, scorer=fuzz.partial_ratio)
        if score > 80:
            print(f"Fuzzy match found: {tentative_player_A}")
            player_A = tentative_player_A
        else:
            print(f"RETURNING {player_B}")
            return player_B
    if player_B not in stat_data['Player'].values and player_B != '[Bye]':
        tentative_player_B, score, _ = process.extractOne(player_B, stat_data['Player'].values, scorer=fuzz.partial_ratio)
        if score > 80:
            print(f"Fuzzy match found: {tentative_player_B}")
            player_B = tentative_player_B
        else:
            print(f"RETURNING {player_A}")
            return player_A
    if player_A == player_B:
        raise ValueError("Players cannot be the same.")

    # ** DATA **
    
    # player_A_data = stat_data[stat_data['Player']==player_A]
    overall_calculated_stats = stat_data.columns[1:30]
    overall_stat_data = stat_data[stat_data.columns[:30]]

    # Getting surface stats
    matchup = pd.DataFrame([[player_A,player_B,surface]],columns=["player_A","player_B","surface"])
    matchup_surface_data = get_matches_surface_stats(matchup,stat_data)

    # Getting overall stats

    # -- Player A --
    matchup_overall = pd.merge(matchup_surface_data, overall_stat_data, left_on='player_A', right_on='Player', how='left')
    
    # Rename columns
    rename_dict = {}

    for col in overall_calculated_stats:
        rename_dict[col] = f"player_A {col}"

    matchup_overall.rename(columns=rename_dict, inplace=True)
    matchup_overall.drop(columns=['Player'], inplace=True)

    # -- Player A --
    matchup_overall = pd.merge(matchup_overall, overall_stat_data, left_on='player_B', right_on='Player', how='left')
    
    # Rename columns
    for col in overall_calculated_stats:
        rename_dict[col] = f"player_B {col}"
        
    matchup_overall.rename(columns=rename_dict, inplace=True)

    matchup_overall.drop(columns=['player_A','player_B','Player','surface'], inplace=True)

    # Removing null values
    matchup_overall = matchup_overall.fillna({'player_A Last Rank':1000,'player_B Last Rank':1000}) # Need to penalize with low (i.e. high #) rank
    matchup_overall = matchup_overall.fillna(0) # The rest can be 0, as 0 is bad for all other stats
    
    # ** PREDICTION **
    prediction = model.predict(matchup_overall)[0]

    if prediction == 0:
        return player_A
    else:
        return player_B

# Predicting an entire bracket with a given model
def bracket_predict(players,surface,num_players,stat_data,model,output_bracket,mode='store'):
    if len(players) == 1:
        winner=players[0] # Final winner

        # If storing, return the final bracket
        if mode=='store':
            return output_bracket
    else:
        winners = []
        for i in range(int(len(players)/2)):
            player_A = players[2*i]
            player_B = players[2*i + 1]

            # Want to output bracket results, not store
            if mode!='store':
                rd=int(math.log2(num_players)) - int(math.log2(len(players)))+1
                indent = rd*"\t"
                
                print("{}-Rd.{}-".format(indent,rd))
                print("{}{} v. {}:".format(indent,player_A,player_B))
            
            winner = predict_winner(player_A,player_B,surface,stat_data,model)
            winners.append(winner)

            if mode!='store': # Want to output the winner
                print("{}W: {}".format(indent,winner))
        #  Want to store winners from each round
        if mode=='store':
            output_bracket.append(winners)
        
        bracket_predict(winners,surface,num_players,stat_data,model,output_bracket,mode)

## Retrieving existing brackets ###

# Map of player aliases for data access
def get_player_alias(player):

    player_aliases = {
        "Albert Ramos Vinolas":"Albert Ramos",
        "Jeffrey John Wolf": "J J Wolf",
        "J.J. Wolf": "J J Wolf",
        "Marcelo Tomas Barrios Vera": "Tomas Barrios Vera"
    }

    if player in player_aliases:
        return player_aliases[player]
    else:
        return player

# Gets a bracket from an ATP-posted draw
def get_atp_draw(atp_url):
    soup = BeautifulSoup(requests.get(atp_url).content, "html.parser")
    
    draw_table = soup.find('table',{'id':'scoresDrawTable'})
    rows = draw_table.find('tbody').find_all('tr')
    match_ups = []
    bracket = []
    
    # For row in the table/bracket
    for row in rows:
        row_children = row.children
        next(row_children)
    
        # r1 matchup HTML
        match_up = next(row_children)    
        player_bios = match_up.find_all('a')
    
        # Adds matchup to list
        cleaned_match_up = []
        for bio in match_up.find_all('a'):
            cleaned_name = bio.text.strip().replace('-',' ',).replace("'",'').replace('-',' ').title()
            cleaned_match_up.append(get_player_alias(cleaned_name))
    
        if len(cleaned_match_up) != 0:
            match_ups.append(cleaned_match_up)
            bracket+=cleaned_match_up
            if len(cleaned_match_up) == 1:
                bracket.append("[Bye]")

    return bracket

### EXPORTING FINAL MODEL ###
def export_model():

    # Data
    matches_2022 = pd.read_csv("https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_2022.csv")
    matches_2023 = pd.read_csv("https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_2023.csv")

    all_matches = pd.concat([matches_2022, matches_2023], axis = 0).reset_index(drop=True)

    # Getting stats
    matches_with_data = get_matches_with_stats(all_matches)
    matches_with_data = matches_with_data.fillna({'player_A Last Rank':1000,'player_B Last Rank':1000}) # Need to penalize with low (i.e. high #) rank
    matches_with_data = matches_with_data.fillna(0) # The rest can be 0, as 0 is bad for all other stats

    # Splitting data
    X = matches_with_data.drop(['player_A','player_B','outcome'], axis=1)
    y = matches_with_data['outcome'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_test.to_csv("X_test.csv")

    # Training model
    predictor = MatchPredict(X_train,y_train)
    model_name = 'final_iter'
    predictor.train('lr',model_name)

    print(predictor.predict(model_name,X_test,y_test))

    # Exporting model
    predictor.export(model_name)

### GETTING SAMPLE BRACKET ###
def get_bracket(stat_data):
    with open("data/wimbledon_2023.html", "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    # Find all player names in the draw
    players = []
    
    # Method 1: Find by div with class "name" inside draw-stats
    for div in soup.find_all("div", class_="name"):
        link = div.find("a")
        if link:
            name = link.get_text(strip=True)
            if name and name not in players:
                players.append(name)
    
    # Method 2 (alternative): Look in the dropdown options for full names
    if len(players) == 0:
        print("No players found in Method 1")
        for option in soup.find_all("option"):
            first = option.get("data-first", "")
            last = option.get("data-last", "")
            if first and last:
                full_name = f"{first} {last}"
                if full_name not in players:
                    players.append(full_name)
    
    print(f"Found {len(players)} players")

    # Convert to fuzzy match
    fuzzy_players = []
    
    for player in players:
        fuzzy_player, score, _ = process.extractOne(player, stat_data['Player'].values, scorer=fuzz.partial_ratio)
        if score > 80:
            print(f"Fuzzy match found: {fuzzy_player}")
            fuzzy_players.append(fuzzy_player)
        else:
            fuzzy_players.append(player)
    
    return fuzzy_players

# Running script
if __name__ == "__main__":
    export_model()