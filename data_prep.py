import os
from util import util
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from joblib import dump, load
#data_folder = os.path.join('.', 'files/data')
#leagues = ['E0', 'E1', 'E2', 'E3', 'I1', 'SP1', 'D1', 'F1', 'N1']

def prep_data(files_path, leagues):
	data_folder = os.path.join(files_path, 'data')
	data = util.fetch_data(data_folder, 2005, 2025, leagues)

	#Clean and run simulations 
	data = util.clean_data(data)
	draw_factor = data['FTR'].value_counts(normalize=True)['D']
	ELO = util.ELO(data, init_rating=1500, draw_factor=draw_factor, k_factor=32, home_advantage=50)
	data = ELO.perform_simulations(data)


	#Train and store models
	for league in leagues:
		league_data = data[data['Div'] == league]
		league_data = util.add_form_column(league_data, 'FTHG', 'FTAG', n=5, operation='Sum', regard_opponent=False, include_current=False)
		league_data['Diff_goals_scored'] = league_data['FTHG_Sum_5'] - league_data['FTAG_Sum_5']
		league_data = util.add_form_column(league_data, 'FTHG', 'FTAG', n=5, operation='Sum', regard_opponent=True, include_current=False)
		league_data['Diff_goals_conceded'] = league_data['FTHG_Sum_5_opponent'] - league_data['FTAG_Sum_5_opponent']
		league_data['Home Goal Difference last 5'] = league_data['FTHG_Sum_5'] - league_data['FTHG_Sum_5_opponent']
		league_data['Away Goal Difference last 5'] = league_data['FTAG_Sum_5'] - league_data['FTAG_Sum_5_opponent']
		league_data['Diff_goal_diff'] = league_data['Home Goal Difference last 5'] - league_data['Away Goal Difference last 5']
		league_data = util.add_form_column(league_data, 'Home', 'Away', n=5, operation='Points', regard_opponent=False, include_current=False)
		league_data['Diff_points'] = league_data['Home_Points_5'] - league_data['Away_Points_5']
		league_data = util.add_form_column(league_data, 'Home ELO', 'Away ELO', n=5, operation='Change', regard_opponent=False, include_current=True)
		league_data['Diff_change_in_ELO'] = league_data['Home ELO_Change_5'] - league_data['Away ELO_Change_5']
		league_data = util.add_form_column(league_data, 'Home ELO', 'Away ELO', n=5, operation='Mean', regard_opponent=True, include_current=False)
		league_data['Diff_opposition_mean_ELO'] = league_data['Home ELO_Mean_5_opponent'] - league_data['Away ELO_Mean_5_opponent']
		league_data = util.add_form_column(league_data, 'HST', 'AST', n=5, operation='Sum', regard_opponent=False, include_current=False)
		league_data['Diff_shots_on_target_attempted'] = league_data['HST_Sum_5'] - league_data['AST_Sum_5']
		league_data = util.add_form_column(league_data, 'HST', 'AST', n=5, operation='Sum', regard_opponent=True, include_current=False)
		league_data['Diff_shots_on_target_allowed'] = league_data['HST_Sum_5_opponent'] - league_data['AST_Sum_5_opponent']
		league_data = util.add_form_column(league_data, 'HS', 'AS', n=5, operation='Sum', regard_opponent=False, include_current=False)
		league_data['Diff_shots_attempted'] = league_data['HS_Sum_5'] - league_data['AS_Sum_5']
		league_data = util.add_form_column(league_data, 'HS', 'AS', n=5, operation='Sum', regard_opponent=True, include_current=False)
		league_data['Diff_shots_allowed'] = league_data['HS_Sum_5_opponent'] - league_data['AS_Sum_5_opponent']
		league_data = util.add_form_column(league_data, 'HC', 'AC', n=5, operation='Sum', regard_opponent=False, include_current=False)
		league_data['Diff_corners_awarded'] = league_data['HC_Sum_5'] - league_data['AC_Sum_5']
		league_data = util.add_form_column(league_data, 'HC', 'AC', n=5, operation='Sum', regard_opponent=True, include_current=False)
		league_data['Diff_corners_conceded'] = league_data['HC_Sum_5_opponent'] - league_data['AC_Sum_5_opponent']
		league_data = util.add_form_column(league_data, 'HF', 'AF', n=5, operation='Sum', regard_opponent=False, include_current=False)
		league_data['Diff_fouls_commited'] = league_data['HF_Sum_5'] - league_data['AF_Sum_5']
		league_data = util.add_form_column(league_data, 'HF', 'AF', n=5, operation='Sum', regard_opponent=True, include_current=False)
		league_data['Diff_fouls_suffered'] = league_data['HF_Sum_5_opponent'] - league_data['AF_Sum_5_opponent']
		league_data = util.add_form_column(league_data, 'HY', 'AY', n=5, operation='Sum', regard_opponent=False, include_current=False)
		league_data['Diff_yellow_cards'] = league_data['HY_Sum_5'] - league_data['AY_Sum_5']
		league_data = util.add_form_column(league_data, 'HR', 'AR', n=5, operation='Sum', regard_opponent=False, include_current=False)
		league_data['Diff_red_cards'] = league_data['HR_Sum_5'] - league_data['AR_Sum_5']
		columns_to_remove = [
			"FTR",
			"HTHG",
			"HTAG",
			"HTR",
			"HS",
			"AS",
			"HST",
			"AST",
			"HF",
			"AF",
			"HC",
			"AC",
			"HY",
			"AY",
			"HR",
			"AR",
			"Home ELO",
			"Away ELO",
			"FTHG_Sum_5",
			"FTAG_Sum_5",
			"FTHG_Sum_5_opponent",
			"FTAG_Sum_5_opponent",
			"Home Goal Difference last 5",
			"Away Goal Difference last 5",
			"Home_Points_5",
			"Away_Points_5",
			"Home ELO_Change_5",
			"Away ELO_Change_5",
			"Home ELO_Mean_5_opponent",
			"Away ELO_Mean_5_opponent",
			"HST_Sum_5",
			"AST_Sum_5",
			"HST_Sum_5_opponent",
			"AST_Sum_5_opponent",
			"HS_Sum_5",
			"AS_Sum_5",
			"HS_Sum_5_opponent",
			"AS_Sum_5_opponent",
			"HC_Sum_5",
			"AC_Sum_5",
			"HC_Sum_5_opponent",
			"AC_Sum_5_opponent",
			"HF_Sum_5",
			"AF_Sum_5",
			"HF_Sum_5_opponent",
			"AF_Sum_5_opponent",
			"HY_Sum_5",
			"AY_Sum_5",
			"HR_Sum_5",
			"AR_Sum_5",
		]
		league_data.drop(columns=columns_to_remove, inplace=True)

		league_data["Outcome"] = league_data.apply(
			lambda row: (row["FTHG"] - row["FTAG"]),
			axis=1,
		)

		X = league_data.copy().drop(
			columns=["Outcome", "FTHG", "FTAG", "Season", "Div", "Date", "HomeTeam", "AwayTeam"],
		)
		y = league_data["Outcome"]
		rf = RandomForestRegressor(n_estimators=200, random_state=42)
		X_train, X_test, y_train, y_test = train_test_split(
			X, y, test_size=0.2, random_state=42)
		rf.fit(X_train,y_train)
		predictions = rf.predict(X_test)
		print('Ordinary stats for model for', league)
		categorized_preds = categorize_preds(predictions, 1, -1)
		categorized_goal_diff = categorize_goal_diff(y_test)

		report = classification_report(categorized_goal_diff, categorized_preds)
		print(report)
		filtered_predictions, filtered_targets = remove_uncertain(categorized_preds, categorized_goal_diff)
		print('Filtered (-1,1) stats for model for', league)
		report = classification_report(filtered_predictions, filtered_targets, output_dict=True)
		print(report)
		report_df = pd.DataFrame(report).transpose()
		report_df.to_csv(f'{files_path}/stats/{league}_report.csv', index=True)

		bins = np.arange(-3, 3.25, 0.25)
		bins = np.append(bins, [np.inf, -np.inf])
		labels = [f"{bins[i]} to {bins[i+1]}" for i in range(len(bins)-1)]
		categories = pd.cut(predictions, bins=np.sort(bins), labels=labels, include_lowest=True)
		frame = pd.DataFrame({'Values': predictions, 'Category': categories, 'Labels': categorized_goal_diff})
		distribution = frame.groupby('Category')['Labels'].value_counts().unstack(fill_value=0)
		distribution['Total'] = frame.groupby('Category')['Values'].count()
		distribution_pct = distribution.div(distribution['Total'], axis=0) * 100

		# Calculate inverse fractions for each label
		for label in [-1, 0, 1]:
			distribution[f'{label}_inv'] = distribution['Total'] / distribution[label].replace(0, np.nan)

		# Combine counts and percentages
		result = pd.concat([distribution, distribution_pct.add_suffix('_%')], axis=1)
		print('Distribution of predictions for', league)
		print(result)
		result.to_csv(f'{files_path}/stats/{league}_stats.csv', index=True, na_rep='0')

		file_path = f'{files_path}/models/{league}_model.joblib'
		dump(rf, file_path)
		print(f'Saved model for {league} to {file_path}')

	#Load current form into file	    
	df_tmp = []
	for league in leagues:
		league_data = util.fetch_data(data_folder, 2024, 2025, [league])
		teams = league_data['HomeTeam'].unique()
		for team in teams:
			#ELO, goals scored, goals conceded, goal difference, points, change in ELO, opposition mean ELO, shots on target attempted, shots on target allowed, shot attempted, shots allows, corner awarded, corners conceded, fouls commited, fouls suffered, yellow cards, red cards
			elo = ELO.ratings[team]
			last_five_matches = util.get_all_matches_of_team(data, team).tail(5)
			goals_scored = 0
			goals_conceded = 0
			goal_difference = 0
			points = 0
			change_in_ELO = 0 #, ta ELO ved nåværende minus første kamp ELO
			oppoisition_mean_ELO = 0 #Er først sum
			shots_on_target_attemped = 0
			shots_on_target_allowed = 0
			shots_attempted = 0
			shots_allowed = 0
			corners_awarded = 0
			corners_allowed = 0
			fouls_commited = 0
			fouls_suffered = 0
			yellow_cards = 0
			red_cards = 0
			i = 0
			for index, match in last_five_matches.iterrows():
				i += 1
				if team == match['HomeTeam']:
					goals_scored += match['FTHG']
					goals_conceded += match['FTAG']
					goal_difference += match['FTHG'] - match['FTAG']
					if match['FTR'] == 'H':
						points += 3
					elif match['FTR'] == 'D':
						points += 1
					if i == len(last_five_matches):
						change_in_ELO = elo - match['Home ELO']
					oppoisition_mean_ELO += match['Away ELO']
					shots_on_target_attemped += match['HST']
					shots_on_target_allowed += match['AST']
					shots_attempted += match['HS']
					shots_allowed += match['AS']
					corners_awarded += match['HC']
					corners_allowed += match['AC']
					fouls_commited +=  match['HF']
					fouls_suffered += match['AF']
					yellow_cards += match['HY']
					red_cards += match['HR']
					
				elif team == match['AwayTeam']:
					goals_scored += match['FTAG']
					goals_conceded += match['FTHG']
					goal_difference += match['FTAG'] - match['FTHG']
					if match['FTR'] == 'A':
						points += 3
					elif match['FTR'] == 'D':
						points += 1
					if i == len(last_five_matches):
						change_in_ELO = elo - match['Away ELO']
					oppoisition_mean_ELO += match['Home ELO']
					shots_on_target_attemped += match['AST']
					shots_on_target_allowed += match['HST']
					shots_attempted += match['AS']
					shots_allowed += match['HS']
					corners_awarded += match['AC']
					corners_allowed += match['HC']
					fouls_commited +=  match['AF']
					fouls_suffered += match['HF']
					yellow_cards += match['AY']
					red_cards += match['AR']
			oppoisition_mean_ELO = oppoisition_mean_ELO / len(last_five_matches)
			df_dict = {
				'Div': league,
				'Team': team,
				'ELO': elo,
				'Goals scored': goals_scored,
				'Goals conceded': goals_conceded,
				'Goals difference': goal_difference,
				'Points': points,
				'Change in ELO': change_in_ELO,
				'Opposition mean ELO': oppoisition_mean_ELO,
				'Shots on target attempted': shots_on_target_attemped,
				'Shots on target allows': shots_on_target_allowed,
				'Shots attemped': shots_attempted,
				'Shots allowed': shots_allowed,
				'Corners awarded': corners_awarded,
				'Corners allowed': corners_allowed,
				'Fouls commited': fouls_commited,
				'Fouls suffered': fouls_suffered,
				'Yellow cards': yellow_cards,
				'Red cards': red_cards
			}	
			df_tmp.append(df_dict)
	df_final = pd.DataFrame(df_tmp)
	df_final.to_csv(f'{data_folder}/current_data.csv', index=False)

def categorize_preds(pred_arr, hw, hl):
    categories = np.where(pred_arr > hw, 1, 
                          np.where((pred_arr <= hw) & (pred_arr > hl), 0, 
                                   -1))
    return np.array(categories)


def categorize_goal_diff(y_test):
    categories = np.where(y_test > 0, 1, 
                        np.where((y_test == 0 ), 0, 
                                -1))
    return np.array(categories)

def remove_uncertain(predictions, targets):
    # Create a boolean mask where predictions is not 0
    mask = predictions != 0
    
    # Use the mask to filter both predictions and targets
    filtered_predictions = predictions[mask]
    filtered_targets = targets[mask]
    
    return filtered_predictions, filtered_targets

if __name__ == '__main__':
	prep_data('./files', ['E0', 'E1', 'E2', 'E3', 'I1', 'SP1', 'D1', 'F1', 'N1'])