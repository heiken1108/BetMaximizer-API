import os
from util import util
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_predict
import pandas as pd






def fetch_data():
	data_folder = os.path.join('.', 'files/data')
	file_name = "Data"
	util.fetch_data_into_file(data_folder, file_name, 2005, 2025, leagues)

def data_prep():
	data_folder = os.path.join('.', 'files/data')
	raw_file_name = "Data"
	#Load
	data = util.load_data(data_folder, raw_file_name)
	#Clean
	data = util.clean_data(data)
	#ELO
	home_factor, draw_factor, away_factor = data['FTR'].value_counts(normalize=True)['H'], data['FTR'].value_counts(normalize=True)['D'], data['FTR'].value_counts(normalize=True)['A']
	ELO = util.ELO(data, init_rating=1500, draw_factor=draw_factor, k_factor=32, home_advantage=50)
	data = ELO.perform_simulations(data)
	#Legg til Andre features
	data = util.add_form_column(data, 'FTHG', 'FTAG', n=5, operation='Sum', regard_opponent=False, include_current=False)
	data['Diff_goals_scored'] = data['FTHG_Sum_5'] - data['FTAG_Sum_5']
	data = util.add_form_column(data, 'FTHG', 'FTAG', n=5, operation='Sum', regard_opponent=True, include_current=False)
	data['Diff_goals_conceded'] = data['FTHG_Sum_5_opponent'] - data['FTAG_Sum_5_opponent']
	data['Home Goal Difference last 5'] = data['FTHG_Sum_5'] - data['FTHG_Sum_5_opponent']
	data['Away Goal Difference last 5'] = data['FTAG_Sum_5'] - data['FTAG_Sum_5_opponent']
	data['Diff_goal_diff'] = data['Home Goal Difference last 5'] - data['Away Goal Difference last 5']
	data = util.add_form_column(data, 'Home', 'Away', n=5, operation='Points', regard_opponent=False, include_current=False)
	data['Diff_points'] = data['Home_Points_5'] - data['Away_Points_5']
	data = util.add_form_column(data, 'Home ELO', 'Away ELO', n=5, operation='Change', regard_opponent=False, include_current=True)
	data['Diff_change_in_ELO'] = data['Home ELO_Change_5'] - data['Away ELO_Change_5']
	data = util.add_form_column(data, 'Home ELO', 'Away ELO', n=5, operation='Mean', regard_opponent=True, include_current=False)
	data['Diff_opposition_mean_ELO'] = data['Home ELO_Mean_5_opponent'] - data['Away ELO_Mean_5_opponent']
	data = util.add_form_column(data, 'HST', 'AST', n=5, operation='Sum', regard_opponent=False, include_current=False)
	data['Diff_shots_on_target_attempted'] = data['HST_Sum_5'] - data['AST_Sum_5']
	data = util.add_form_column(data, 'HST', 'AST', n=5, operation='Sum', regard_opponent=True, include_current=False)
	data['Diff_shots_on_target_allowed'] = data['HST_Sum_5_opponent'] - data['AST_Sum_5_opponent']
	data = util.add_form_column(data, 'HS', 'AS', n=5, operation='Sum', regard_opponent=False, include_current=False)
	data['Diff_shots_attempted'] = data['HS_Sum_5'] - data['AS_Sum_5']
	data = util.add_form_column(data, 'HS', 'AS', n=5, operation='Sum', regard_opponent=True, include_current=False)
	data['Diff_shots_allowed'] = data['HS_Sum_5_opponent'] - data['AS_Sum_5_opponent']
	data = util.add_form_column(data, 'HC', 'AC', n=5, operation='Sum', regard_opponent=False, include_current=False)
	data['Diff_corners_awarded'] = data['HC_Sum_5'] - data['AC_Sum_5']
	data = util.add_form_column(data, 'HC', 'AC', n=5, operation='Sum', regard_opponent=True, include_current=False)
	data['Diff_corners_conceded'] = data['HC_Sum_5_opponent'] - data['AC_Sum_5_opponent']
	data = util.add_form_column(data, 'HF', 'AF', n=5, operation='Sum', regard_opponent=False, include_current=False)
	data['Diff_fouls_commited'] = data['HF_Sum_5'] - data['AF_Sum_5']
	data = util.add_form_column(data, 'HF', 'AF', n=5, operation='Sum', regard_opponent=True, include_current=False)
	data['Diff_fouls_suffered'] = data['HF_Sum_5_opponent'] - data['AF_Sum_5_opponent']
	data = util.add_form_column(data, 'HY', 'AY', n=5, operation='Sum', regard_opponent=False, include_current=False)
	data['Diff_yellow_cards'] = data['HY_Sum_5'] - data['AY_Sum_5']
	data = util.add_form_column(data, 'HR', 'AR', n=5, operation='Sum', regard_opponent=False, include_current=False)
	data['Diff_red_cards'] = data['HR_Sum_5'] - data['AR_Sum_5']
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
	data.drop(columns=columns_to_remove, inplace=True)

	#Fjerne unødvendige features
	#Lagre til fil
	data.to_csv("files/data/Prepared_data.csv", index=False)


def train_models():
	data_folder = os.path.join('.', 'files/data')
	data = util.load_data(data_folder, "Prepared_data")

	# Add Outcome column
	data["Outcome"] = data.apply(
		lambda row: (row["FTHG"] - row["FTAG"]),
		axis=1,
	)
	data = data[data['Div'] == 'E0'] 
	#data = data[data['Date'] > '2007-08-01']

	# Prepare features (X) and target (y)
	X = data.copy().drop(
		columns=["Outcome", "FTHG", "FTAG", "Season", "Div", "Date", "HomeTeam", "AwayTeam"],
	)
	y = data["Outcome"]

	# Split data into training and testing sets
	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=0.2, random_state=42
	)
	

	# Perform Random Forest Regression
	rf = RandomForestRegressor(n_estimators=200, random_state=42)
	rf.fit(X_train, y_train)
	data['Predicted_Outcome'] = cross_val_predict(rf, X, y, cv=5)
	print(data.tail(20))

	data = data.iloc[5000:]
	predicted_over_0 = (data['Predicted_Outcome'] > 0).sum()
	predicted_exactly_0 = (data['Predicted_Outcome'] == 0).sum()
	predicted_below_0 = (data['Predicted_Outcome'] < 0).sum()
	Actual_over_0 = (data['Outcome'] > 0).sum()
	Actual_exactly_0 = (data['Outcome'] == 0).sum()
	Actual_below_0 = (data['Outcome'] < 0).sum()
	print('Home wins', predicted_over_0, Actual_over_0, 'Accuracy:', (Actual_over_0/predicted_over_0))
	print('Away wins', predicted_below_0, Actual_below_0, 'Accuracy:', (Actual_below_0/predicted_below_0))
	
	# Predict on the test set
	predictions = rf.predict(X_test)

	#Bruk regresjonstall til å kategorisee

	categorized_preds = categorize_preds(predictions, thresholds['hw'], thresholds['hl'])
	categorized_goal_diff = categorize_goal_diff(y_test)
	print('Predicted draws amount:', len(categorized_preds[categorized_preds == 0]))
	print('Actual draws amount:', len(categorized_goal_diff[categorized_goal_diff == 0]))

	report = classification_report(categorized_goal_diff, categorized_preds)
	print(report)
	filtered_predictions, filtered_targets = remove_uncertain(categorized_preds, categorized_goal_diff)
	report = classification_report(filtered_predictions, filtered_targets)
	print(report)
	print(f'Bet Accceptance Rate: {len(filtered_predictions)/len(categorized_preds)}')

thresholds = {"hw": 1.0, "hl": -1.0}

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
	leagues = ['E0', 'E1', 'E2', 'E3']
	train_models()
