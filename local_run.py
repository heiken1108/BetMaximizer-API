import pandas as pd
from joblib import dump, load

def get_team_row(team_name):
    data_folder = 'files/data'
    df = pd.read_csv(f'{data_folder}/current_data.csv')
    # Filter the DataFrame for the given team name
    team_row = df.loc[df['Team'] == team_name]

    if not team_row.empty:
        return team_row.iloc[0]  # Return the first row (should be the only one if unique)
    else:
        return None  # Return None if the team is not found
    

def get_diff_dataframe(home_team, away_team) -> pd.DataFrame:
	home_row = get_team_row(home_team)
	away_row = get_team_row(away_team)
	diff_df = {
		'ELO diff': home_row['ELO'] - away_row['ELO'],
		'Diff_goals_scored': home_row['Goals scored'] - away_row['Goals scored'],
		'Diff_goals_conceded': home_row['Goals conceded'] - away_row['Goals conceded'],
		'Diff_goal_diff': home_row['Goals difference'] - away_row['Goals difference'],
		'Diff_points': home_row['Points'] - away_row['Points'],
		'Diff_change_in_ELO': home_row['Change in ELO'] - away_row['Change in ELO'],
		'Diff_opposition_mean_ELO': home_row['Opposition mean ELO'] - away_row['Opposition mean ELO'],
		'Diff_shots_on_target_attempted': home_row['Shots on target attempted'] - away_row['Shots on target attempted'],
		'Diff_shots_on_target_allowed': home_row['Shots on target allows'] - away_row['Shots on target allows'],
		'Diff_shots_attempted': home_row['Shots attemped'] - away_row['Shots attemped'],
		'Diff_shots_allowed': home_row['Shots allowed'] - away_row['Shots allowed'],
		'Diff_corners_awarded': home_row['Corners awarded'] - away_row['Corners awarded'],
		'Diff_corners_conceded': home_row['Corners allowed'] - away_row['Corners allowed'],
		'Diff_fouls_commited': home_row['Fouls commited'] - away_row['Fouls commited'],
		'Diff_fouls_suffered': home_row['Fouls suffered'] - away_row['Fouls suffered'],
		'Diff_yellow_cards': home_row['Yellow cards'] - away_row['Yellow cards'],
		'Diff_red_cards': home_row['Red cards'] - away_row['Red cards']
	}
	return pd.DataFrame([diff_df])

def get_prediction(home_team, away_team, league):
	model = load(f'files/models/{league}_model.joblib')
	input = get_diff_dataframe(home_team, away_team).values
	# Predict the match result
	pred = model.predict(input)
	return pred

if __name__ == '__main__':
    home_team = "Wolves"
    away_team = "Nott'm Forest"
    league = 'E0'
    print(f'{home_team} v. {away_team} has xGD of {get_prediction(home_team, away_team, league)[0]}')