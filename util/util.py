import pandas as pd
import os
import numpy as np


def fetch_data(data_folder, start_year, end_year, leagues) -> pd.DataFrame:
    url_template = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"
    cols = [
        "Div",
        "Date",
        "HomeTeam",
        "AwayTeam",
        "FTHG",
        "FTAG",
        "FTR",
        "HTHG",
        "HTAG",
        "HTR",
        "Referee",
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
        "HBP",
    ]

    # Generate seasons list
    seasons = []
    for year in range(start_year, end_year):
        start = str(year)[-2:]
        end = str(year + 1)[-2:]
        seasons.append(start + end)

    df_tmp = []
    for season in seasons:
        for league in leagues:
            try:
                try:
                    print("Fetching data for", season, league)
                    df = pd.read_csv(url_template.format(season=season, league=league))
                except:
                    df = pd.read_csv(
                        url_template.format(season=season, league=league),
                        encoding="latin",
                    )
                try:
                    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%y")
                except ValueError:
                    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y")
            except:
                print(
                    "No data for",
                    season,
                    league,
                    url_template.format(season=season, league=league),
                )
                continue

            existing_cols = [col for col in cols if col in df.columns]
            df = df[existing_cols]
            df["Season"] = str(season).zfill(4)
            df_tmp.append(df)
    df = pd.concat(df_tmp)
    return df


####

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


def fetch_data_into_file(data_folder, file_name, start_year, end_year, leagues = ['E0']) -> None:
    url_template = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"
    cols = [
        "Div",
        "Date",
        "HomeTeam",
        "AwayTeam",
        "FTHG",
        "FTAG",
        "FTR",
        "HTHG",
        "HTAG",
        "HTR",
        "Referee",
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
        "HBP",
    ]

    # Generate seasons list
    seasons = []
    for year in range(start_year, end_year):
        start = str(year)[-2:]
        end = str(year + 1)[-2:]
        seasons.append(start + end)

    df_tmp = []
    for season in seasons:
        for league in leagues:
            try:
                try:
                    print("Fetching data for", season, league)
                    df = pd.read_csv(url_template.format(season=season, league=league))
                except:
                    df = pd.read_csv(
                        url_template.format(season=season, league=league),
                        encoding="latin",
                    )
                try:
                    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%y")
                except ValueError:
                    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y")
            except:
                print(
                    "No data for",
                    season,
                    league,
                    url_template.format(season=season, league=league),
                )
                continue

            existing_cols = [col for col in cols if col in df.columns]
            df = df[existing_cols]
            df["Season"] = str(season).zfill(4)
            df_tmp.append(df)
    df = pd.concat(df_tmp)

    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    file_path = os.path.join(data_folder, file_name + ".csv")
    df.to_csv(file_path, index=False)
    print("Data fetched and saved to", file_path)
    
def load_data(data_folder, file_name) -> pd.DataFrame:
    file_path = os.path.join(data_folder, file_name + ".csv")
    df = pd.read_csv(file_path, parse_dates=["Date"], dtype={"Season": str})
    return df

def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data = data.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    if "Referee" in data.columns:
        data.drop(columns="Referee", inplace=True)  # Fjerner kolonnen Referee
    data.dropna(inplace=True)  # Fjerner rader med manglende verdier
    data = data.reset_index(drop=True)
    return data

class ELO:  # Kan gjøre slik at home_advantage lages slik at den helles gir home_factor for kamper der hjemme og borte har samme rating
    def __init__(
        self, data, init_rating=1500, draw_factor=0.25, k_factor=32, home_advantage=100
    ):
        self.data = data
        self.init_rating = init_rating
        self.draw_factor = draw_factor
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.ratings = {}
        self.leagues_elo = {} #Fiks herfra: http://clubelo.com
        self.add_teams(data)

    def add_teams(self, data: pd.DataFrame):
        home_teams = data["HomeTeam"].unique()
        away_teams = data["AwayTeam"].unique()
        teams = list(set(home_teams) | set(away_teams))
        for team in teams:

            r = data[data["HomeTeam"] == team].iloc[0]

            if r["Div"] == "E0": #Dette må gjøres om til å tåle alle ligaer
                self.ratings[team] = self.init_rating
            elif r["Div"] == "E1":
                self.ratings[team] = self.init_rating - 200
            elif r["Div"] == "E2":
                self.ratings[team] = self.init_rating - 400
            elif r["Div"] == "E3":
                self.ratings[team] = self.init_rating - 600
            else:
                self.ratings[team] = self.init_rating

    def calculate_new_rating(self, home_elo, away_elo, result):
        if result == "H":
            s_home, s_away = 1, 0
        elif result == "D":
            s_home, s_away = 0.5, 0.5
        else:
            s_home, s_away = 0, 1
        e_home, e_d, e_away = self.expect_result(
            home_elo + self.home_advantage, away_elo
        )

        new_rating_home = home_elo + self.k_factor * (s_home - (e_home + e_d / 2))
        new_rating_away = away_elo + self.k_factor * (s_away - (e_away + e_d / 2))
        return new_rating_home, new_rating_away

    def expect_result(self, home_elo, away_elo):
        elo_diff = home_elo - away_elo
        excepted_home_without_draws = 1 / (1 + 10 ** (-elo_diff / 400))
        expected_away_without_draws = 1 / (1 + 10 ** (elo_diff / 400))
        real_expected_draw = self.draw_factor * (
            1 - abs(excepted_home_without_draws - expected_away_without_draws)
        )
        real_expected_home = excepted_home_without_draws - real_expected_draw / 2
        real_expected_away = expected_away_without_draws - real_expected_draw / 2
        return real_expected_home, real_expected_draw, real_expected_away

    def perform_matchup(self, home_team, away_team, result) -> None:
        try:
            old_rating_home = self.ratings[home_team]
            old_rating_away = self.ratings[away_team]
            new_rating_home, new_rating_away = self.calculate_new_rating(
                old_rating_home, old_rating_away, result
            )
            self.ratings[home_team] = new_rating_home
            self.ratings[away_team] = new_rating_away
            return old_rating_home, old_rating_away
        except KeyError:
            print("One or both teams does not exist")
            return None

    def perform_simulations(self, data) -> pd.DataFrame:
        data["Home ELO"] = None
        data["Away ELO"] = None
        data["ELO diff"] = None
        for index, row in data.iterrows():
            old_rating_home, old_rating_away = self.perform_matchup(
                row["HomeTeam"], row["AwayTeam"], row["FTR"]
            )
            data.at[index, "Home ELO"] = old_rating_home
            data.at[index, "Away ELO"] = old_rating_away
            data.at[index, "ELO diff"] = old_rating_home - old_rating_away
        for column in ["Home ELO", "Away ELO", "ELO diff"]:
            data[column] = pd.to_numeric(data[column])
        return data

    def get_probabilities(self, data) -> pd.DataFrame:
        data["Home_prob_ELO"] = None
        data["Draw_prob_ELO"] = None
        data["Away_prob_ELO"] = None
        for index, row in data.iterrows():
            home_prob, draw_prob, away_prob = self.expect_result(
                row["Home ELO"] + self.home_advantage, row["Away ELO"]
            )
            data.at[index, "Home_prob_ELO"] = home_prob
            data.at[index, "Draw_prob_ELO"] = draw_prob
            data.at[index, "Away_prob_ELO"] = away_prob
        for column in ["Home_prob_ELO", "Draw_prob_ELO", "Away_prob_ELO"]:
            data[column] = pd.to_numeric(data[column])
        return data


def extract_elo_history(data, team) -> pd.DataFrame:
    elo_history = []
    for index, row in data.iterrows():
        if row["HomeTeam"] == team:
            elo_history.append(
                {
                    "Date": row["Date"],
                    "Opponent": row["AwayTeam"],
                    "ELO": row["Home ELO"],
                    "Result": row["FTR"],
                }
            )
        elif row["AwayTeam"] == team:
            elo_history.append(
                {
                    "Date": row["Date"],
                    "Opponent": row["HomeTeam"],
                    "ELO": row["Away ELO"],
                    "Result": row["FTR"],
                }
            )
    return pd.DataFrame(elo_history)

def get_all_matches_of_team(data, team):
    c = data.copy()
    return c[(c["HomeTeam"] == team) | (c["AwayTeam"] == team)]

def add_form_column(
    data: pd.DataFrame,
    home_column,
    away_column,
    n=5,
    operation="Sum",
    regard_opponent=False,
    include_current=False,
):
    """
    Function that performs the operation on the n last matches for each team.
    If regard_opponent is True, the operation is performed on the opponents column instead.
    Example: Home_column = FTHG, Away_column=FTAG, Operation = Sum, n = 5, regard_opponent = False creates columns to describe how many goals the team has scored in the last 5 matches.
    Example: Home_column = FTHG, Away_column=FTAG, Operation = Sum, n = 5, regard_opponent = True creates columns to describe how many of goals the team has conceded in the last 5 matches.
    Args:
            data (pd.DataFrame): The dataframe to add the columns to.
            home_column (str): The column to use if the team is at home.
            away_column (str): The column to use if the team is away.
            n (int): The number of matches to consider.
            operation (str): The operation to perform. Can be 'Sum', 'Mean' or 'Change'.
            regard_opponent (bool): If True, the operation is performed on the opponents column instead. E.g. Can be used to get mean of opponent ELO
            include_current (bool): If True, the current match is included in the operation. Used if column is already dependent on previous matches, such as Home ELO and Away ELO.
    """
    new_column_name_home = (
        home_column
        + "_"
        + operation
        + "_"
        + str(n)
        + ("_opponent" if regard_opponent else "")
    )
    new_column_name_away = (
        away_column
        + "_"
        + operation
        + "_"
        + str(n)
        + ("_opponent" if regard_opponent else "")
    )
    data[new_column_name_home] = None
    data[new_column_name_away] = None
    teams = data["HomeTeam"].unique()
    for team in teams:
        matches = get_all_matches_of_team(data, team)
        scores = {}
        pos = 0 if not include_current else 1
        for index, row in matches.iterrows():
            start_pos = max(0, pos - n)
            relevant_matches = matches.iloc[start_pos:pos]
            s = 0
            if operation == "Sum":
                for index_r, row_r in relevant_matches.iterrows():
                    if row_r["HomeTeam"] == team:
                        if regard_opponent:
                            s += row_r[away_column]
                        else:
                            s += row_r[home_column]
                    else:
                        if regard_opponent:
                            s += row_r[home_column]
                        else:
                            s += row_r[away_column]
            elif operation == "Mean":
                for index_r, row_r in relevant_matches.iterrows():
                    if row_r["HomeTeam"] == team:
                        if regard_opponent:
                            s += row_r[away_column]
                        else:
                            s += row_r[home_column]
                    else:
                        if regard_opponent:
                            s += row_r[home_column]
                        else:
                            s += row_r[away_column]
                if len(relevant_matches) == 0:
                    s = 0
                else:
                    s = s / len(relevant_matches)
            elif operation == "Change":
                if len(relevant_matches) == 0:
                    s = 0
                else:
                    first_row = relevant_matches.iloc[0]
                    last_row = relevant_matches.iloc[-1]
                    first_score = (
                        first_row[home_column]
                        if first_row["HomeTeam"] == team
                        else first_row[away_column]
                    )
                    last_score = (
                        last_row[home_column]
                        if last_row["HomeTeam"] == team
                        else last_row[away_column]
                    )
                    s = last_score - first_score
            elif operation == "Points":
                for index_r, row_r in relevant_matches.iterrows():
                    if row_r["HomeTeam"] == team:
                        if row_r["FTHG"] > row_r["FTAG"]:
                            s += 3
                        elif row_r["FTHG"] == row_r["FTAG"]:
                            s += 1
                        else:
                            s += 0
                    else:
                        if row_r["FTAG"] > row_r["FTHG"]:
                            s += 3
                        elif row_r["FTAG"] == row_r["FTHG"]:
                            s += 1
                        else:
                            s += 0
            scores[index] = s
            pos += 1

        for key, value in scores.items():
            if data.at[key, "HomeTeam"] == team:
                data.at[key, new_column_name_home] = value
            else:
                data.at[key, new_column_name_away] = value
    data[new_column_name_home] = pd.to_numeric(
        data[new_column_name_home], errors="coerce"
    )
    data[new_column_name_away] = pd.to_numeric(
        data[new_column_name_away], errors="coerce"
    )
    return data
