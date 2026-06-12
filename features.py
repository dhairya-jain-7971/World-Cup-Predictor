import pandas as pd
from collections import defaultdict

def create_target(df):
    df["result"] = 0
    df.loc[df["home_score"] > df["away_score"], "result"] = 2  # Home win
    df.loc[df["home_score"] < df["away_score"], "result"] = 0  # Away win
    df.loc[df["home_score"] == df["away_score"], "result"] = 1  # Draw
    return df

INITIAL_ELO = 1500
K = 32

def add_elo_features(df):
    # Local dict — no cross-run leakage / mutation
    team_elo = defaultdict(lambda: INITIAL_ELO)

    home_elos, away_elos, elo_diffs = [], [], []

    df = df.sort_values("date")

    for _, row in df.iterrows():
        home_team, away_team = row["home_team"], row["away_team"]
        home_elo, away_elo = team_elo[home_team], team_elo[away_team]

        home_elos.append(home_elo)
        away_elos.append(away_elo)
        elo_diffs.append(home_elo - away_elo)

        expected_home = 1 / (1 + 10 ** ((away_elo - home_elo) / 400))
        expected_away = 1 - expected_home

        if row["home_score"] > row["away_score"]:
            actual_home, actual_away = 1, 0
        elif row["home_score"] < row["away_score"]:
            actual_home, actual_away = 0, 1
        else:
            actual_home, actual_away = 0.5, 0.5

        team_elo[home_team] += K * (actual_home - expected_home)
        team_elo[away_team] += K * (actual_away - expected_away)

    df["home_elo"] = home_elos
    df["away_elo"] = away_elos
    df["elo_difference"] = elo_diffs
    return df


def add_form_features(df):
    team_history = defaultdict(list)
    home_forms_5, away_forms_5 = [], []
    home_forms_10, away_forms_10 = [], []

    df = df.sort_values("date")

    for _, row in df.iterrows():
        home_team, away_team = row["home_team"], row["away_team"]

        home_last5 = team_history[home_team][-5:]
        away_last5 = team_history[away_team][-5:]
        home_last10 = team_history[home_team][-10:]
        away_last10 = team_history[away_team][-10:]

        home_forms_5.append(sum(home_last5)/len(home_last5) if home_last5 else 0.5)
        away_forms_5.append(sum(away_last5)/len(away_last5) if away_last5 else 0.5)
        home_forms_10.append(sum(home_last10)/len(home_last10) if home_last10 else 0.5)
        away_forms_10.append(sum(away_last10)/len(away_last10) if away_last10 else 0.5)

        if row["home_score"] > row["away_score"]:
            team_history[home_team].append(1)
            team_history[away_team].append(0)
        elif row["home_score"] < row["away_score"]:
            team_history[home_team].append(0)
            team_history[away_team].append(1)
        else:
            team_history[home_team].append(0.5)
            team_history[away_team].append(0.5)

    df["home_form_last5"] = home_forms_5
    df["away_form_last5"] = away_forms_5
    df["home_form_last10"] = home_forms_10
    df["away_form_last10"] = away_forms_10
    return df


def add_goals_features(df):
    team_goals_for = defaultdict(list)
    team_goals_against = defaultdict(list)

    home_avg_scored_5, away_avg_scored_5 = [], []
    home_avg_conceded_5, away_avg_conceded_5 = [], []
    home_goal_diff_5, away_goal_diff_5 = [], []

    df = df.sort_values("date")

    for _, row in df.iterrows():
        home_team, away_team = row["home_team"], row["away_team"]

        h_for, h_against = team_goals_for[home_team][-5:], team_goals_against[home_team][-5:]
        a_for, a_against = team_goals_for[away_team][-5:], team_goals_against[away_team][-5:]

        h_avg_s = sum(h_for)/len(h_for) if h_for else 0
        h_avg_c = sum(h_against)/len(h_against) if h_against else 0
        a_avg_s = sum(a_for)/len(a_for) if a_for else 0
        a_avg_c = sum(a_against)/len(a_against) if a_against else 0

        home_avg_scored_5.append(h_avg_s)
        home_avg_conceded_5.append(h_avg_c)
        away_avg_scored_5.append(a_avg_s)
        away_avg_conceded_5.append(a_avg_c)
        home_goal_diff_5.append(h_avg_s - h_avg_c)
        away_goal_diff_5.append(a_avg_s - a_avg_c)

        team_goals_for[home_team].append(row["home_score"])
        team_goals_for[away_team].append(row["away_score"])
        team_goals_against[home_team].append(row["away_score"])
        team_goals_against[away_team].append(row["home_score"])

    df["home_avg_goals_scored_last5"] = home_avg_scored_5
    df["away_avg_goals_scored_last5"] = away_avg_scored_5
    df["home_avg_goals_conceded_last5"] = home_avg_conceded_5
    df["away_avg_goals_conceded_last5"] = away_avg_conceded_5
    df["home_goal_difference_last5"] = home_goal_diff_5
    df["away_goal_difference_last5"] = away_goal_diff_5
    return df


def add_win_rate_features(df):
    team_results = defaultdict(list)
    home_win_rate_10, away_win_rate_10 = [], []

    df = df.sort_values("date")

    for _, row in df.iterrows():
        home_team, away_team = row["home_team"], row["away_team"]

        h_last10 = team_results[home_team][-10:]
        a_last10 = team_results[away_team][-10:]

        home_win_rate_10.append(sum(1 for r in h_last10 if r == 1)/len(h_last10) if h_last10 else 0)
        away_win_rate_10.append(sum(1 for r in a_last10 if r == 1)/len(a_last10) if a_last10 else 0)

        if row["home_score"] > row["away_score"]:
            team_results[home_team].append(1)
            team_results[away_team].append(0)
        elif row["home_score"] < row["away_score"]:
            team_results[home_team].append(0)
            team_results[away_team].append(1)
        else:
            team_results[home_team].append(0.5)
            team_results[away_team].append(0.5)

    df["home_win_rate_last10"] = home_win_rate_10
    df["away_win_rate_last10"] = away_win_rate_10
    return df


def add_win_streak_features(df):
    team_streaks = defaultdict(int)
    team_last_results = defaultdict(lambda: None)

    home_streaks, away_streaks = [], []

    df = df.sort_values("date")

    for _, row in df.iterrows():
        home_team, away_team = row["home_team"], row["away_team"]

        home_streaks.append(team_streaks[home_team])
        away_streaks.append(team_streaks[away_team])

        if row["home_score"] > row["away_score"]:
            home_result, away_result = 1, 0
        elif row["home_score"] < row["away_score"]:
            home_result, away_result = 0, 1
        else:
            home_result, away_result = 0.5, 0.5

        team_streaks[home_team] = team_streaks[home_team] + 1 if (home_result == 1 and team_last_results[home_team] == 1) else (1 if home_result == 1 else 0)
        team_streaks[away_team] = team_streaks[away_team] + 1 if (away_result == 1 and team_last_results[away_team] == 1) else (1 if away_result == 1 else 0)

        team_last_results[home_team] = home_result
        team_last_results[away_team] = away_result

    df["home_current_win_streak"] = home_streaks
    df["away_current_win_streak"] = away_streaks
    return df


# Sentinel for "no prior match" — large value, not 0
NO_PRIOR_MATCH_DAYS = 365

def add_days_since_last_match(df):
    team_last_match_date = defaultdict(lambda: None)
    home_days, away_days = [], []
    home_is_first, away_is_first = [], []

    df = df.sort_values("date")

    for _, row in df.iterrows():
        home_team, away_team = row["home_team"], row["away_team"]
        current_date = row["date"]

        if team_last_match_date[home_team] is None:
            home_days.append(NO_PRIOR_MATCH_DAYS)
            home_is_first.append(1)
        else:
            home_days.append((current_date - team_last_match_date[home_team]).days)
            home_is_first.append(0)

        if team_last_match_date[away_team] is None:
            away_days.append(NO_PRIOR_MATCH_DAYS)
            away_is_first.append(1)
        else:
            away_days.append((current_date - team_last_match_date[away_team]).days)
            away_is_first.append(0)

        team_last_match_date[home_team] = current_date
        team_last_match_date[away_team] = current_date

    df["home_days_since_last_match"] = home_days
    df["away_days_since_last_match"] = away_days
    df["home_is_first_match"] = home_is_first
    df["away_is_first_match"] = away_is_first
    return df


def add_neutral_and_tournament_features(df):
    if "neutral" not in df.columns:
        df["neutral"] = 0
    if "tournament_importance" not in df.columns:
        df["tournament_importance"] = 1
    return df