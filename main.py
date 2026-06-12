from dataset import load_data
from features import (
    create_target,
    add_elo_features,
    add_form_features,
    add_goals_features,
    add_win_rate_features,
    add_win_streak_features,
    add_days_since_last_match,
    add_neutral_and_tournament_features,
)
from model import MatchPredictor
from collections import defaultdict
import random



df = load_data()

df = create_target(df)
df = add_elo_features(df)
df = add_form_features(df)
df = add_goals_features(df)
df = add_win_rate_features(df)
df = add_win_streak_features(df)
df = add_days_since_last_match(df)
df = add_neutral_and_tournament_features(df)

feature_columns = [
    "home_elo", "away_elo", "elo_difference",
    "home_form_last5", "away_form_last5",
    "home_form_last10", "away_form_last10",
    "home_avg_goals_scored_last5", "away_avg_goals_scored_last5",
    "home_avg_goals_conceded_last5", "away_avg_goals_conceded_last5",
    "home_goal_difference_last5", "away_goal_difference_last5",
    "home_win_rate_last10", "away_win_rate_last10",
    "home_current_win_streak", "away_current_win_streak",
    "home_days_since_last_match", "away_days_since_last_match",
    "home_is_first_match", "away_is_first_match",
    "neutral", "tournament_importance",
]

X = df[feature_columns]
y = df["result"]

predictor = MatchPredictor()
accuracy, probabilities = predictor.train(X, y, df)

print(f"\n✓ Model trained successfully!")
print(f"✓ Total features used: {len(feature_columns)}")
print(f"✓ Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")


# ============================================================
# 2026 WORLD CUP GROUPS (final draw)
# ============================================================
# Team names mapped to your dataset's naming convention
groups = {
    "A": ["Mexico", "South Korea", "Czechia", "South Africa"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["USA", "Paraguay", "Australia", "Türkiye"],
    "E": ["Germany", "Curaçao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cabo Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}


# ============================================================
# Carry forward real latest per-team stats from the dataset
# ============================================================
STAT_COLS = {
    'elo':             ('home_elo', 'away_elo'),
    'form5':           ('home_form_last5', 'away_form_last5'),
    'form10':          ('home_form_last10', 'away_form_last10'),
    'goals_scored5':   ('home_avg_goals_scored_last5', 'away_avg_goals_scored_last5'),
    'goals_conceded5': ('home_avg_goals_conceded_last5', 'away_avg_goals_conceded_last5'),
    'goal_diff5':      ('home_goal_difference_last5', 'away_goal_difference_last5'),
    'win_rate10':      ('home_win_rate_last10', 'away_win_rate_last10'),
    'streak':          ('home_current_win_streak', 'away_current_win_streak'),
}

DEFAULTS = {
    'elo': 1500, 'form5': 0.5, 'form10': 0.5,
    'goals_scored5': 1.5, 'goals_conceded5': 1.5, 'goal_diff5': 0.0,
    'win_rate10': 0.5, 'streak': 0,
}

team_stats = defaultdict(lambda: dict(DEFAULTS))

df_sorted = df.sort_values("date")
for _, row in df_sorted.iterrows():
    for stat, (home_col, away_col) in STAT_COLS.items():
        team_stats[row["home_team"]][stat] = row[home_col]
        team_stats[row["away_team"]][stat] = row[away_col]

# Warn about teams not found in the dataset (fall back to defaults)
all_wc_teams = [t for g in groups.values() for t in g]
unseen = [t for t in all_wc_teams if t not in team_stats]
if unseen:
    print(f"\n⚠ Teams not found in dataset (using default stats): {unseen}")


def build_feature_vector(team1, team2):
    s1, s2 = team_stats[team1], team_stats[team2]
    return [
        s1['elo'], s2['elo'], s1['elo'] - s2['elo'],
        s1['form5'], s2['form5'], s1['form10'], s2['form10'],
        s1['goals_scored5'], s2['goals_scored5'],
        s1['goals_conceded5'], s2['goals_conceded5'],
        s1['goal_diff5'], s2['goal_diff5'],
        s1['win_rate10'], s2['win_rate10'],
        s1['streak'], s2['streak'],
        7, 7,    # days since last match (assume rested)
        0, 0,    # is_first_match flags
        1, 3,    # neutral=1 (World Cup), tournament_importance=3 (high)
    ]


def simulate_match(team1, team2):
    """Return (winner, loser). Draws resolved via penalties (stronger Elo wins)."""
    features = build_feature_vector(team1, team2)
    proba = predictor.model.predict_proba([features])[0]
    # proba[0] = away win (team2), proba[1] = draw, proba[2] = home win (team1)
    rand = random.random()

    if rand < proba[0]:
        return team2, team1
    elif rand < proba[0] + proba[1]:
        s1, s2 = team_stats[team1], team_stats[team2]
        return (team1, team2) if s1['elo'] >= s2['elo'] else (team2, team1)
    else:
        return team1, team2


# ============================================================
# Round of 32 — fixed slot structure (official 2026 bracket)
# ============================================================
# Fixed slots: pure 1st/2nd-place pairings, no third-place involved
FIXED_R32_SLOTS = [
    ("2A", "2B"),   # Match 73
    ("1F", "2C"),   # Match 75
    ("1C", "2F"),   # Match 76
    ("2E", "2I"),   # Match 78
    ("2K", "2L"),   # Match 83
    ("1H", "2J"),   # Match 84
    ("1J", "2H"),   # Match 86
    ("2D", "2G"),   # Match 87
]

# Wildcard slots: group winner vs best-ranked eligible 3rd-place team.
# Each slot lists its candidate groups in FIFA's published order.
WILDCARD_R32_SLOTS = [
    ("1E", ["A", "B", "C", "D", "F"]),   # Match 74
    ("1I", ["C", "D", "F", "G", "H"]),   # Match 77
    ("1A", ["C", "E", "F", "H", "I"]),   # Match 79
    ("1L", ["E", "H", "I", "J", "K"]),   # Match 80
    ("1D", ["B", "E", "F", "I", "J"]),   # Match 81
    ("1G", ["A", "E", "H", "I", "J"]),   # Match 82
    ("1B", ["E", "F", "G", "I", "J"]),   # Match 85
]
# Match 86 (1J vs 2H) above already covers the 14th fixed-style match;
# the remaining slot for the final wildcard third-place team:
EXTRA_WILDCARD_SLOT = ("1K", ["A", "B", "C", "D", "G"])  # last 3rd-place slot


def run_group_stage():
    """Round-robin within each group, returns standings per group."""
    standings = {}
    third_place_pool = []  # (group, team, points, goal_diff, goals_scored)

    for group_name, teams in groups.items():
        points = defaultdict(int)
        goals_for = defaultdict(int)
        goals_against = defaultdict(int)

        for i in range(len(teams)):
            for j in range(i + 1, len(teams)):
                t1, t2 = teams[i], teams[j]
                winner, loser = simulate_match(t1, t2)

                # Approximate goals: winner +1 GD, draws excluded since
                # simulate_match always returns a decisive result
                if winner == t1:
                    points[t1] += 3
                else:
                    points[t2] += 3
                goals_for[winner] += 2
                goals_for[loser] += 1
                goals_against[winner] += 1
                goals_against[loser] += 2

        ranked = sorted(
            teams,
            key=lambda t: (points[t], goals_for[t] - goals_against[t], goals_for[t]),
            reverse=True
        )

        standings[group_name] = {
            "1st": ranked[0],
            "2nd": ranked[1],
            "3rd": ranked[2],
            "4th": ranked[3],
        }

        third_team = ranked[2]
        third_place_pool.append((
            group_name, third_team,
            points[third_team],
            goals_for[third_team] - goals_against[third_team],
            goals_for[third_team]
        ))

    return standings, third_place_pool


def select_best_third_place(third_place_pool):
    """Rank all 12 third-place teams, return the best 8 with their group letters."""
    ranked = sorted(third_place_pool, key=lambda x: (x[2], x[3], x[4]), reverse=True)
    return [(g, t) for g, t, *_ in ranked[:8]]


def resolve_slot(slot_label, standings):
    """Resolve a slot label like '1A' or '2B' to an actual team."""
    pos_map = {"1": "1st", "2": "2nd", "3": "3rd"}
    pos, group = slot_label[0], slot_label[1:]
    return standings[group][pos_map[pos]]


def build_round_of_32(standings, third_place_pool):
    """Assign all 32 teams to their R32 matchups."""
    matchups = []

    # Fixed slots
    for a, b in FIXED_R32_SLOTS:
        matchups.append((resolve_slot(a, standings), resolve_slot(b, standings)))

    # Wildcard slots: assign best third-place teams to eligible slots
    best_thirds = select_best_third_place(third_place_pool)  # list of (group, team)
    available = list(best_thirds)

    all_wildcard_slots = WILDCARD_R32_SLOTS + [EXTRA_WILDCARD_SLOT]

    for slot_label, eligible_groups in all_wildcard_slots:
        winner_group = slot_label[1:]
        assigned = None

        for idx, (g, team) in enumerate(available):
            # Eligible for this slot, and not the same group as the opponent
            if g in eligible_groups and g != winner_group:
                assigned = available.pop(idx)
                break

        if assigned is None:
            # Fallback: take any remaining team, regardless of eligibility
            if available:
                assigned = available.pop(0)
            else:
                continue  # shouldn't happen with 8 slots / 8 teams

        matchups.append((resolve_slot(slot_label, standings), assigned[1]))

    return matchups


def simulate_knockout_round(matchups):
    winners = []
    for t1, t2 in matchups:
        winner, _ = simulate_match(t1, t2)
        winners.append(winner)
    return winners


def pair_up(teams):
    return [(teams[i], teams[i + 1]) for i in range(0, len(teams), 2)]


def simulate_world_cup():
    standings, third_place_pool = run_group_stage()
    r32_matchups = build_round_of_32(standings, third_place_pool)

    r16_teams = simulate_knockout_round(r32_matchups)        # 32 -> 16
    qf_teams = simulate_knockout_round(pair_up(r16_teams))   # 16 -> 8
    sf_teams = simulate_knockout_round(pair_up(qf_teams))    # 8 -> 4
    final_teams = simulate_knockout_round(pair_up(sf_teams)) # 4 -> 2

    if len(final_teams) >= 2:
        champion, _ = simulate_match(final_teams[0], final_teams[1])
        return champion
    return None


# ============================================================
# Run simulations
# ============================================================
champion_counts = defaultdict(int)
iterations = 1000

print(f"\nSimulating {iterations:,} World Cup tournaments...")
for i in range(iterations):
    if (i + 1) % 100 == 0:
        print(f"  Completed {i+1:,} / {iterations:,}...")
    champ = simulate_world_cup()
    if champ:
        champion_counts[champ] += 1

print(f"\n✓ Simulation complete!\n")
print("World Cup Winner Probabilities (Top 16):")
print("-" * 50)
for rank, (team, wins) in enumerate(
    sorted(champion_counts.items(), key=lambda x: x[1], reverse=True)[:16], 1
):
    prob = (wins / iterations) * 100
    bar = "█" * int(prob / 2)
    print(f"{rank:2d}. {team:20s} {prob:6.2f}% {bar}")

print("\n" + "=" * 60)