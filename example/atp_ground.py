"""
ATP Tennis Player Skill Analysis with Surface-Specific Modeling
================================================================

This example demonstrates how to use TrueSkillThroughTime to analyze tennis player
performance while accounting for surface-specific skills (clay, grass, hard court).

The model creates a "virtual" player for each surface, allowing it to capture how
a player's skill varies depending on the court type. This is achieved by having
each player represented twice in every match: once for their general skill and
once for their surface-specific skill (e.g., "Federer" and "Federer_grass").

Data Format:
-----------
The input CSV should contain the following columns:
- w1_id, w2_id: Winner player IDs (w2_id used for doubles)
- l1_id, l2_id: Loser player IDs (l2_id used for doubles)
- double: 't' for doubles matches, other values for singles
- ground: Surface type identifier (e.g., 'clay', 'grass', 'hard')
- time_start: Match date in 'YYYY-MM-DD' format

Model Parameters:
----------------
- beta: Home advantage (0.0 for tennis, as there's no home advantage)
- sigma: Performance noise/variability (1.0 indicates moderate day-to-day variation)
- gamma: Global skill change rate over time (0.01 for gradual changes)
- Player priors: Initial skill distribution (mean=0.0, std=1.6, gamma=0.036)

Output:
-------
After convergence, the History object contains skill distributions for all players
over time, accessible via h.learning_curves() and other History methods.
"""

import pandas as pd
# sudo pip3 install trueskillthroughtime
# import sys
# sys.path.append('..')
from trueskillthroughtime import *
import time
from datetime import datetime

# ============================================================================
# Data Loading
# ============================================================================
# Load the CSV file containing historical tennis match data
df = pd.read_csv('input/history.csv', low_memory=False)

# ============================================================================
# Match Composition Setup
# ============================================================================
# Create match compositions where each match includes both the original player
# and their "surface version" (player_id + ground suffix).
# 
# For doubles (d == 't'): teams have 4 entries each [[w1,w1+g,w2,w2+g],[l1,l1+g,l2,l2+g]]
# For singles: teams have 2 entries each [[w1,w1+g],[l1,l1+g]]
# 
# This dual representation allows the model to learn both:
# - General player skill (w1)
# - Surface-specific skill adjustment (w1+g)
columns = zip(df.w1_id, df.w2_id, df.l1_id, df.l2_id, df.double, df.ground)
composition = [[[w1,w1+g,w2,w2+g],[l1,l1+g,l2,l2+g]] if d == 't' else [[w1,w1+g],[l1,l1+g]] for w1, w2, l1, l2, d, g in columns ]

# ============================================================================
# Time Conversion
# ============================================================================
# Convert match dates to timestamps in days since epoch
# This temporal information allows the algorithm to model skill evolution over time
times = [ datetime.strptime(t, "%Y-%m-%d").timestamp()/(60*60*24) for t in df.time_start]

# ============================================================================
# Player Initialization
# ============================================================================
# Extract unique player IDs from the dataset
columns = zip(df.w1_id, df.w2_id, df.l1_id, df.l2_id)
player_ids = set([ player for game in columns for player in game ])

# Initialize prior distributions for each player:
# - Gaussian(0., 1.6): Initial skill belief (mean=0, std=1.6)
# - beta=1.0: Home advantage parameter (placeholder for tennis)
# - gamma=0.036: Per-player skill drift rate (how fast individual skills change)
priors = dict([(p, Player(Gaussian(0., 1.6), 1.0, 0.036) ) for p in player_ids])

# ============================================================================
# Model Construction and Convergence
# ============================================================================
# Create the History object with all match data and model parameters:
# - composition: List of all matches with their team structures
# - times: Temporal ordering of matches
# - beta=0.0: No home advantage (appropriate for professional tennis)
# - sigma=1.0: Performance noise (standard deviation of day-to-day variation)
# - gamma=0.01: Global skill change rate (gradual evolution)
# - priors: Initial player skill distributions
h = History(composition = composition, times = times, beta = 0.0, sigma = 1.0, gamma = 0.01, priors = priors)

# Run the convergence algorithm:
# - epsilon=0.01: Convergence threshold (minimum change between iterations)
# - iterations=10: Maximum number of iterations to run
h.convergence(epsilon=0.01, iterations=10)

# After convergence, use h.learning_curves() to access player skill trajectories
