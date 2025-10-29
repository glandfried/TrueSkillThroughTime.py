#!/usr/bin/env python
"""
Example demonstrating incremental updates using History.add_events()

This shows how to:
1. Create an initial History with some games
2. Add new games incrementally without recomputing from scratch
3. Handle both with and without timestamps
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import trueskillthroughtime as ttt

def main():
    print("=" * 60)
    print("TrueSkillThroughTime: Incremental Update Example")
    print("=" * 60)

    # Example 1: Basic incremental update without timestamps
    print("\n1. Basic Incremental Update (no timestamps)")
    print("-" * 40)

    # Initial games
    composition_initial = [
        [["Alice"], ["Bob"]],      # Alice vs Bob
        [["Alice"], ["Charlie"]],  # Alice vs Charlie
    ]
    results_initial = [
        [1, 0],  # Alice beats Bob
        [0, 1],  # Charlie beats Alice
    ]

    # Create initial history
    h = ttt.History(composition_initial, results_initial,
                   mu=25.0, sigma=8.333, beta=4.166, gamma=0.083)

    # Run convergence
    h.convergence(epsilon=1e-4, iterations=10, verbose=False)

    # Print initial skills
    print("Initial skills after 2 games:")
    lc = h.learning_curves()
    for player in ["Alice", "Bob", "Charlie"]:
        if player in lc:
            final_skill = lc[player][-1][1]
            print(f"  {player}: μ={final_skill.mu:.2f}, σ={final_skill.sigma:.2f}")

    # Add new games incrementally
    print("\nAdding new games...")
    new_composition = [
        [["Bob"], ["Charlie"]],    # Bob vs Charlie
        [["Alice"], ["Bob"]],      # Alice vs Bob (rematch)
    ]
    new_results = [
        [1, 0],  # Bob beats Charlie
        [0, 1],  # Bob beats Alice
    ]

    h.add_events(new_composition, new_results)

    # Run convergence again
    h.convergence(epsilon=1e-4, iterations=10, verbose=False)

    # Print updated skills
    print("\nUpdated skills after 4 games total:")
    lc = h.learning_curves()
    for player in ["Alice", "Bob", "Charlie"]:
        if player in lc:
            final_skill = lc[player][-1][1]
            print(f"  {player}: μ={final_skill.mu:.2f}, σ={final_skill.sigma:.2f}")

    # Example 2: Incremental update with timestamps
    print("\n2. Incremental Update with Timestamps")
    print("-" * 40)

    # Initial games with timestamps
    composition_ts = [
        [["Team1"], ["Team2"]],
        [["Team1"], ["Team3"]],
    ]
    results_ts = [[1, 0], [1, 0]]
    times_ts = [100, 200]  # Day 100 and 200

    # Create history with timestamps
    h_ts = ttt.History(composition_ts, results_ts, times_ts,
                      mu=25.0, sigma=8.333, beta=4.166, gamma=0.25)

    h_ts.convergence(epsilon=1e-4, iterations=10, verbose=False)

    print("Initial skills (with time decay):")
    lc_ts = h_ts.learning_curves()
    for team in ["Team1", "Team2", "Team3"]:
        if team in lc_ts:
            final_skill = lc_ts[team][-1][1]
            print(f"  {team}: μ={final_skill.mu:.2f}, σ={final_skill.sigma:.2f}")

    # Add games at later times
    print("\nAdding games at days 300 and 400...")
    new_comp_ts = [
        [["Team2"], ["Team3"]],
        [["Team1"], ["Team2"]],
    ]
    new_res_ts = [[0, 1], [1, 0]]
    new_times = [300, 400]

    h_ts.add_events(new_comp_ts, new_res_ts, new_times)
    h_ts.convergence(epsilon=1e-4, iterations=10, verbose=False)

    print("\nFinal skills (showing time decay effect):")
    lc_ts = h_ts.learning_curves()
    for team in ["Team1", "Team2", "Team3"]:
        if team in lc_ts:
            final_skill = lc_ts[team][-1][1]
            print(f"  {team}: μ={final_skill.mu:.2f}, σ={final_skill.sigma:.2f}")

    # Example 3: Adding new players
    print("\n3. Adding New Players Incrementally")
    print("-" * 40)

    # Add games with new players
    new_players_comp = [
        [["David"], ["Eve"]],      # Two new players
        [["Alice"], ["David"]],    # Existing vs new player
    ]
    new_players_res = [[1, 0], [1, 0]]

    print("Adding games with new players David and Eve...")
    h.add_events(new_players_comp, new_players_res)
    h.convergence(epsilon=1e-4, iterations=10, verbose=False)

    print("\nAll player skills including new players:")
    lc = h.learning_curves()
    for player in ["Alice", "Bob", "Charlie", "David", "Eve"]:
        if player in lc:
            final_skill = lc[player][-1][1]
            print(f"  {player}: μ={final_skill.mu:.2f}, σ={final_skill.sigma:.2f}")

    print("\n" + "=" * 60)
    print("Incremental updates complete!")
    print("This allows you to maintain a running skill estimate")
    print("without recomputing everything from scratch.")
    print("=" * 60)

if __name__ == "__main__":
    main()