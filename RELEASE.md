# v2.0.0

## Major Changes

This major release brings significant enhancements to the `trueskillthroughtime` package while maintaining the same module name and package structure.

### Multiple Observation Models
- **Ordinal Model**: Traditional ranking-based outcomes (win/loss/draw)
- **Continuous Model**: Continuous score differences (e.g., time, distance)
- **Discrete Model**: Discrete count-based scores (e.g., goals, points)
  - Implements Poisson-based likelihood with fixed-point approximation (Guo et al. 2012)
  - Supports score-based Bayesian skill learning

**Mixed Observation Models in History:**
The `obs` parameter in `History` allows flexible observation model specification:
- **Single model for all games**: Pass a list with one element, e.g., `obs=["Ordinal"]`
- **Per-game model**: Pass a list with one element per game, e.g., `obs=["Ordinal", "Discrete", "Continuous"]`

```python
# Example: Mix different observation models in the same history
composition = [
    [["team_a"], ["team_b"]],  # Ordinal game (win/loss)
    [["player_1"], ["player_2"]],  # Discrete game (score difference)
    [["athlete_x"], ["athlete_y"]]  # Continuous game (time difference)
]
results = [
    [1, 0],      # Ordinal result
    [5, 3],      # Discrete scores
    [10.5, 11.2] # Continuous scores (seconds)
]
obs = ["Ordinal", "Discrete", "Continuous"]  # One per game

h = ttt.History(composition, results, obs=obs)
h.convergence()
```

### API Improvements
- Added `obs` parameter to `Game` and `History` classes
  - Accepts: `"Ordinal"`, `"Continuous"`, or `"Discrete"`
  - Per-game observation model specification supported in `History`
- Enhanced `Game` class initialization with clearer parameter documentation
- Improved `History` class with better batch handling

### Incremental History Updates - NEW FEATURE
- **`add_history()` method**: Add new games to an existing History
  - Allows incremental updates without recreating the entire history
  - Maintains all previous skill estimates and convergence
  - Supports adding new players dynamically
  - Perfect for real-time applications and streaming data

**Example Usage:**
```python
# Create initial history
composition = [[["alice"], ["bob"]], [["bob"], ["charlie"]]]
results = [[1, 0], [1, 0]]
h = ttt.History(composition, results, times=[1, 2])
h.convergence()

# Later, add new games
new_composition = [[["alice"], ["charlie"]], [["bob"], ["alice"]]]
new_results = [[1, 0], [0, 1]]
h.add_history(new_composition, new_results, times=[3, 4])
h.convergence()  # Re-converge with new data

# All learning curves now include the new games
lc = h.learning_curves()
```

### Code Quality
- **Complete English docstrings** for all public APIs:
  - Module-level documentation
  - All classes (Gaussian, Player, Game, History, Skill, GameType)
  - All public methods and functions
  - Comprehensive parameter descriptions and examples
- **PEP 8 compliance**: Code formatting improvements
  - Consistent spacing around operators
  - Proper line lengths and indentation
  - Improved variable naming (English translations)
- Cleaner code structure with better separation of concerns

### Algorithm Enhancements
- `fixed_point_approx()`: New function for Gaussian approximation of Poisson observations
- Improved numerical stability in likelihood computations
- Better convergence tracking with `iteration()` method
- Enhanced evidence computation for all observation models

### Documentation
- Comprehensive test suite documentation (`runtest.py`)
- Detailed docstrings with usage examples
- Clear parameter descriptions and return types
- Better inline comments throughout codebase

### Performance
- Optimized message passing in `likelihood_convergence()`
- More efficient batch processing in `History`
- Improved memory usage in skill tracking

### Bug Fixes
- More robust error handling in input validation
- Fixed edge cases in draw probability computation
- Improved handling of extreme skill values

### Testing
- Expanded test coverage for new observation models
- Added tests for discrete score outcomes
- Improved test documentation with descriptive docstrings
- All tests passing with new implementation

## Breaking Changes

### Convergence Evaluation Changes
- **Convergence now evaluates likelihood changes instead of parameter changes**
  - The verbose output during `convergence()` will show different step values
  - The final skill estimates remain the same
  - Impact: If you were monitoring step values in verbose mode, the numbers will be different (but results are equivalent)

### Time Parameter Behavior
- **Behavior change when `times` parameter is not provided to `History`**
  - Previous version: May have had different default time handling
  - Current version: Uses sequential indices as time points when not specified
  - **Recommendation**: Always explicitly pass the `times` parameter to `History` for consistent behavior
  - This ensures deterministic results across versions

### Other API Changes

#### New Methods
- `History.add_history()`: Add new games to existing history incrementally
- `History.iteration()`: Improved convergence tracking method

#### New Parameters
- `Game.__init__(obs=...)`: Observation model parameter (`"Ordinal"`, `"Continuous"`, or `"Discrete"`)
- `History.__init__(obs=...)`: Observation model(s) for games (single model or per-game list)
- `History.add_history(obs=...)`: Observation model(s) for new games being added
- `History.learning_curves(who=..., online=...)`: 
  - `who`: Filter learning curves by player names (default: None returns all players)
  - `online`: Switch between batch posteriors (default: False) and online estimates (True)

#### Modified Behavior
- `History.convergence()`: Now tracks likelihood changes instead of parameter changes
  - Return values unchanged
  - Verbose output format changed
- `History.__init__(times=...)`: Behavior changed when `times` not provided
  - Now uses sequential indices as default
  - Strongly recommended to always pass explicit `times` parameter

#### Deprecated (but still functional)
- None - all previous API calls remain compatible

#### Removed
- None - backward compatibility maintained

## References
- Guo, S., Zoeter, O., & Archambeau, C. (2012). "Score-based Bayesian skill learning"

---

# v1.1.0

- We enable the weight procedure https://github.com/glandfried/TrueSkillThroughTime.py/pull/6

# v1.0.0

- We no longer use the numba package

# v0.0.3

- Fixed multiplayer evidence
