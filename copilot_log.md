User: hey, I want to refactor this to make it more maintainable. Can you suggest some functions to factor out?

GitHub Copilot: I'll analyze the file to suggest refactoring opportunities. 

Read [](file:///Users/benjaminwee/Documents/afl_probabilistic_model/sandbox/modelling.py)

Based on my analysis, here are the key functions I'd suggest factoring out for better maintainability:

## Suggested Refactoring

1. **`load_data()`** - Load parquet files and initial transforms
2. **`calculate_win_loss_records()`** - Calculate cumulative wins/losses for home and away teams
3. **`prepare_main_features()`** - Join ladder data with win/loss records
4. **`prepare_feature_dataframe()`** - Join results with features for both home and away teams
5. **`prepare_test_data()`** - Create test dataset from fixture and final round stats
6. **`setup_model_data()`** - Extract feature arrays and coordinates for the model
7. **`fit_bayesian_model()`** - Run the PyMC sampling
8. **`generate_predictions()`** - Calculate logistic predictions from model parameters

Would you like me to implement this refactoring? I can break the code into these logical functions with proper parameters and return values, which will make it easier to test, reuse, and modify individual pieces.

User: before you do that, can you explain how this code works

GitHub Copilot: Sure! Here's how the code flows:

## Overall Workflow

**1. Data Loading** (lines 11-15)
- Loads three datasets from parquet: ladder standings, historical results, and upcoming fixture
- Fixture also adjusts Round numbers

**2. Results Processing** (lines 17-22)
- Adds `home_win` binary column (1 if home team won, 0 otherwise)
- Filters out playoff rounds (QF, EF, SF, PF, GF) - only regular season
- Selects relevant columns

**3. Win/Loss Tracking** (lines 24-39)
- Splits results into home and away perspectives
- Calculates cumulative wins and losses for each team per season
- Shifts the cumulative stats to get "previous round" versions (prev_total_wins, prev_total_loss)
- Fills nulls with 0 for first round

**4. Feature Engineering** (lines 41-50)
- Joins ladder percentage stats with win/loss records
- Gets previous round percentages and combines everything into `main_features`

**5. Complete Features** (lines 52-87)
- Joins results with main_features twice (for home and away teams)
- Renames columns to distinguish home vs away (e.g., `home_team_prev_percentage`, `away_team_prev_percentage`)
- Calculates derived features: win percentages and point differentials (delta_wins, delta_loss)

**6. Train/Test Split** (lines 89-94)
- Trains on all data EXCEPT the prediction round
- Test data is the entire prediction season (to extract latest stats)

**7. Test Data Preparation** (lines 96-131)
- Extracts the most recent stats for each team from the prediction season
- Joins with fixture to get the upcoming matchups
- Enriches with latest team statistics

**8. Model Data** (lines 133-145)
- Extracts feature arrays split into two groups: percentages and delta wins
- Creates coordinate labels for interpretability

**9. Bayesian Model** (lines 147-162)
- Fits a logistic regression with PyMC
- Intercept `a`, coefficients for percentages `bp`, coefficients for deltas `bw`
- Uses Bernoulli likelihood with logit link function
- Samples from posterior

**10. Predictions** (lines 164-173)
- Extracts posterior means for model parameters
- Applies to test data to get log-odds
- Converts to probabilities using logistic function
- Computes log odds for both outcomes

User: Great thank you. Can you please proceed with your suggested refactoring? Create a new .py file within sandbox that contains this refactored functions as well as code that will execute these functions and produce a prediction. 

GitHub Copilot: Created [](file:///Users/benjaminwee/Documents/afl_probabilistic_model/sandbox/modelling_refactored.py)

Done! I've created modelling_refactored.py with the following improvements:

**Refactored Functions:**
- `load_data()` - Loads parquet files
- `process_results()` - Adds home_win column and filters to regular season
- `calculate_win_loss_records()` - Computes cumulative wins/losses
- `prepare_main_features()` - Joins ladder with win/loss records
- `prepare_feature_dataframe()` - Creates full feature set for both teams
- `split_train_test()` - Splits data by prediction round
- `prepare_test_data()` - Prepares fixture data for prediction
- `setup_model_data()` - Extracts numpy arrays and coordinates
- `fit_bayesian_model()` - Runs PyMC sampling
- `extract_parameters()` - Extracts posterior means
- `generate_predictions()` - Computes final predictions
- `logistic()` - Helper for logistic function

**Improvements:**
- Configuration centralized at the top
- Each function has a single responsibility
- Type hints for clarity
- Docstrings explaining what each function does
- `main()` function orchestrates the entire pipeline with progress messages
- Results printed in a readable format showing matchups and predictions

You can run it with:
```bash
python sandbox/modelling_refactored.py
```

Made changes.