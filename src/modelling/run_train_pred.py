#!/usr/bin/env python3
"""Small CLI runner for the modelling pipeline.

This module keeps the `main` function small and delegates the heavy
lifting to `pipeline.run_pipeline` so the pipeline code is easy to
import and test.
"""

from src.modelling.pipeline import run_pipeline
from src.modelling.predictions import print_predictions
from src.modelling.config import DATA_CONFIG


def main():
    test_data, predictions = run_pipeline()

    print("\n" + "=" * 80)
    print(f"Predictions for Round {DATA_CONFIG['predict_round'][1]}, "
          f"Season {DATA_CONFIG['predict_round'][0]}")
    print("=" * 80)
    print("\nFixture:")
    print(test_data.select("Home.Team", "Away.Team"))
    
    print_predictions(test_data, predictions)


if __name__ == "__main__":
    main()
