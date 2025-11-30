import pandas as pd
import argparse

from src.prediction import GridPred
from sklearn.ensemble import RandomForestRegressor


DO_PROJECTION = False


def main():
    # Parser args
    parser = argparse.ArgumentParser(
        description="Run the a GridPred crime prediction model."
    )

    # 1. Required positional: crime CSV path
    parser.add_argument(
        "input_file_path", type=str, help="Path to the input crime data CSV file."
    )

    # 2. Optional file inputs
    parser.add_argument(
        "--input_features_path",
        type=str,
        help="Path to the input predictive features CSV file.",
        default=None,
    )
    parser.add_argument(
        "--input_region_path",
        type=str,
        help="Path to the study region shapefile.",
        default=None,
    )

    # 3. Required column names from the crime CSV
    parser.add_argument(
        "--crime_time_variable",
        type=str,
        required=True,
        help="Column representing the temporal crime variable in the input crime CSV.",
    )

    parser.add_argument(
        "--features_names_variable",
        type=str,
        help="Column representing feature names in the predictive features CSV file.",
    )

    parser.add_argument(
        "--input_crs",
        type=str,
        default="EPSG:4326",
        help="EPSG code for the input point CSV data (e.g., 'EPSG:3508').",
    )

    parser.add_argument(
        "--grid_size",
        type=int,
        default=300,
        help="Size of the grid cell units (e.g., 300 meters).",
    )

    args = parser.parse_args()

    # ---------------------------------- #
    # Do Gridpred data prep stuff here
    # ---------------------------------- #

    

    gridpred = GridPred(
        input_crime_data=args.input_file_path,
        input_features_data=args.input_features_path,
        input_study_region=args.input_region_path,
        crime_time_variable=args.crime_time_variable,
        features_names_variable=args.features_names_variable,
        input_crs=args.input_crs
    )

    gridpred.prepare_data(args.grid_size, DO_PROJECTION)

    X = gridpred.X
    y = gridpred.y

    rf = RandomForestRegressor(n_estimators=500, criterion="poisson", random_state=42)
    rf.fit(X, y)

    # Predict
    y_pred = rf.predict(X)

    importances = pd.Series(rf.feature_importances_, index=X.columns)
    print(importances.sort_values(ascending=False))


if __name__ == "__main__":
    main()
