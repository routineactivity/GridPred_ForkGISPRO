from src.prediction import GridPred
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

INPUT_FILE_PATH = "/home/gmcirco/Documents/Projects/data/crime_2019-2023_v2.csv"
INPUT_FEATURES_PATH = "/home/gmcirco/Documents/Projects/data/features_malmo.csv"
INPUT_REGION_PATH = (
    "/home/gmcirco/Documents/Projects/data/malmo_shapefiles/DeSo_Malmö.shp"
)

gridpred = GridPred(
    input_crime_data=INPUT_FILE_PATH,
    input_features_data=INPUT_FEATURES_PATH,
    input_study_region=INPUT_REGION_PATH,
    crime_time_variable="yearvar",
    features_names_variable="type",
)

gridpred.prepare_data(300, do_projection=True)

X = gridpred.X
y = gridpred.y

rf = RandomForestRegressor(n_estimators=500, criterion="poisson", random_state=42)
rf.fit(X, y)

# Predict
y_pred = rf.predict(X)

importances = pd.Series(rf.feature_importances_, index=X.columns)
print(importances.sort_values(ascending=False))