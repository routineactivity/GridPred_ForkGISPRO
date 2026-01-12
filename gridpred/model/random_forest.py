from sklearn.ensemble import RandomForestRegressor
from gridpred.model.base import GridPredPredictor


class RandomForestGridPred(GridPredPredictor):
    """
    Random Forest implementation of GridPredPredictor.
    """

    def build_model(self):
        return RandomForestRegressor(**self.model_params)
