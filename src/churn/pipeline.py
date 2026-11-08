"""The model pipeline: SMOTE -> scaling -> Gradient Boosting.

imblearn's Pipeline applies SMOTE only while fitting, so validation folds,
test data and real customers are never oversampled.
"""

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler


def build_pipeline(random_state: int = 42, params: dict | None = None) -> Pipeline:
    pipe = Pipeline([
        ("smote", SMOTE(random_state=random_state)),
        ("scaler", MinMaxScaler()),
        ("model", GradientBoostingClassifier(random_state=random_state)),
    ])
    if params:
        pipe.set_params(**params)
    return pipe
