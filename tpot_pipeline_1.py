import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LassoLarsCV, RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.tree import DecisionTreeRegressor
from tpot.builtins import StackingEstimator

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=None)

# Average CV score on the training set was:-7271.420093652521
exported_pipeline = make_pipeline(
    make_union(
        make_pipeline(
            RBFSampler(gamma=0.4),
            StackingEstimator(estimator=ExtraTreesRegressor(bootstrap=False, max_features=0.4, min_samples_leaf=16, min_samples_split=11, n_estimators=100)),
            StackingEstimator(estimator=LassoLarsCV(normalize=True)),
            MinMaxScaler(),
            RobustScaler()
        ),
        RBFSampler(gamma=0.0)
    ),
    StackingEstimator(estimator=DecisionTreeRegressor(max_depth=9, min_samples_leaf=11, min_samples_split=8)),
    RidgeCV()
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
