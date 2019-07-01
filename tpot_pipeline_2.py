import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=None)

# Average CV score on the training set was:-9566.54313104908
exported_pipeline = make_pipeline(
    RBFSampler(gamma=0.4),
    SelectPercentile(score_func=f_regression, percentile=81),
    StackingEstimator(estimator=ElasticNetCV(l1_ratio=0.30000000000000004, tol=1e-05)),
    StackingEstimator(estimator=RidgeCV()),
    RidgeCV()
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
