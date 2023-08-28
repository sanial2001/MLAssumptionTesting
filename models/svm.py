import testing
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


def test_assumption_svm(df):
    ats_object = testing.AssumptionTestingStats()
    X, y = df.iloc[:, :-1], df.iloc[:, 3]
    sc_X = StandardScaler()
    sc_y = StandardScaler()
    features = sc_X.fit_transform(X)
    label = sc_y.fit_transform(y.values.reshape(-1, 1))
    model = SVR(kernel='linear')
    model.fit(features, label)
    linear = ats_object.linear_test(model,X, y)
    multicollinearity = ats_object.calc_vif(X)
    #print(linear, multicollinearity)
    return not linear and multicollinearity