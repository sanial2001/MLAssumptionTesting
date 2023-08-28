import testing
import pandas as pd
from sklearn.linear_model import LinearRegression


def test_assumption_linear_regression(df):
    ats_object = testing.AssumptionTestingStats()
    X, y = df.iloc[:, :-1], df.iloc[:, 3]
    model = LinearRegression()
    model.fit(X, y)
    linear = ats_object.linear_test(model, X, y)
    residual_normality = ats_object.normal_residual_test(model, X, y)
    error_autocorrelation = ats_object.autocorrelation_assumption(model, X, y)
    homodescascity = ats_object.homoscedasticity_assumption(X)
    multicollinearity = ats_object.calc_vif(X)
    # print(linear, residual_normality, error_autocorrelation, homodescascity, multicollinearity)
    val = linear and residual_normality and error_autocorrelation and homodescascity and multicollinearity
    return val
