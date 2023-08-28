from sklearn.tree import DecisionTreeRegressor
import testing
from sklearn.preprocessing import StandardScaler


def test_assumption_decision_tree(df):
    ats_object = testing.AssumptionTestingStats()
    X, y = df.iloc[:, :-1], df.iloc[:, 3]
    model = DecisionTreeRegressor()
    model.fit(X, y)
    linear = ats_object.linear_test(model, X, y)
    return not linear
