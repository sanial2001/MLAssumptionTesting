import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from scipy.stats import normaltest
import statsmodels.stats.api as sms
import statsmodels.regression.linear_model as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from statsmodels.formula.api import ols
from statsmodels.stats.stattools import durbin_watson
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.stats.diagnostic import normal_ad
from scipy.stats import levene
import pingouin as pg
from scipy.stats import pearsonr


class AssumptionTestingStats:
    # Linearity
    def linear_test(self, model, X, y):
        # calculate the predicted values
        y_pred = model.predict(X)
        # calculate the correlation coefficient
        r, p_value = pearsonr(y, y_pred)
        #print(r, p_value)
        if p_value > 0.05 or r == 1 or r == -1:
            #print("not linear")
            return False
        else:
            #print("linear")
            return True

    # Normality of residuals
    def calculate_residuals(self, model, features, label):
        """
        Creates predictions on the features with the model and calculates residuals
        """
        predictions = model.predict(features)
        df_results = pd.DataFrame({'Actual': label, 'Predicted': predictions})
        df_results['Residuals'] = abs(df_results['Actual']) - abs(df_results['Predicted'])

        return df_results

    def normal_residual_test(self, model, X, y):
        df_results = self.calculate_residuals(model, X, y)
        p_value = normal_ad(df_results['Residuals'])[1]
        if p_value < 0.05:
            # print('Residuals are not normally distributed')
            return False
        else:
            # print('Residuals are normally distributed')
            return True

    # Multicollinearity among variables
    def calc_vif(self, X):
        # Calculating VIF
        vif = pd.DataFrame()
        vif["variables"] = X.columns
        vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        return all(val < 100 for val in vif["VIF"].values)

    # Autocorrelation of error terms
    def autocorrelation_assumption(self, model, X, y):
        df_results = self.calculate_residuals(model, X, y)
        durbinWatson = durbin_watson(df_results['Residuals'])
        if durbinWatson < 1.5:
            # print('Signs of positive autocorrelation', '\n')
            return False
        elif durbinWatson > 2.5:
            # print('Signs of negative autocorrelation', '\n')
            return False
        else:
            # print('Little to no autocorrelation', '\n')
            return True

    # Variance across error terms
    def homoscedasticity_assumption(self, X):
        test = pg.homoscedasticity(X, method='levene', alpha=0.05)
        p_value = test['pval'][0]
        if p_value > 0.05:
            # print('homoscedastic data (equal-variance)')
            return True
        else:
            # print('heteroscedastic data (unequal-variance)')
            return False

    # rainbow_lin_test(y, X)
    # normal_residual_test(model, X, y)
    # calc_vif(X)
    # autocorrelation_assumption(model, X, y)
    # homoscedasticity_assumption(X)
