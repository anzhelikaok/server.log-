import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import r2_score

class MoleculesComplexPredictor:

    def train(self, mol_data_file):

        df = pd.read_csv(mol_data_file, delimiter=';')
        data = df[['N', 'FUNC_N', 'RANDIC','WEINER','Y']]
        X_train = data[['N', 'FUNC_N','RANDIC','WEINER']].values
        Y_train = data['Y'].values
        self.sc = StandardScaler()
        self.sc.fit(X_train)
        self.scaler = MinMaxScaler()
        self.scaler.fit(X_train)
        X_train_scaled = self.sc.transform(X_train)

        self.avg_y = np.average(Y_train)

        self.lr = RandomForestRegressor()
        self.lr.fit(X_train_scaled, Y_train)

        self.rf = RandomForestRegressor(n_estimators=1000, n_jobs=-1)
        self.rf.fit(X_train_scaled, Y_train)

        print(r2_score(self.rf.predict(X_train_scaled), Y_train))
        return df.columns[:-1], self.rf.feature_importances_,0# self.lr.coef_

    def predict_csv(self, input):
        df = pd.read_csv(input, delimiter=';')
        data = df[['N', 'FUNC_N','RANDIC', 'WEINER', 'Y']]
        X_test = data[['N', 'FUNC_N','RANDIC', 'WEINER']].values
        Y_test = data['Y'].values

        x_test_scaled = self.scaler.transform(X_test)
        Y_predicted = self.lr.predict(x_test_scaled)

        df1 = pd.DataFrame({'Actual': Y_test, 'Predicted': self.avg_y})
        print(df1)

        print("Score =", self.lr.score(x_test_scaled, Y_test))

        print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, np.full(len(Y_test), self.avg_y)))
        print('Mean Squared Error:', metrics.mean_squared_error(Y_test, np.full(len(Y_test), self.avg_y)))
        print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, np.full(len(Y_test), self.avg_y))))

    def predict(self, input):
        x_test_scaled = self.scaler.transform(input)
        return self.lr.predict(x_test_scaled)

mcp = MoleculesComplexPredictor()
dc, fi, lrc = mcp.train("mol_data.csv")
mcp.predict_csv("test.csv")
