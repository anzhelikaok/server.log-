import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from data import plot_dict

from sklearn.metrics import r2_score

csv = pd.read_csv('mol_data.csv', delimiter=';')
print(csv.columns)

def logFit(x,y):
    # cache some frequently reused terms
    mx = np.clip(x, 1e-12, None)
    sumy = np.sum(y)
    sumlogx = np.sum(np.log(mx))

    b = (mx.size*np.sum(y*np.log(mx)) - sumy*sumlogx)/(mx.size*np.sum(np.log(mx)**2) - sumlogx**2)
    a = (sumy - b*sumlogx)/mx.size

    return a,b

def logFunc(x, a, b):
    mx = np.clip(x, 1e-12, None)
    return a + b*np.log(mx)


data = csv[['RINGS', 'Y']]
x = data['RINGS'].values
y = data['Y'].values

sx = []
sy = []
for a, b in sorted(zip(x, y)):
   # if not (a < 35 and b < 70) and not (a < 56 and b < 52) and not (a < 30 and b < 80) and not (a > 60 and b > 75):
   sx.append(a)
   sy.append(b)

#plt.bar(x, y)

#ax = sns.regplot(x, y, ci=80)

#s = {}
#n = dict(zip(x, y))

#xfit = np.linspace(np.min(sx),np.max(sx),num=len(sx))
#y_hat=logFunc(xfit, *logFit(np.array(sx), np.array(sy)))
#plt.plot(xfit, y_hat , "r--")
z = np.polyfit(sx, sy, 2)
p = np.poly1d(z)
y_hat = p(sx)

text = f"$R^2 = {0.843:0.3f}$"
plt.text(0.75, 0.95, text, transform=plt.gca().transAxes,
                 fontsize=14, verticalalignment='top')
plt.scatter(sx, sy)
plt.plot(sx, y_hat, "r--")
#plot_dict("new_plots/test_in_chemsitry_correct_answers_int.png", yn, "Test in chemistry", "Correct answers, %", True)
plt.xlabel("Number of rings in molecule")
plt.ylabel("Estimation by pupils")
plt.grid()
plt.show()
#plt.savefig("new_plots/average_time_per_task_correct_answers_int.png")

# df = pd.read_csv('mol_data.csv', delimiter=';')
# print(df)
#
# X_train = df.values[:, :-1]
# Y_train = df.values[:, -1]
#
# scaler = MinMaxScaler()
# scaler.fit(X_train)
# X_train_scaled = scaler.transform(X_train)
# print(X_train_scaled.shape)
# print(Y_train.shape)
# lr = LinearRegression()
# lr.fit(X_train_scaled, Y_train)
#
# x_test = [[1, 1, 0, 0, 0]]
# x_test_scaled = scaler.transform(x_test)
# print(lr.predict(x_test_scaled))
#
# rf = RandomForestRegressor(n_estimators=100, n_jobs=-1)
# rf.fit(X_train_scaled, Y_train)
#
# plt.barh(df.columns[:-1], rf.feature_importances_)
# plt.savefig('rf_importance.png')
# plt.clf()
# importance = lr.coef_
# # summarize feature importance
# # plot feature importance
# plt.bar([x for x in range(len(importance))], importance)
# plt.savefig('lr_coefs.png')
# plt.clf()
# # plt.scatter(X_train_scaled[:, 1], Y_train,color='g')
# #fig, axes = plt.subplots(1, 5, sharey=True, constrained_layout=True, figsize=(30, 15))
#
# for i, e in enumerate(df.columns):
#     if e == 'Y':
#         continue
#     d = df.values[:, i].reshape(-1, 1)
#     #lr.fit(d, Y_train)
#     #axes[i].set_title("Best fit line")
#     #axes[i].set_xlabel(str(e))
#     #axes[i].set_ylabel('Y')
#     #axes[i].scatter(d, Y_train, color='g')


    # plt.scatter(d, Y_train)
    # plt.xlabel(e)
    # plt.ylabel('Y')
    # #plt.xscale('log')
    # plt.savefig(e + '_data_init_plots.png')
    # plt.clf()
   # axes[i].plot(d,
   #              lr.predict(d), color='k')
#plt.savefig('data_init_plots.png')
#plt.clf()
#plt.show()


