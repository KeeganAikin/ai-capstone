import pandas
import numpy as np
from sklearn.model_selection import train_test_split,  ShuffleSplit, cross_validate
from sklearn.model_selection import ShuffleSplit
from sklearn import metrics

from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from pyswarms.single.global_best import GlobalBestPSO

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

from pyswarms.utils.plotters.plotters import plot_contour
from pyswarms.utils.plotters.formatters import Designer
from matplotlib import pyplot as plt

# Metrics:
#   RMSE -> Root Mean Squared Error
#   R2 -> Coefficient of Determination
#   MAE -> Mean Absolute Error
#   APE -> Absolute Error


def RMSE(model, X, y):
    return metrics.root_mean_squared_error(y,model.predict(X))

def R2(model, X, y):
    return metrics.r2_score(y,model.predict(X))

def MAE(model, X, y):
    return metrics.mean_absolute_error(y,model.predict(X))

def APE(model, X, y):
    return metrics.mean_absolute_percentage_error(y,model.predict(X))


def all_scores(model, X, y):
    return {
        "RMSE": RMSE(model, X, y),
        "R2": R2(model, X, y),
        "MAE": MAE(model, X, y),
        "APE": APE(model, X, y)
    }

def testANN(param, X, y, random_splits):
    # max iterations
    # size of hidden layer
    # learning rate
    mlp = MLPRegressor(max_iter=round(param[0]), hidden_layer_sizes=[round(param[1])], learning_rate_init=param[2]/1000,random_state=3, early_stopping=False)
    scores = cross_validate(mlp, X, y, cv=random_splits, scoring=all_scores, return_train_score=True)
    return np.mean(scores["test_APE"])-np.mean(scores["test_R2"])#2*np.mean(scores["test_APE"])-np.mean(scores["test_R2"])

def testRF(param, X, y, random_splits):
    # number of estimators
    # max depth
    # minimum samples per leaf
    rf = RandomForestRegressor(n_estimators=round(param[0]), max_depth=round(param[1]), min_samples_leaf=round(param[2]), random_state=1)
    scores = cross_validate(rf, X, y, cv=random_splits, scoring=all_scores, return_train_score=True)
    return 2*np.mean(scores["test_APE"])-np.mean(scores["test_R2"])

def testSVR(param, X, y, random_splits):
    # C
    # epsilon
    svr = SVR(C=param[0], epsilon=param[1])
    scores = cross_validate(svr, X, y, cv=random_splits, scoring=all_scores, return_train_score=True)
    return 2*np.mean(scores["test_APE"])-np.mean(scores["test_R2"])

def testDT(param, X, y, random_splits):#CART Tree
    # max depth
    # minimum samples per leaf
    dt = DecisionTreeRegressor(max_depth=round(param[0]), min_samples_leaf=round(param[1]))
    scores = cross_validate(dt, X, y, cv=random_splits, scoring=all_scores, return_train_score=True)
    return 2*np.mean(scores["test_APE"])-np.mean(scores["test_R2"])

def optimiseModel(params, modelfunc, X, y, random_splits):
    out = np.array([modelfunc(param, X, y, random_splits) for param in params])
    return out

if __name__ == "__main__":

    df = pandas.read_csv('data.csv', index_col='NUM')

    del df["NAME"]
    del df["COUNTRY"]

    data, test = train_test_split(
        df,
        test_size = 0.2,
        random_state = 5)

    print(data.shape)
    print(test.shape)

    X = data.loc[:,['Mine Annual Production (Million Tonne)',
        'Stripping Ratio', 'Mill Annual Production (Thousand Tonne)',
        'Reserve Mean Grade % Cu EQU.', 'LOM']]
    y = data["CAPEX US$ millions"]

    test_X = test.loc[:,['Mine Annual Production (Million Tonne)',
        'Stripping Ratio', 'Mill Annual Production (Thousand Tonne)',
        'Reserve Mean Grade % Cu EQU.', 'LOM']]
    test_y = test["CAPEX US$ millions"]

    # Monte Carlo cross validation
    random_splits = ShuffleSplit(n_splits=60, test_size=0.2, random_state=1)

    #lr = np.array(range(20,81,2))
    #layer_size = np.array(range(14,41,2))
    #epochs = np.array(range(50, 1001, 50))

    min_leaf = np.array(range(1,6))

    num_est = np.array(range(20,241,5))

    scores = np.ones((5,45))
    print(scores.shape)
    print(min_leaf.shape)
    print(num_est.shape)
    import matplotlib.pyplot as plt

    for i in range(len(min_leaf)):
        val = min_leaf[i]
        if not i%2:
            print(val)
        params = np.array([num_est,np.ones(11)*12,np.ones(11)*val]).T

        results = optimiseModel(params, testRF, X, y, random_splits)

        scores[i] = results.copy()

    np.save("RFscores",scores)

    ind = np.unravel_index(scores.argmin(), scores.shape)

    print(np.min(scores))

    print(min_leaf[ind[0]])
    print(num_est[ind[1]])

    plt.contour(num_est, min_leaf, scores,levels=20)

    plt.savefig("RFContourGraph")

    quit()

    RFbounds = (np.array([10,  2,  1]),
                np.array([100, 15, 10]))
    
    ANNbounds = (np.array([10,  10, 1]),
                 np.array([100, 100, 30]))
    # bounds:
    #   upper + lower bounds of parameter values to test
    #   format -> (np array of lower bounds,
    #              np array of upper bounds)

    options = {'c1': 0.6, 'c2': 0.1, 'w': 0.9}
    optimizer = GlobalBestPSO(n_particles=40, dimensions=3, options=options, bounds=ANNbounds, bh_strategy="reflective")# dimensions = number of parameters to optimise
    print("begin optimisation")
    cost, pos = optimizer.optimize(optimiseModel, 200, verbose=True, modelfunc=testANN, n_processes=5, X=X, y=y, random_splits=random_splits) #change this to optimise different models (e.g. testRF, testSVR, testDT)

    print(cost)
    print("optimal parameters: ",pos)

    #np.save("pos_history",np.array(optimizer.pos_history))

    print(len(optimizer.pos_history))
    print(optimizer.pos_history[0].shape)

    print(optimizer.pos_history[0][:5])

    pos_history1 = []
    pos_history2 = []
    for pos_list in optimizer.pos_history:
        pos_history1.append(np.delete(pos_list,[0,3],1))
        pos_history2.append(np.delete(pos_list,[1,2],1))


    print(len(pos_history1))
    print(pos_history1[0].shape)
    print(pos_history1[0][:5])
    print(pos_history1[-1][:5])

    #needs to be 2d array
    anim = plot_contour(pos_history1, designer=Designer(limits=[(10,100),(4,20)]))

    anim.save('ANNplot1.gif', writer='imagemagick', fps=10)

    anim = plot_contour(pos_history2, designer=Designer(limits=[(10,100),(1,30)]))

    anim.save('ANNplot2.gif', writer='imagemagick', fps=10)
