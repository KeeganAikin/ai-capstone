# -*- coding: utf-8 -*-

import pandas
import numpy as np
from sklearn.model_selection import train_test_split, KFold, ShuffleSplit, cross_val_score, cross_validate
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
import matplotlib.animation as animation
from matplotlib import pyplot as plt
from IPython.display import Image

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
    # size of layers
    # number of hidden layers
    # learning rate (Scale??)
    mlp = MLPRegressor(max_iter=round(param[0]*10), hidden_layer_sizes=[round(param[1]) for _ in range(round(param[2]/5))], learning_rate_init=param[3]/1000,random_state=3, early_stopping=False)
    scores = cross_validate(mlp, X, y, cv=random_splits, scoring=all_scores, return_train_score=True)
    return np.mean(scores["test_APE"])-np.mean(scores["test_R2"])#2*np.mean(scores["test_APE"])-np.mean(scores["test_R2"])

def testRF(param, X, y, random_splits):
    # number of estimators
    # max depth
    # minimum samples per leaf
    rf = RandomForestRegressor(n_estimators=round(param[0]), max_depth=round(param[1]), min_samples_leaf=round(param[2]))
    scores = cross_validate(rf, X, y, cv=random_splits, scoring=all_scores, return_train_score=True)
    return 2*np.mean(scores["test_APE"])-np.mean(scores["test_R2"])

def testSVR(param, X, y, random_splits):
    # C
    # epsilon?
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

    df = pandas.read_csv('ai-capstone/data.csv', index_col='NUM')

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

    test_X = data.loc[:,['Mine Annual Production (Million Tonne)',
        'Stripping Ratio', 'Mill Annual Production (Thousand Tonne)',
        'Reserve Mean Grade % Cu EQU.', 'LOM']]
    test_y = data["CAPEX US$ millions"]

    num_folds = 6

    # Monte Carlo cross validation
    random_splits = ShuffleSplit(n_splits=60, test_size=0.2, random_state=1)

    lr = np.array(range(20,81,2))
    layer_size = np.array(range(14,41,2))
    #epochs = np.array(range(50, 1001, 50))

    #max_depth = np.array(range(2,11))

    #min_leaf = np.array(range(1,11))

    scores = np.ones((4,31))#np.ones((20,41))
    print(scores.shape)
    print(layer_size.shape)
    print(lr.shape)
    #quit()
    import matplotlib.pyplot as plt

    for i in range(len(layer_size)):#layer_size
        #continue
        val = layer_size[i]
        if not i%2:
            print(val)
        params = np.array([np.ones(31)*40,np.ones(31)*val,np.ones(31)*5,lr]).T#np.array([np.ones(10)*val,min_leaf]).T#

        results = optimiseModel(params, testANN, X, y, random_splits)

        #print(results.shape)

        scores[i] = results.copy()

    #ANNscores2 - np.ones((31,31))
    #ANNscores - np.ones((19,31))
    #ANNscores3 - np.ones((19,31))

    #np.save("ANNscores2",scores)

    scores = np.load("ANNscores2.npy")#[5:]
    #print(scores[5:].shape)
    #quit()
    smoothed_scores = np.ones((12,29))#np.ones((29,29))

    for x in range(29):
        for y in range(12):
            smoothed_scores[y][x] = sum(scores[y+1+i][x+1+j] for i in [-1,0,1] for j in [-1,0,1])

    
    print(smoothed_scores.shape)
    print(layer_size.shape)
    print(lr.shape)

    ind = np.unravel_index(smoothed_scores.argmin(), smoothed_scores.shape)

    print(np.min(smoothed_scores))

    print(layer_size[ind[0]+1])
    print(lr[ind[1]+1])

    plt.contour(lr[1:-1], layer_size[1:-1], smoothed_scores,levels=20)

    plt.savefig("ANNSmoothedContourGraph3")


    #print(scores.shape)
    #print(layer_size.shape)
    #print(lr.shape)

    #ind = np.unravel_index(scores.argmin(), scores.shape)

    #print(np.min(scores))

    #print(layer_size[ind[0]])
    #print(lr[ind[1]])

    #plt.contour(lr, layer_size, scores,levels=20)

    #plt.savefig("ANNContourGraph2")

    quit()

    # max iterations
    # size of layers
    # number of hidden layers
    # learning rate (Scale??)
    
    
    TESTbounds = (np.array([-10, -10]),
                np.array([10, 10]))

    RFbounds = (np.array([10,  2,  1]),
                np.array([100, 15, 10]))
    
    ANNbounds = (np.array([10,  10,  4, 1]),
                 np.array([100, 100, 20, 30]))
    # bounds:
    #   upper + lower bounds of parameter values to test
    #   format -> (np array of lower bounds,
    #              np array of upper bounds)

    options = {'c1': 0.6, 'c2': 0.1, 'w': 0.9}# changed c1 0.5 -> 0.4, c2 0.3 -> 0.1
    optimizer = GlobalBestPSO(n_particles=80, dimensions=4, options=options, bounds=ANNbounds, bh_strategy="reflective")# dimensions = number of parameters to optimise
    print("begin optimisation")
    cost, pos = optimizer.optimize(optimiseModel, 50, verbose=True, modelfunc=testANN, n_processes=5, X=X, y=y, random_splits=random_splits) #change this to optimise different models (e.g. testRF, testSVR, testDT)

    print(cost)
    print("optimal parameters: ",pos)


    # --------------------- RF experimentation results ----------------------

    # R2 (10 particles, 100 iter)
    # 59   6   3
    # 57  11   3

    # R2 (10 particles, 1000 iter)
    # 42  11   3

    # R2+APE (10 particles, 500 iter)
    # 49  11   2
    # 45   9   4

    # R2+APE (20 particles, 500 iter)
    # 84   9   3
    # 61  10   3

    # R2+APE (20 particles, 100 iter)
    # 84   9   3
    # 60  12   2

    # -> increased MC crossvalidation n_splits to increase consistency

    # R2+APE (10 particles, 100 iter)
    # 60   6   4

    # R2+APE (20 particles, 200 iter)
    # 74  13   3


    # R2+2*APE
    # 100   10   3

    # -----------------------------------------------------------------------

    # -------------------- ANN experimentation results ----------------------

    # R2+2*APE (20 particles, 100 iter)
    # 128   60   2   0.0171
    # 378   54   1   0.0444

    # R2+2*APE (40 particles, 200 iter)
    # 160   26   3   0.0142
    # 218   27   3   0.0200
    # 264   41   2   0.0170

    # Added random seed for ShuffleSplit

    # R2+2*APE (40 particles, 50 iter)
    # 749   33   1   0.00271
    # 145   66   3   0.00862
    # 203   77   3   0.01007
    # 485   58   2   0.02753
    # 921   50   1   0.00242


    # R2+2*APE
    # 100   10   3

    # -----------------------------------------------------------------------


    # could use multiprocessing to speed it up, but then can't use ipynb

    # NEXT: try plotting optimisation

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

    ANNbounds = (np.array([10,  10,  4, 1]),
                 np.array([100, 100, 20, 30]))

    #needs to be 2d array
    anim = plot_contour(pos_history1, designer=Designer(limits=[(10,100),(4,20)]))

    anim.save('ANNplot6.gif', writer='imagemagick', fps=10)

    anim = plot_contour(pos_history2, designer=Designer(limits=[(10,100),(1,30)]))

    anim.save('ANNplot7.gif', writer='imagemagick', fps=10)
