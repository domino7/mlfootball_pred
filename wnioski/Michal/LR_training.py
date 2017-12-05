import unidecode
from datetime import datetime
from scipy.stats import uniform as sp_rand
from numpy.random import choice as rand_list
import Tkinter
import csv
import datetime
import random
import math
import operator
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.datasets import load_iris
import numpy as np
import pandas
import random
import sklearn
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from scipy.stats import mode
from operator import itemgetter
from scipy.stats import randint as sp_randint


# In[101]:
def generateRandomTuple(min_size, max_size, min_element, max_element):
    length = sp_randint(min_size, max_size)
    result
    for x in range(length):
        result.append(sp_randint(min_element, max_element))
    return tuple(result)

def convert_float(x):
  return int(round(x))

# Load data
all_vectors_raw = pandas.read_csv("/home/micc/Pulpit/mlfootball_pred/learning_vectors/v05/version5-complete.csv")
important_features = ["H_age","A_age","H_TMV","A_TMV","FTHG","FTAG","HS","AS","HST","AST","H_Form03","A_Form03","H_Form05","A_Form05","H_MeanShots03","A_MeanShots03","H_MeanShots05","A_MeanShots05","H_MeanShotsOnTarget03","A_MeanShotsOnTarget03","H_MeanShotsOnTarget05","A_MeanShotsOnTarget05","H_MeanFullTimeGoals03","A_MeanFullTimeGoals03","H_MeanFullTimeGoals05","A_MeanFullTimeGoals05","H_WeightedMeanShots03","A_WeightedMeanShots03","H_WeightedMeanShots05","A_WeightedMeanShots05","H_WeightedMeanShotsOnTarget03","A_WeightedMeanShotsOnTarget03","H_WeightedMeanShotsOnTarget05","A_WeightedMeanShotsOnTarget05","H_WeightedMeanFullTimeGoals03","A_WeightedMeanFullTimeGoals03","H_WeightedMeanFullTimeGoals05",'A_WeightedMeanFullTimeGoals05']
important_features_no_goals = ["H_age","A_age","H_TMV","A_TMV","HS","AS","HST","AST","H_Form03","A_Form03","H_Form05","A_Form05","H_MeanShots03","A_MeanShots03","H_MeanShots05","A_MeanShots05","H_MeanShotsOnTarget03","A_MeanShotsOnTarget03","H_MeanShotsOnTarget05","A_MeanShotsOnTarget05","H_MeanFullTimeGoals03","A_MeanFullTimeGoals03","H_MeanFullTimeGoals05","A_MeanFullTimeGoals05","H_WeightedMeanShots03","A_WeightedMeanShots03","H_WeightedMeanShots05","A_WeightedMeanShots05","H_WeightedMeanShotsOnTarget03","A_WeightedMeanShotsOnTarget03","H_WeightedMeanShotsOnTarget05","A_WeightedMeanShotsOnTarget05","H_WeightedMeanFullTimeGoals03","A_WeightedMeanFullTimeGoals03","H_WeightedMeanFullTimeGoals05",'A_WeightedMeanFullTimeGoals05']
results = ["Result"]
all_vectors_raw = all_vectors_raw.fillna(method='ffill')
all_vectors_raw[important_features_no_goals] = all_vectors_raw[important_features_no_goals].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
all_premiere_vectors = all_vectors_raw[all_vectors_raw["League_id"] == 1729]
all_laliga_vectors = all_vectors_raw[all_vectors_raw["League_id"] == 21518]
seasons = ["2008/2009","2009/2010","2010/2011","2011/2012","2012/2013","2013/2014", "2014/2015", "2015/2016"]
premiere_seasons = {}
laliga_seasons = {}
for season in seasons:
    premiere_seasons[season] = all_premiere_vectors[all_premiere_vectors["Season"] == season]
    laliga_seasons[season] = all_laliga_vectors[all_laliga_vectors["Season"] == season]


# In[102]:


# Check single-feature prediction for all "important features"
important_features_paired = [["H_age","A_age"],["H_TMV","A_TMV"],["FTHG","FTAG"],["HS","AS"],["HST","AST"],["H_Form03","A_Form03"],["H_Form05","A_Form05"],["H_MeanShots03","A_MeanShots03"],["H_MeanShots05","A_MeanShots05"],["H_MeanShotsOnTarget03","A_MeanShotsOnTarget03"],["H_MeanShotsOnTarget05","A_MeanShotsOnTarget05"],["H_MeanFullTimeGoals03","A_MeanFullTimeGoals03"],["H_MeanFullTimeGoals05","A_MeanFullTimeGoals05"],["H_WeightedMeanShots03","A_WeightedMeanShots03"],["H_WeightedMeanShots05","A_WeightedMeanShots05"],["H_WeightedMeanShotsOnTarget03","A_WeightedMeanShotsOnTarget03"],["H_WeightedMeanShotsOnTarget05","A_WeightedMeanShotsOnTarget05"],["H_WeightedMeanFullTimeGoals03","A_WeightedMeanFullTimeGoals03"],["H_WeightedMeanFullTimeGoals05",'A_WeightedMeanFullTimeGoals05']]
scoring = 'accuracy'
model = sklearn.neural_network.MLPClassifier()
selected_features = ["H_age","A_age","H_TMV","A_TMV","H_MeanShotsOnTarget03","A_MeanShotsOnTarget03","H_MeanShotsOnTarget05","A_MeanShotsOnTarget05","H_MeanFullTimeGoals03","A_MeanFullTimeGoals03","H_MeanFullTimeGoals05","A_MeanFullTimeGoals05","H_WeightedMeanShots05","A_WeightedMeanShots05","H_WeightedMeanShotsOnTarget05","A_WeightedMeanShotsOnTarget05","H_WeightedMeanFullTimeGoals05",'A_WeightedMeanFullTimeGoals05']
param_grid = {'hidden_layer_sizes' : np.random.random_integers(50, size=(1,20)),
              'activation' : ['identity', 'logistic', 'tanh', 'relu'],
              'solver' : ['lbfgs', 'sgd', 'adam'],
              'alpha' : sp_rand(),
              'learning_rate' : ['constant', 'invscaling', 'adaptive'],
              'max_iter' : [2000]}
selected_features = ["H_age","A_age","H_TMV","A_TMV","H_MeanShotsOnTarget03","A_MeanShotsOnTarget03","H_MeanShotsOnTarget05","A_MeanShotsOnTarget05","H_MeanFullTimeGoals03","A_MeanFullTimeGoals03","H_MeanFullTimeGoals05","A_MeanFullTimeGoals05","H_WeightedMeanShots05","A_WeightedMeanShots05","H_WeightedMeanShotsOnTarget05","A_WeightedMeanShotsOnTarget05","H_WeightedMeanFullTimeGoals05",'A_WeightedMeanFullTimeGoals05']

msk = np.random.rand(len(all_vectors_raw)) < 0.8

param_grid_lr = {'fit_intercept' : [True, False],
                 'normalize' : [True, False]}
param_grid_lasso = {'fit_intercept' : [True, False],
                    'normalize' : [True, False],
                    'alpha' : sp_rand(),
                    'tol' : sp_rand(),
                    'warm_start': [True, False],
                    'positive' : [True, False]}
param_grid_elas  = {'fit_intercept' : [True, False],
                    'normalize' : [True, False],
                    'alpha' : sp_rand(),
                    'tol' : sp_rand(),
                    'warm_start': [True, False],
                    'positive' : [True, False],
                    'l1_ratio' : sp_rand()}

models =  []
models.append(('Linear regression', LinearRegression(), param_grid_lr, 4))
models.append(('Lasso', sklearn.linear_model.Lasso(), param_grid_lasso, 1000))
models.append(('ElasticNet', sklearn.linear_model.ElasticNet(), param_grid_elas, 1000))

selected_vectors = all_vectors_raw[selected_features]
reults = all_vectors_raw[['FTHG', 'FTAG']]
match_outcome = all_vectors_raw['Result']




# In[106]:
for name, model, grid, model_iters in models:
  X_train = selected_vectors[msk]
  Y_train = reults[msk]
  match_outcome_train = match_outcome[msk]
  match_outcome_train = match_outcome_train.values
  X_train = X_train.values
  Y_train = Y_train.values

  X_test = selected_vectors[~msk]
  Y_test = reults[~msk]
  match_outcome_test = match_outcome[~msk]
  match_outcome_test = match_outcome_test.values
  X_test = X_test.values
  Y_test = Y_test.values

  rsearch = RandomizedSearchCV(estimator=model, param_distributions=grid, n_iter=model_iters)
  rsearch.fit(X_train, Y_train)

  regr = rsearch.best_estimator_
  regr.fit(X_train, Y_train)
  predicted = regr.predict(X_test)

  predicted = pandas.DataFrame(predicted)
  predicted = predicted.applymap(convert_float)

  success = 0

  for i, row in predicted.iterrows():
    if row[0] > row[1]:
      result = 0
    elif row[0] == row[1]:
      result = 1
    else:
      result = 2

    if result == match_outcome_test[i]:
      success = success + 1
    # else:
    #   print "H: " + str(row[0]) + "\tA: " + str(row[1]) + "\tExpected result: " + str(match_outcome_test[i])

  print "---" + name + "---"
  print str(round(float(success) / float(len(match_outcome_test)),4) * 100) + "%"
  print ""