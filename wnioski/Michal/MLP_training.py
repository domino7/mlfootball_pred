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

# Load data
all_vectors_raw = pandas.read_csv("/home/micc/Pulpit/mlfootball_pred/learning_vectors/v05/version5-complete.csv")
important_features = ["H_age","A_age","H_TMV","A_TMV","FTHG","FTAG","HS","AS","HST","AST","H_Form03","A_Form03","H_Form05","A_Form05","H_MeanShots03","A_MeanShots03","H_MeanShots05","A_MeanShots05","H_MeanShotsOnTarget03","A_MeanShotsOnTarget03","H_MeanShotsOnTarget05","A_MeanShotsOnTarget05","H_MeanFullTimeGoals03","A_MeanFullTimeGoals03","H_MeanFullTimeGoals05","A_MeanFullTimeGoals05","H_WeightedMeanShots03","A_WeightedMeanShots03","H_WeightedMeanShots05","A_WeightedMeanShots05","H_WeightedMeanShotsOnTarget03","A_WeightedMeanShotsOnTarget03","H_WeightedMeanShotsOnTarget05","A_WeightedMeanShotsOnTarget05","H_WeightedMeanFullTimeGoals03","A_WeightedMeanFullTimeGoals03","H_WeightedMeanFullTimeGoals05",'A_WeightedMeanFullTimeGoals05']
results = ["Result"]
all_vectors_raw = all_vectors_raw.fillna(method='ffill')
all_vectors_raw[important_features] = all_vectors_raw[important_features].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
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


# In[106]:
X = all_vectors_raw[selected_features]
Y = all_vectors_raw['Result']
X = X.values
Y = Y.values


print "MLP Classifier best"
rsearch = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=1000, verbose=100)
rsearch.fit(X, Y)
#print(rsearch)
print str(rsearch.best_score_) + "%\t-\t" + "MLP" + "=" + str(rsearch.best_estimator_)
model = rsearch.best_estimator_
cv_result = model_selection.cross_val_score(model, X, Y, cv=5, scoring=scoring)
msg = "%s: %f (%f)" % ("MLP", cv_result.mean(), cv_result.std())
print msg
print ""

