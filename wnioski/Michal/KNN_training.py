
# coding: utf-8

# In[100]:


import unidecode
from datetime import datetime
from scipy.stats import uniform as sp_rand
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
model = KNeighborsClassifier()
results = []
for feature in important_features_paired:
    X = all_vectors_raw[feature]
    Y = all_vectors_raw['Result']
    X = X.values
    Y = Y.values
    cv_result = model_selection.cross_val_score(model, X, Y, cv=5, scoring=scoring)
    results.append([' and '.join(feature), cv_result.mean(), cv_result.std()])
#         print "---" + ' and '.join(feature) + "---"
#         msg = "%s: %f (%f)" % ("KNN", cv_result.mean(), cv_result.std())
#         print msg(
#         print ""
for result in sorted(results, key=itemgetter(1), reverse=True):
    print "---%s--- \n mean: %s \t std: (%s) \n" % (result[0], result[1], result[2])


# In[106]:
X = all_vectors_raw[feature]
Y = all_vectors_raw['Result']
X = X.values
Y = Y.values
seed = 7
models =  []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
models.append(('Multinomial NB', sklearn.naive_bayes.MultinomialNB()))
models.append(('Bernoulli NB', sklearn.naive_bayes.BernoulliNB()))
models.append(('Perceptron', sklearn.linear_model.Perceptron()))
models.append(('SGD', sklearn.linear_model.SGDClassifier()))
models.append(('PassiveAggresive', sklearn.linear_model.PassiveAggressiveClassifier()))
models.append(('MLP Classifier', sklearn.neural_network.MLPClassifier()))
results = []
names = []
scoring = 'accuracy'
for name, model in models:
    kfold = model_selection.TimeSeriesSplit(n_splits=5)
    cv_result = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
    results.append(cv_result)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_result.mean(), cv_result.std())
    print msg



print "Randomized search for best parameters"
selected_features = ["H_age","A_age","H_TMV","A_TMV","H_MeanShotsOnTarget03","A_MeanShotsOnTarget03","H_MeanShotsOnTarget05","A_MeanShotsOnTarget05","H_MeanFullTimeGoals03","A_MeanFullTimeGoals03","H_MeanFullTimeGoals05","A_MeanFullTimeGoals05","H_WeightedMeanShots05","A_WeightedMeanShots05","H_WeightedMeanShotsOnTarget05","A_WeightedMeanShotsOnTarget05","H_WeightedMeanFullTimeGoals05",'A_WeightedMeanFullTimeGoals05']
param_grid = {'weights': ['uniform', 'distance'], 'algorithm' : ['auto', 'ball_tree', 'kd_tree', 'brute'], 'p' : sp_randint(2, 20), 'n_neighbors': sp_randint(1, 200), 'metric' : ["euclidean", "manhattan", "chebyshev"]}
model = KNeighborsClassifier()
# X = all_vectors_raw[selected_features]
# Y = all_vectors_raw['Result']
# X = X.values
# Y = Y.values
# rsearch = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=100)
# rsearch.fit(X, Y)
# #print(rsearch)
# print str(rsearch.best_score_) + "%\t-\t" + float_param + "=" + str(rsearch.best_estimator_)

# print "ALL DATA - FITTING"
# X = all_vectors_raw[selected_features]
# Y = all_vectors_raw['Result']
# X = X.values
# Y = Y.values
# rsearch = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=1000)
# rsearch.fit(X, Y)
# #print(rsearch)
# print str(rsearch.best_score_) + "%\t-\t" + "best" + "=" + str(rsearch.best_estimator_)
# model = rsearch.best_estimator_
# cv_result = model_selection.cross_val_score(model, X, Y, cv=5, scoring=scoring)
# msg = "%s: %f (%f)" % ("BEST", cv_result.mean(), cv_result.std())
# print ""

model = KNeighborsClassifier(algorithm='ball_tree', leaf_size=30, metric='manhattan',
                             metric_params=None, n_jobs=1, n_neighbors=194, p=8,
                             weights='uniform')

print "SEASON BY SEASON USING BEST MODEL"
for key, value in premiere_seasons.iteritems():
    X = value[selected_features]
    Y = value['Result']
    X = X.values
    Y = Y.values
    print "---" + key + "---"
    kfold = model_selection.TimeSeriesSplit(n_splits=5)
    cv_result = model_selection.cross_val_score(model, X, Y, cv=10, scoring=scoring)
    msg = "%s: %f (%f)" % (key, cv_result.mean(), cv_result.std())
    print msg
    print ""

print "Laliga"
for key, value in laliga_seasons.iteritems():
    X = value[selected_features]
    Y = value['Result']
    X = X.values
    Y = Y.values
    print "---" + key + "---"
    kfold = model_selection.TimeSeriesSplit(n_splits=5)
    cv_result = model_selection.cross_val_score(model, X, Y, cv=10, scoring=scoring)
    msg = "%s: %f (%f)" % (key, cv_result.mean(), cv_result.std())
    print msg
    print ""


# In[ ]:
# Compare 
print "LOOKING FOR DIFFERENT PARAMTERES"
print "Premiere league"
for key, value in premiere_seasons.iteritems():
    X = value[selected_features]
    Y = value['Result']
    X = X.values
    Y = Y.values
    print "---" + key + "---"
    rsearch = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=1000)
    rsearch.fit(X, Y)
    #print(rsearch)
    print str(rsearch.best_score_) + "%\t-\t" + key + "=" + str(rsearch.best_estimator_)
    model = rsearch.best_estimator_
    cv_result = model_selection.cross_val_score(model, X, Y, cv=5, scoring=scoring)
    msg = "%s: %f (%f)" % (key, cv_result.mean(), cv_result.std())
    print msg
    print ""

print "Laliga"
for key, value in laliga_seasons.iteritems():
    X = value[selected_features]
    Y = value['Result']
    X = X.values
    Y = Y.values
    print "---" + key + "---"
    rsearch = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=1000)
    rsearch.fit(X, Y)
    #print(rsearch)
    print str(rsearch.best_score_) + "%\t-\t" + key + "=" + str(rsearch.best_estimator_)
    model = rsearch.best_estimator_
    cv_result = model_selection.cross_val_score(model, X, Y, cv=5, scoring=scoring)
    msg = "%s: %f (%f)" % (key, cv_result.mean(), cv_result.std())
    print msg
    print ""

