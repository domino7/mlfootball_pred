import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import numpy as np
import itertools

match = pd.read_csv('learning_vectors/v05/version5-complete.csv', sep=',')



predictors = ['H_age','A_age','H_TMV','A_TMV',
              'H_Form03','A_Form03','H_Form05','A_Form05','H_MeanShots03','A_MeanShots03','H_MeanShots05','A_MeanShots05',
              'H_MeanShotsOnTarget03','A_MeanShotsOnTarget03','H_Better_MeanShotsOnTarget05','A_Better_MeanShotsOnTarget05']

predictors2 = [
    'H_Pass','H_Shoot','H_Pressure','H_chPass','H_chCross','H_dAggr','H_dWidth','A_Speed','A_Pass','A_Shoot','A_Pressure','A_chPass','A_chCross','A_dAggr','A_dWidth',
    'H_age','A_age','H_TMV','A_TMV',
    'H_Form03','A_Form03','H_Form05','A_Form05',
    'H_MeanShots03','A_MeanShots03','H_MeanShots05','A_MeanShots05',
    'H_MeanShotsOnTarget03','A_MeanShotsOnTarget03','H_MeanShotsOnTarget05','A_MeanShotsOnTarget05',
    'H_MeanFullTimeGoals03','A_MeanFullTimeGoals03','H_MeanFullTimeGoals05','A_MeanFullTimeGoals05',
    'H_WeightedMeanShots03','A_WeightedMeanShots03','H_WeightedMeanShots05','A_WeightedMeanShots05',
    'H_WeightedMeanShotsOnTarget03','A_WeightedMeanShotsOnTarget03','H_WeightedMeanShotsOnTarget05','A_WeightedMeanShotsOnTarget05',
    'H_WeightedMeanFullTimeGoals03','A_WeightedMeanFullTimeGoals03','H_WeightedMeanFullTimeGoals05','A_WeightedMeanFullTimeGoals05'
]


current_predictors = predictors2

seasons = match.Season.unique()
league_ids = match.League_id.unique()


def train_and_test_each_league_separately():
    """
    fits model and test data for each competition separately
    :return: nothing, prints results to std
    """
    for s,l in itertools.product(seasons, league_ids):
        curr_matches = match.loc[(match['Season'] == s) & (match['League_id'] == l)]

        results = []

        for i in range(50):
            train, test = train_test_split(curr_matches, test_size=0.3)
            X_train = pd.DataFrame(train[current_predictors])
            y_train = train['Result']
            X_test = pd.DataFrame(test[current_predictors])
            y_test = test['Result']

            scaler = StandardScaler().fit(X_train)
            X_train_scaled = scaler.transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            clf = SVC(C=1.0, kernel='linear')
            clf.fit(X_train_scaled, y_train)

            res = clf.score(X_test_scaled, y_test)
            # print("res: ", res)
            results.append(res)

        results = np.asarray(results)
        print("%s %s (mean, std): (%0.3f, %0.3f)" % (s, l, results.mean(), results.std()))


def train_all_test_each_league_separately():
    """
    trains model for all training set (70% of dataset)
    computes accuracy for each competition from test set (30%) separately
    :return: nothing, prints results to std
    """
    res_dct = {}
    for s, l in itertools.product(seasons, league_ids):
        res_dct['lst_%s_%s' % (s, l)] = []

    iterations = []

    for i in range(2):
        train, test = train_test_split(match, test_size=0.3)
        X_train = pd.DataFrame(train[current_predictors])
        y_train = train['Result']

        scaler = StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)

        clf = SVC(C=1.0, kernel='linear')
        clf.fit(X_train_scaled, y_train)

        results_iter = []
        for s, l in itertools.product(seasons, league_ids):
            single_test = test.loc[(test['Season'] == s) & (test['League_id'] == l)]
            X_single_test = pd.DataFrame(single_test[current_predictors])
            y_single_test = single_test['Result']
            X_single_test_scaled = scaler.transform(X_single_test)

            res = clf.score(X_single_test_scaled, y_single_test)
            # print("res: ", res)
            results_iter.append(res)
            res_dct['lst_%s_%s' % (s, l)].append(res)

        results_iter = np.asarray(results_iter)
        # print("(mean, std): (%0.3f, %0.3f)" % (results_iter.mean(), results_iter.std()))
        iterations.append(results_iter)
    for s, l in itertools.product(seasons, league_ids):
        single_results = np.asarray(res_dct['lst_%s_%s' % (s, l)])
        print("%s %s (mean, std): (%0.3f, %0.3f)" % (s, l, single_results.mean(), single_results.std()))




# train_and_test_each_league_separately()
train_all_test_each_league_separately()