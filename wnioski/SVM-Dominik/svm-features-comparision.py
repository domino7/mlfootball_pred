import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import numpy as np

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



def svm_predictors_compariser(current_predictors):
    for predictor in current_predictors:
        results = []
        for i in range(50):
            train, test = train_test_split(match, test_size=0.3)
            X_train = pd.DataFrame(train[predictor])
            y_train = train['Result']
            X_test = pd.DataFrame(test[predictor])
            y_test = test['Result']

            scaler = StandardScaler().fit(X_train)
            X_train_scaled = scaler.transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            clf = SVC(C=1.0, kernel='linear')
            clf.fit(X_train_scaled, y_train)

            res = clf.score(X_test_scaled, y_test)
            results.append(res)

        results = np.asarray(results)
        print("%s (mean, std): (%0.3f, %0.3f)" % (predictor, results.mean(), results.std()))


# svm_predictors_compariser(predictors2)

base_components = [p[2:] for p in predictors2 if p.startswith("H_")]
predictors3_diff = ['diff_%s' % p for p in base_components]
for comp in base_components:
    match['diff_%s' % (comp)] = match['H_%s' % (comp)] - match['A_%s' % (comp)]

svm_predictors_compariser(predictors3_diff)
