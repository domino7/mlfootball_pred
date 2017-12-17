from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

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

#
# train, test = train_test_split(match, test_size=0.3)
#
#
# X_train = train[predictors2]
# y_train = train['Result']
# X_test = test[predictors2]
# y_test = test['Result']

base_components = [p[2:] for p in predictors2 if p.startswith("H_")]
predictors3_diff = ['diff_%s' % p for p in base_components]
for comp in base_components:
    match['diff_%s' % (comp)] = match['H_%s' % (comp)] - match['A_%s' % (comp)]

X = match[predictors3_diff]
y = match['Result']


scaler = StandardScaler().fit(X)
X_train_scaled = scaler.transform(X)

parameter_candidates = []

parameter_candidates_linear = {
           "kernel" : ['linear'],
           "C" : [1]}


parameter_candidates_poly = {
           "kernel" : ['poly'],
           "gamma" : [1, 1e-1, 1e-2, 1e-3, 1e-4],
           "degree" : [2,3,4],
           "C" : [1, 10, 50]}

parameter_candidates_rbf = {
           "kernel" : ['rbf'],
           "gamma" : [1, 1e-1, 1e-2, 1e-3, 1e-4],
           "C" : [1, 10, 50, 100, 500]}

parameter_candidates.append(parameter_candidates_linear)
parameter_candidates.append(parameter_candidates_poly)
parameter_candidates.append(parameter_candidates_rbf)

for pc in parameter_candidates:
    print("\n\n\n PC: ", pc)
    # Create a classifier object with the classifier and parameter candidates
    clf = GridSearchCV(estimator=SVC(), param_grid=pc, n_jobs=-1,  verbose=10,  cv=3)

    # Train the classifier on data1's feature and target data
    clf.fit(X, y)
    clf.best_params_
    print("Best params:")
    print(clf.best_params_)