#########   ML   ###########

# df contains all features, and the categories in a columns  'Y'
# df = pd.DataFrame(data=np.array([[1,2,3,1],[5,5,5,0],[1,2,5,0],[2,1,2,0],[4,1,3,1],[4,11,2,0]]),columns = ['X1','X2','X3','Y'])

# selection of features
feats = df.columns

# train / test separation
from sklearn.cross_validation import train_test_split

train, test = train_test_split(df, test_size=0.3)

feats = ['X1', 'X2', 'X3']
xTrain = train[feats]
xTest = test[feats]

yTrain = train['Y']
yTrue = test['Y'].values.transpose()

# yTest=test['Y'].value
# yTrue=yTest.transpose()


##################    Comparaison des modeles predictifs  ###################
from utils.ClassifierMetrics import classifierMetrics

results = pd.DataFrame(columns=['precision', 'recall', 'f1-score'])

# Arbre_cart
from sklearn import tree

model = tree.DecisionTreeRegressor()
model.fit(xTrain, yTrain)
predicted = model.predict(xTest)
predicted[np.where(predicted < 0)] = 0
met = classifierMetrics(predicted, yTrue)
results.loc['CART tree'] = [met['precision'], met['recall'], met['f1_score']]

# RANDOM_FOREST
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()
model.fit(xTrain, yTrain)
predicted = model.predict(xTest)
predicted[np.where(predicted <= 0.5)] = 0
predicted[np.where(predicted > 0.5)] = 1
met = classifierMetrics(predicted, yTrue)
results.loc['Random Forest'] = [met['precision'], met['recall'], met['f1_score']]

# Extra_tree_regressor
from sklearn.ensemble import ExtraTreesRegressor

model = ExtraTreesRegressor()
model.fit(xTrain, yTrain)
predicted = model.predict(xTest)
predicted[np.where(predicted <= 0.5)] = 0
predicted[np.where(predicted > 0.5)] = 1
met = classifierMetrics(predicted, yTrue)
results.loc['Extra Tree'] = [met['precision'], met['recall'], met['f1_score']]

# Linear SVR
from sklearn.svm import LinearSVR

model = LinearSVR()
model.fit(xTrain, yTrain)
predicted = model.predict(xTest)
predicted[np.where(predicted <= 0.5)] = 0
predicted[np.where(predicted > 0.5)] = 1
met = classifierMetrics(predicted, yTrue)
results.loc['linear SVR'] = [met['precision'], met['recall'], met['f1_score']]

# BOOSTED TREE REGRESSION
from sklearn.ensemble import GradientBoostingRegressor

model = GradientBoostingRegressor()
model.fit(xTrain, yTrain)
predicted = model.predict(xTest)
predicted[np.where(predicted <= 0.5)] = 0
predicted[np.where(predicted > 0.5)] = 1
met = classifierMetrics(predicted, yTrue)
results.loc['Boosted Tree'] = [met['precision'], met['recall'], met['f1_score']]

# Neural Networks
# SVM


# show comparison of algos
print(results.sort(columns='f1-score', ascending=False))

##################   Ex : Random Forest description  ###################


feature_importance = model.feature_importances_
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
fig = plt.figure(11)
subplot(111)
barh(pos, feature_importance[sorted_idx], align='center', color='grey')
yticks(pos, xTrain.columns[sorted_idx])
xlabel('Relative importance')
title('Variable importance')
show(block=False)
