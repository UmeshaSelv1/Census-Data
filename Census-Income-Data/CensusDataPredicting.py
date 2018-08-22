import pandas
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn import cross_validation
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.externals import joblib


url = "/Users/user/.spyder-py3/incomeCountry"
names = ['age', 'sector', 'id', 'qualification', 'working', 'status', 'job', 'family', 'race', 'sex', 'one', 'two', 'three', 'countries', 'payscale']
dataset = pandas.read_csv(url, names=names)

print(dataset.shape)
print(dataset.head(20))
# Summary
print(dataset.describe())
# Class distribution
print(dataset.groupby('payscale').size())
 
def load_csv(filename):
               file = open(filename, "incomeCountry")
               lines = reader(file)
               dataset = list(lines)
               return dataset 

#Convert string to float
for column in dataset.columns:
    if dataset[column].dtype == type(object):
        le = LabelEncoder()
        dataset[column] = le.fit_transform(dataset[column])
  
# Split-out validation dataset
array = dataset.values
X = array[:,0:14]
Y = array[:,14]
 
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
 

#Test options
num_folds = 7
num_instances = len(X_train)
seed = 7
scoring = 'accuracy'
 

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
 

# Evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
    

# Compare Algorithm
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# Making Prediction
lr = LogisticRegression()
lr.fit(X_train, Y_train)
predictions = lr.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))


pipe = make_pipeline(StandardScaler(), LogisticRegression())
pipe.fit(X_train, Y_train)
joblib.dump(pipe, 'incomeCountry.pkl')
 
pipe = joblib.load('incomeCountry.pkl')
pr = pandas.read_csv("/Users/user/.spyder-py3/incomeCountryTest")
 

#Convert string to float
for column in pr.columns:
    if pr[column].dtype == type(object):
        le = LabelEncoder()
        pr[column] = le.fit_transform(pr[column])
 

pred = pandas.Series(pipe.predict(pr))

for x in pred:
    if (x == 1):
        print(">50K")
    else:
        print("<=50K")    
    

# print(str(pred))

   





