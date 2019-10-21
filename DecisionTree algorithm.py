import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

#Using my_data as the Drug.csv data read by pandas
my_data = pd.read_csv("drug200.csv", delimiter=",")
print(my_data[0:5])

X = my_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
print(X[0:5])

#convert categorical variables like sex BP ... to numerical values.
le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F','M'])
X[:,1] = le_sex.transform(X[:,1])

le_BP = preprocessing.LabelEncoder()
le_BP.fit([ 'LOW', 'NORMAL', 'HIGH'])
X[:,2] = le_BP.transform(X[:,2])

le_Chol = preprocessing.LabelEncoder()
le_Chol.fit([ 'NORMAL', 'HIGH'])
X[:,3] = le_Chol.transform(X[:,3])

print( "Converted : ", X[0:5])

#Filling the target variable
y = my_data["Drug"]
print("Y :",y[0:5])

#using train/test split on the decision tree from sklearn.cross_validation
X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)

# Create a DecisionTreeClassifier.
drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)

#Training the data
drugTree.fit(X_trainset,y_trainset)

#Prediction
predTree = drugTree.predict(X_testset)
print (predTree [0:5])
print (y_testset [0:5])

#Evaluation: check the accuracy of our model
print("DecisionTrees's Accuracy using  sklearn: ", metrics.accuracy_score(y_testset, predTree))
print("DecisionTrees's Accuracy using numpy: ",acc=np.mean(y_testset==predTree))

