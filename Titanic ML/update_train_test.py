import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
pd.options.mode.chained_assignment = None

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


train_test_data = [train, test] 
train_test_data = train_test_data.copy()
# Feature Extraction


for dataset in train_test_data:
	dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.') 
	# extracting titles from Name column.
	# NAME FEATURE

for dataset in train_test_data:
    
	#replace some less common titles with the name "Other".
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Other')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

for dataset in train_test_data:
    #mapping conveting into numeric
    dataset['Title'] = dataset['Title'].map({"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Other": 5})
    dataset['Title'] = dataset['Title'].fillna(0)
    
    
    #SEX FEATURE
for dataset in train_test_data:
    #represent 0 as female and 1 as male
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)


    #EMBARKED FEATURE
for dataset in train_test_data:
    # replace "nan" values with "S" because that have maximum value of number of passengers
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
for dataset in train_test_data:
    #converting to number
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)


    #AGE FEATURE
for dataset in train_test_data:
    # fill the NULL values of Age with a random number between (mean_age - std_age) and (mean_age + std_age)
    age_avg = dataset['Age'].mean()
    age_std = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()
    
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)

    #devide into agebands  
train['AgeBand'] = pd.cut(train['Age'], 5)

for dataset in train_test_data:
    #map Age according to AgeBand
	dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
	dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
	dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
	dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
	dataset.loc[ dataset['Age'] > 64, 'Age'] = 4



    #FARE FEATURE
for dataset in train_test_data:
    # replace missing Fare values with the median of Fare
    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())

train['FareBand'] = pd.qcut(train['Fare'], 4)


for dataset in train_test_data:
    # mapping according to FareBand
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)


    #SIBSP & PARCH FEATURE
for dataset in train_test_data:
    # combining SibSp & Parch feature, we create a new feature named FamilySize
    dataset['FamilySize'] = dataset['SibSp'] +  dataset['Parch'] + 1
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

# Feature Selection
features_drop = ['Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'FamilySize']
train = train.drop(features_drop, axis=1)
test = test.drop(features_drop, axis=1)
train = train.drop(['PassengerId', 'FareBand', 'AgeBand'], axis=1)

# print(train.head())
# print(train.shape)
# print(test.head())
# print(test.shape)
output_train = pd.DataFrame({'Survived': train.Survived, 'Pclass': train.Pclass, 'Sex': train.Sex, 'Age': train.Age, 'Fare': train.Fare, 'Embarked': train.Embarked,'Title': train.Title, 'IsAlone': train.IsAlone})
output_train.to_csv('train_updated.csv', index=False)


output_test = pd.DataFrame({'PassengerId': test.PassengerId, 'Pclass': test.Pclass, 'Sex': test.Sex, 'Age': test.Age, 'Fare': test.Fare, 'Embarked': test.Embarked,'Title': test.Title, 'IsAlone': test.IsAlone})
output_test.to_csv('train_test.csv', index=False)

# X_train = train.drop('Survived', axis=1)
# Y_train = train['Survived']
# X_test = test.drop("PassengerId", axis=1).copy()

# # print(type(X_train))
# # print(Y_train)
# # print(X_test)
# # print(X_train.values)

# X = X_train.values
# Y = Y_train.values
# X_test = X_test.values

# # print(X_test)

# m = len(X_train)
# n = len(X_test)
# # print(X_train)
# def sigma(z):
# 	return 1/(1 + np.exp(-z))
# X = np.append(np.ones((m, 1)), X_train, axis=1)
# X_test = np.append(np.ones((n, 1)), X_test, axis=1)
# Y = Y_train
# # print(Y)
# theta = np.array([0., 0., 0., 0., 0., 0., 0., 0.])
# lr = 0.01

# dw_s=0.
# beta=0.999
# for _ in range(1000000):
# 	Z = X.dot(theta)
# 	a = sigma(Z)
# 	dz = a - Y
# 	dw = X.T.dot(dz)/m
# 	dw_s = dw_s*beta + (1-beta)*dw
# 	theta -= lr*dw_s

# print(theta)

# def predict(x):
# 	p = sigma(x.dot(theta))
# 	if p > 0.5:
# 		return 1;
# 	return 0;

# predictions = [ predict(x) for x in X_test]
# print(len(predictions))

# # aa = -theta[1]/theta[2]
# # bb = -theta[0]/theta[2]
# # def g(x):
# # 	return aa*x + bb

# # red = np.where(Y==1)
# # blue = np.where(Y==0)
# # plt.plot(X1[red], X2[red], 'ro')
# # plt.plot(X1[blue], X2[blue], 'bo')

# # _x = [-100, 1100]
# # _y = [g(i) for i in _x]
# # plt.plot(_x, _y)

# # plt.show()

