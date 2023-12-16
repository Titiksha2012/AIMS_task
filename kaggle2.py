# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from scipy.stats import yeojohnson, zscore

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
        from scipy import stats


# Read the training data
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

# Feature Engineering
# Create a new feature 'FamilySize' by combining 'SibSp' and 'Parch'
train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch']

# Impute missing values in 'Age' using KNN imputation
imputer = KNNImputer(n_neighbors=5)
train_data[['Age']] = imputer.fit_transform(train_data[['Age']])

# Create a new feature 'IsAlone'
train_data['IsAlone'] = (train_data['FamilySize'] == 0).astype(int)

# Apply Box-Cox transformation to 'Fare'
train_data['Fare'], fare_lambda = boxcox(train_data['Fare'] + 1e-6)

# Create a correlation matrix
correlation_matrix = train_data[['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize', 'IsAlone']].corr()

# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

# Identify and remove outliers using the Z-score method
z_scores = np.abs(zscore(train_data[['Fare', 'Age', 'FamilySize']]))
outlier_indices = np.where(z_scores > 3)[0]
train_data_cleaned = train_data.drop(outlier_indices)

# Display the removed outliers
print("Number of outliers removed:", len(outlier_indices))
print("Outlier indices:", outlier_indices)
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
test_data.head()
women = train_data.loc[train_data.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)

print("% of women who survived:", rate_women)
men = train_data.loc[train_data.Sex == 'male']["Survived"]
rate_men = sum(men)/len(men)

print("% of men who survived:", rate_men)
from sklearn.ensemble import RandomForestClassifier

y = train_data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")
