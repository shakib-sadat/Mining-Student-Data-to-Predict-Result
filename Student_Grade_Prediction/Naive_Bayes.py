from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import ConfusionMatrixDisplay

dataset = pd.read_csv('student-mat-pass-or-fail.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = GaussianNB()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)

plt.figure(figsize=(10, 10))
sns.regplot(y_test, y_pred, fit_reg=True)

plt.figure(figsize=(4, 4))
sns.countplot(data=dataset, x='pass')

plt.figure(figsize=(4, 4))
sns.histplot(data=dataset)

correlation = dataset.corr()
plt.figure(figsize=(20, 16))
sns.heatmap(correlation, cmap='coolwarm')

CM_display = ConfusionMatrixDisplay(cm).plot()
