# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 03:39:46 2021

@author: jerem
"""

# Import necessary modules for data preprocessing, model building, and performance evaluation.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score
import matplotlib.pyplot as plt

# Read the dataset and check if there are null values or not.
df = pd.read_csv('hero_lineups_extended_roles.csv')
print(df)
print(df.isnull().sum())
print(df.info())

# Split the dataset into 80% train data and 20% test data.
X_train, X_test, y_train, y_test = train_test_split(df.drop('Dire_1', axis=1), df['Dire_1'],
                                                    train_size=0.80, test_size=0.20, random_state=0)

# Build an SVM model, train it, and test it by predicting the test data.
svm = SVC(kernel='poly', degree=2, gamma=1, C=1)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

# Show the confusion matrix based on the SVM model performance.
cm_svm = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred_svm, labels=svm.classes_),
                                display_labels=svm.classes_)
cm_svm.plot()
plt.title('SVM Confusion Matrix')
plt.show()

# Show the classification report based on the SVM model performance.
print('Classification Report (SVM):')
print(classification_report(y_test, y_pred_svm))

# Build a Decision Tree model, train it, and test it by predicting the test data.
decision_tree = DecisionTreeClassifier(criterion='entropy', max_depth=10)
decision_tree.fit(X_train, y_train).predict(X_test)
y_pred_decision_tree = decision_tree.predict(X_test)

# Show the confusion matrix based on the Decision Tree model performance.
cm_decision_tree = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred_decision_tree, labels=decision_tree.classes_),
                                          display_labels=decision_tree.classes_)
cm_decision_tree.plot()
plt.title('Decision Tree Confusion Matrix')
plt.show()

# Show the classification report based on the Decision Tree model performance.
print('Classification Report (Decision Tree):')
print(classification_report(y_test, y_pred_decision_tree))

# Build an NB model, train it, and test it by predicting the test data.
nb = CategoricalNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)

# Show the confusion matrix based on the NB model performance.
cm_nb = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred_nb, labels=nb.classes_),
                               display_labels=nb.classes_)
cm_nb.plot()
plt.title('NB Confusion Matrix')
plt.show()

# Show the classification report based on the NB model performance.
print('Classification Report (NB):')
print(classification_report(y_test, y_pred_nb))

# Compare the 3 model's accuracy scores.
models = ['SVM', 'Decision Tree', 'NB']
accuracy_scores = [accuracy_score(y_test, y_pred_svm), accuracy_score(y_test, y_pred_decision_tree), accuracy_score(y_test, y_pred_nb)]
plt.bar(models, accuracy_scores)
plt.title("Models' Accuracy Scores")
plt.xlabel('Model')
plt.ylabel('Accuracy Score')
plt.show()