from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np

X_train = np.load('S:\\ds440\\trainingrecords\\final\\train.npy')
y_train = np.load('S:\\ds440\\trainingrecords\\final\\train_label.npy')
X_test = np.load('S:\\ds440\\trainingrecords\\final\\test.npy')
y_test = np.load('S:\\ds440\\trainingrecords\\final\\test_label.npy')

clf = RandomForestClassifier(max_depth = 4, random_state = 0)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
ConfusionMatrix = confusion_matrix(y_test, y_pred)
print(ConfusionMatrix)
print(classification_report(y_test, y_pred))