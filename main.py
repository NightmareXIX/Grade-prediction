import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model

data = pd.read_csv("student-mat.csv", sep=";")
data = data[["G1", "G2", "G3", "studytime", "failures", "absences", "famrel", "freetime"]]

predict = "G3"

x = np.array(data.drop([predict], axis=1))
y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

lr = linear_model.LinearRegression()
lr.fit(x_train, y_train)

acc = lr.score(x_test, y_test)

predictions = lr.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

print("Accuracy: ", acc)
