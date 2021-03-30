# Importing all dependencies
import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model, preprocessing
import pandas as pd
import numpy as np

data = pd.read_csv('Datasets/car.data')

# Features: buying, maint, dooor, persons, lug_boot, safety, class
# Classes: 

le = preprocessing.LabelEncoder()

# Converting all str data to int lists
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))

labels = "class"

x = list(zip(buying, maint, door, persons, lug_boot, safety))  # Attribute data
y = list(cls) # Label Data

# Splitting all data into training and testing
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1) 

model = KNeighborsClassifier(n_neighbors=9) # KNN Model Initializing 
# tweaking the n_neighbors might help in increasing accuracy

model.fit(x_train, y_train) # Fitting in the data
acc = model.score(x_test, y_test)
print(acc)

predicted = model.predict(x_test)

names = ['unacc', 'acc', 'good', 'vgood']

for x in range(len(predicted)):
    print(f'Predicted: {names[predicted[x]]}, Data: {x_test[x]}, Actual: {names[y_test[x]]}')
