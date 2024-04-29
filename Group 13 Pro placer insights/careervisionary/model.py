import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import preprocessing

# Step 1: Prepare the data
df = pd.read_csv('collegePlace.csv')

x = df.drop(['PlacedOrNot', 'Age', 'Hostel'], axis='columns')  # Drop unnecessary columns
y = df['PlacedOrNot']

# Encode categorical variables
le = preprocessing.LabelEncoder()
x['Gender'] = le.fit_transform(x['Gender'])
x['Stream'] = le.fit_transform(x['Stream'])

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=100)

# Step 2: Train the SVM model
svm_model = SVC(kernel='linear')  # Using a linear kernel for simplicity
svm_model.fit(x_train, y_train)

# Step 3: Save the SVM model
pickle.dump(svm_model, open('svm_model.pkl', 'wb'))

# Optionally, you can load the model and make predictions
loaded_model = pickle.load(open('svm_model.pkl', 'rb'))
print(loaded_model.predict([[1, 1, 1, 0, 0]]))
