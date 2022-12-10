import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd
app = Flask(__name__)

# app=Flask(__name__)
# #load the model
pickled_model = pickle.load(open('knnmodel.pkl', 'rb'))
# scaler=pickle.load(open('scaling.pkl', 'rb'))

# @app.route('/')
# def home():
#     return render_template('home.html')

# @app.route('/predict_api', methods=["POST"])
# def predict_api():
#     data=request.json['data']
#     print(data)
#     print(np.array(list(data.values())).reshape(1, -1))
#     new_data=scalar.transform(np.array(list(data.values())).reshape(1, -1))
#     regre



# if __name__=="__main__":
#     app.run(debug=True)

import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

# diabetes = pd.read_csv('diabetes.csv')
# y = diabetes["Outcome"].values
# X = diabetes.drop('Outcome', axis=1)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
# # Create a StandardScater model and fit it to the training data

# scaler = StandardScaler()
# scaler.fit(X_train)
# X_train_scaled = scaler.transform(X_train)
# X_test_scaled = scaler.transform(X_test)
# knn = KNeighborsClassifier(n_neighbors=7)
# knn.fit(X_train_scaled, y_train)
# print('k=7 Test Acc: %.3f' % knn.score(X_test_scaled, y_test))

# app=Flask(__name__)

@app.route('/')
def man():
    return render_template("home.html")


@app.route('/predict', methods=["POST"])
def home():
    pickled_model = pickle.load(open('knnmodel.pkl', 'rb'))

    Pregnancies = request.form['a']
    Glucose = request.form['b']
    BloodPressure = request.form['c']
    SkinThickness = request.form['d']
    Insulin = request.form['e']
    BMI = request.form['f']
    DiabetesPedigreeFunction = request.form['g']
    Age = request.form['h']
    
    # arr = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age ]])
    # prediction = knnmodel.predict(arr)
    predict_example = (Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age)    
    input_data_as_numpy_array = np.asarray(predict_example)
    input_data_reshape = input_data_as_numpy_array.reshape(1, -1)
    std_data = scaler.transform(input_data_reshape)
    prediction = knn.predict(std_data)



    return render_template('prediction_result.html', data=prediction)


if __name__=="__main__":
    app.run(debug=True)