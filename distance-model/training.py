from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse

from sklearn import datasets
import pickle
import joblib

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, help='input csv file path')
parser.add_argument('--output', type=str, help='output model file path')
args = parser.parse_args()

model_name = 'mymodel.pkl' if args.output is None else args.output

df = pd.read_csv(args.input)
y_coordinates = df['y_coordinate'].to_numpy()
distances = df['distance'].to_numpy()

user_degree = 2

poly_features = PolynomialFeatures(degree=user_degree, include_bias=False)
x_train_poly = poly_features.fit_transform(y_coordinates.reshape(-1, 1))

lin_reg = LinearRegression()
lin_reg.fit(x_train_poly, distances)

#모델을 파일에 저장함
joblib.dump(lin_reg, 'mymodel.pkl')

print('Success train model')



'''
X_new = np.linspace(500, 1080, 580).reshape(-1, 1)
X_new_poly = poly_features.transform(X_new)
y_new = lin_reg.predict(X_new_poly)
print('===============================================')
plt.figure()
plt.scatter(y_coordinates, distances, alpha=0.3, color ="orange")
plt.plot(X_new, y_new, "r-", linewidth=2, label="Predictions")
plt.title('Polynomial Regression by weight and height(degree = %d) '%user_degree)
plt.xlabel("y")
plt.ylabel("distance")
plt.legend(loc="upper left")
plt.show()
'''

