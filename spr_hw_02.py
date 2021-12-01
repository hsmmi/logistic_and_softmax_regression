import numpy as np
from linear_regression import MSE, accuracy, gradient_ascent_for_maximum_likelihood, linear_regression, linear_regression_evaluation, sigmoid
from plotter import plot_decision_boundary
from preprocessing import train_test_split

X_train, X_test, y_train, y_test, classes_train, classes_test = train_test_split('dataset/iris.csv',(0,2),(0,2),0.80,normalization='z_score')
theta = gradient_ascent_for_maximum_likelihood(X_train, y_train, 300, plotter= 0, printer = 1)


# plot_decision_boundary(X_train, classes_train, theta, 'Train set')
# plot_decision_boundary(X_test, classes_test, theta, 'Test Set')

print(accuracy(X_train,y_train,theta))
print(accuracy(X_test,y_test,theta))