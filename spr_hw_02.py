import numpy as np
from logistic_regression import MSE, accuracy, gradient_ascent_for_maximum_likelihood, logistic_regression, logistic_regression_evaluation, one_vs_one, one_vs_rest, predict_logistic_regression, probability, sigmoid
from plotter import plot_decision_boundary
from preprocessing import train_test_split

# Binary Classification 
X_train, X_test, y_train, y_test, classes_train, classes_test = train_test_split('dataset/iris.csv',(0,2),(0,2),0.80,normalization='z_score')
theta = gradient_ascent_for_maximum_likelihood(X_train, y_train, 300, plotter= 0, printer = 1)
print(theta)
# plot_decision_boundary(X_train, classes_train, theta, 'Train set')
# plot_decision_boundary(X_test, classes_test, theta, 'Test Set')

predict_BC_train = predict_logistic_regression(X_train,theta)
print(accuracy(predict_BC_train,y_train))
predict_BC_test = predict_logistic_regression(X_test,theta)
print(accuracy(predict_BC_test,y_test))

X_train, X_test, y_train, y_test, classes_train, classes_test = train_test_split('dataset/iris.csv',train_size=0.80,normalization='z_score')


train_accuracy_ova, test_accuracy_ova = one_vs_rest(X_train, X_test, y_train, y_test, 100)
print(train_accuracy_ova, test_accuracy_ova)
train_accuracy_ovo, test_accuracy_ovo = one_vs_one(X_train, X_test, y_train, y_test, classes_train, 100)
print(train_accuracy_ovo, test_accuracy_ovo)

print('hi')