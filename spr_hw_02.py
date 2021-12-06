import numpy as np
from logistic_regression import MSE, MSE_prediction, accuracy, gradient_ascent_for_maximum_likelihood_logistic_regression, one_vs_one, one_vs_rest, predict_logistic_regression, probability, sigmoid
from plotter import plot_decision_boundary
from preprocessing import train_test_split
from softmax import gradient_gradient_ascent_for_maximum_likelihood_softmax, predict, y_to_one_hot

# # Binary Classification 
# X_train, X_test, y_train, y_test, classes_train, classes_test = train_test_split('dataset/iris.csv',(0,2),(0,2),0.80,normalization='z_score')
# theta = gradient_ascent_for_maximum_likelihood_logistic_regression(X_train, y_train, 300, plotter= 0, printer = 0)
# print(f'equation of the decision boundary is {round(theta[1][0],6)} x1 + {round(theta[2][0],6)} x2 = {round(-theta[0][0],6)}\n')

# # plot_decision_boundary(X_train, classes_train, theta, 'Train set')
# # plot_decision_boundary(X_test, classes_test, theta, 'Test Set')

# predict_BC_train = predict_logistic_regression(X_train,theta)
# MSE_train_BC = MSE_prediction(predict_BC_train, y_train)
# print(f'training MSE is {MSE_train_BC}\n')

# predict_BC_test = predict_logistic_regression(X_test,theta)
# MSE_test_BC = MSE_prediction(predict_BC_test, y_test)
# print(f'testing MSE is {MSE_test_BC}\n')

# X_train, X_test, y_train, y_test, classes_train, classes_test = train_test_split('dataset/iris.csv',train_size=0.80,normalization='z_score')

# train_accuracy_ova, test_accuracy_ova = one_vs_rest(X_train, X_test, y_train, y_test, 1)
# print(f'accuracy in one-vs-all on train is {round(train_accuracy_ova,2)} and on test is {round(test_accuracy_ova,2)}\n')
# train_accuracy_ovo, test_accuracy_ovo = one_vs_one(X_train, X_test, y_train, y_test, classes_train, 1)
# print(f'accuracy in one-vs-one on train is {round(train_accuracy_ovo,2)} and on test is {round(test_accuracy_ovo,2)}\n')

X_train, X_test, y_train, y_test, classes_train, classes_test = train_test_split('dataset/iris.csv',train_size=0.80,normalization="z_score")

theta = gradient_gradient_ascent_for_maximum_likelihood_softmax(X_train, y_train, 0.1, printer=0, plotter=0)
prediction_train_softmax = predict(X_train, theta)
accuracy_train_softmax = accuracy(prediction_train_softmax,y_train)
prediction_test_softmax = predict(X_test, theta)
accuracy_test_softmax = accuracy(prediction_test_softmax,y_test)
print(f'accuracy in softmax on train is {round(accuracy_train_softmax,2)} and on test is {round(accuracy_test_softmax,2)}\n')
