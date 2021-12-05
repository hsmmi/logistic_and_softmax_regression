import numpy as np
from logistic_regression import MSE, accuracy, gradient_ascent_for_maximum_likelihood, logistic_regression, logistic_regression_evaluation, sigmoid
from plotter import plot_decision_boundary
from preprocessing import train_test_split

# Binary Classification 
X_train, X_test, y_train, y_test, classes_train, classes_test = train_test_split('dataset/iris.csv',(0,2),(0,2),0.80,normalization='z_score')
theta = gradient_ascent_for_maximum_likelihood(X_train, y_train, 300, plotter= 0, printer = 1)
print(theta)
# plot_decision_boundary(X_train, classes_train, theta, 'Train set')
# plot_decision_boundary(X_test, classes_test, theta, 'Test Set')

print(accuracy(X_train,y_train,theta))
print(accuracy(X_test,y_test,theta))

X_train, X_test, y_train, y_test, classes_train, classes_test = train_test_split('dataset/iris.csv',train_size=0.80,normalization='z_score')

def pick_one_vs_all(y_input, class_lable):
    return np.array(y_input == class_lable, dtype=float).reshape(-1,1)

y_0_rest_train = pick_one_vs_all(y_train,0)
y_0_rest_test = pick_one_vs_all(y_test,0)
theta_0_rest = gradient_ascent_for_maximum_likelihood(X_train, y_0_rest_train, 1, plotter= 0, printer = 0)
print(theta_0_rest)
print(accuracy(X_train,y_0_rest_train,theta_0_rest))
print(accuracy(X_test,y_0_rest_test,theta_0_rest))

def pick_one_vs_one(classes_input, class_0, class_1):
    X_return = np.concatenate((classes_input[class_0],classes_input[class_1]))
    y_return = np.concatenate((np.zeros(len(classes_input[class_0])), \
        np.zeros(len(classes_input[class_1]))+1)).reshape(-1,1)
    return X_return, y_return

X_0_1_train, y_0_1_train = pick_one_vs_one(classes_train, 0, 1)
X_0_1_test, y_0_1_test = pick_one_vs_one(classes_test, 0, 1)


print('hi')