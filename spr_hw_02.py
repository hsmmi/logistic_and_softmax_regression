from logistic_regression import binary_classification, one_vs_one, one_vs_rest
from plotter import plot_decision_boundary
from preprocessing import train_test_split
from softmax import softmax_classification

def logistic_regresion_binary_classification():
    X_train, X_test, y_train, y_test, classes_train, classes_test = train_test_split('dataset/iris.csv',(0,2),(0,2),0.80,normalization='z_score')

    theta, train_accuracy_BC, test_accuracy_BC = binary_classification(X_train, X_test, y_train, y_test, 1, plotter= 0, printer = 0)[0:3]
    print(f'Equation of the decision boundary is {round(theta[1][0],6)} x1 + {round(theta[2][0],6)} x2 = {round(-theta[0][0],6)}\n')

    plot_decision_boundary(X_train, classes_train, theta, 'Train set')
    plot_decision_boundary(X_test, classes_test, theta, 'Test Set')

    print(f'Accuracy in binary classification on train is {round(train_accuracy_BC,2)}% and on test is {round(test_accuracy_BC,2)}%\n')

def logistic_regression_multiclass_classification():
    X_train, X_test, y_train, y_test, classes_train = train_test_split('dataset/iris.csv',train_size=0.80,normalization='z_score')[0:5]

    train_accuracy_ova, test_accuracy_ova = one_vs_rest(X_train, X_test, y_train, y_test, 1, plotter=0, printer=0)[0:2]
    print(f'Accuracy in one-vs-all on train is {round(train_accuracy_ova,2)}% and on test is {round(test_accuracy_ova,2)}%\n')
    train_accuracy_ovo, test_accuracy_ovo = one_vs_one(X_train, X_test, y_train, y_test, classes_train, 1)[0:2]
    print(f'Accuracy in one-vs-one on train is {round(train_accuracy_ovo,2)}% and on test is {round(test_accuracy_ovo,2)}%\n')

def softmax_classification():
    X_train, X_test, y_train, y_test = train_test_split('dataset/iris.csv',train_size=0.80,normalization="z_score")[0:4]

    accuracy_train_softmax, accuracy_test_softmax = softmax_classification(X_train, X_test, y_train, y_test, 1, printer=0, plotter=0)[1:3]
    print(f'Accuracy in softmax on train is {round(accuracy_train_softmax,2)}% and on test is {round(accuracy_test_softmax,2)}%\n')

# logistic_regresion_binary_classification() # uncomment to run logistic regresion _ binary classification
# logistic_regression_multiclass_classification() # uncomment to run logistic regresion _ multiclass classification
# softmax_classification() # uncomment to run softmax classification

