import numpy as np
from matplotlib import pyplot as plt

def sigmoid(z):
    """
    Gets vector or scaler z 
    Return sigmoid(z)
    """
    z = z.astype(float)
    return 1 / (1 + np.exp(-z)) 

def probability(X_input, theta):
    """ 
    Gets matrix X_input and theta and compute P(X_input | theta)
    """
    return sigmoid(X_input @ theta)

def predict_logistic_regression(X_input, theta):
    """
    Gets matrix X_input and theta and compute P(X_input | theta)
    Return 1 if P(X_input | theta) >= 0.5 and 0 otherwise
    """
    P = probability(X_input, theta)
    number_of_half = np.sum([P==0.5])
    random_label = np.random.randint(0,2,size=number_of_half)
    P[P == 0.5] = random_label
    P = (P > 0.5)
    P = P.astype(int)
    return P

def accuracy(predict_input ,y_input):
    """
    Gets vector predicted value and vector y_input theta then compute 
    the percentage of the sample that was predicted correctly
    """
    return sum(predict_input == y_input)[0] / len(predict_input) * 100

def gradient_ascent_for_maximum_log_likelihood_logistic_regression(X_input ,y_input, alpha, printer=0, plotter=0):
    """
    It gets matrix samples(X_input) and vector labels(y_input) which should be {0, 1}
    Compute learning parameters with updating all theta(i) to maximize the likelihood
    in each epoch and stop if #epochs exit the threshold or difference of
    mean square error was less than eps
    Return learned parameters
    """

    normal_alpha = alpha/len(X_input)

    def gradient(X_input, theta, y_input):
        return X_input.T @ (y_input - sigmoid(X_input @ theta))

    def update_theta(X_input ,y_input, theta, alpha):
        return theta + alpha * gradient(X_input, theta, y_input)
    
    def step_decay(epoch,epochs_drop):
        """
        It 3/4 the learning rate every epochs_drop epochs
        """
        drop = 3/4
        return drop**((1+epoch)//epochs_drop)

    def log_likelihood(X_input ,y_input, theta):
        """
        Gets matrix X_input, vector y_input, and vector theta then compute log likelihod
        """
        return np.mean(y_input * np.log(probability(X_input, theta)) + (1 - y_input) * np.log(1 - probability(X_input, theta)))

    theta = np.zeros((len(X_input[0]),1))

    max_epoch = int(1e5)
    log_likelihood_log = [log_likelihood(X_input ,y_input, theta)]
    epoch = 0
    eps = 1e-6

    for i in range(1,max_epoch):
        theta = update_theta(X_input ,y_input, theta, normal_alpha * step_decay(i,10))
        epoch += 1
        log_likelihood_log.append(log_likelihood(X_input ,y_input, theta))
        if(i > 1 and abs(log_likelihood_log[-2]-log_likelihood_log[-1]) < eps):
            break
    
    if(printer):
        print(f'log_likelihood in each epochs are \n{log_likelihood_log}')
        print(f'After {epoch} epochs')
  
    if(plotter):
        plt.plot(range(0,len(log_likelihood_log)), log_likelihood_log, ".--", label="log_likelihood")
        plt.legend(loc="lower right")
        plt.xlabel('Iteratioin')
        plt.ylabel('log_likelihood')
        plt.title(f'Learning rate {alpha} and final log_likelihood {round(log_likelihood_log[-1],6)}')
        plt.show()
    
    return theta

def binary_classification(X_train, X_test, y_train, y_test, alpha, printer=0, plotter=0):
    """
    Gets matrix X_train and X_test, vector labels y_train and y_test, 
    and learning rate alpha then find learning parameters and predict 
    train and test samples and compute accuracy train and test 
    """
    theta = gradient_ascent_for_maximum_log_likelihood_logistic_regression(X_train, y_train, alpha, printer, plotter)

    predict_BC_train = predict_logistic_regression(X_train,theta)
    predict_BC_test = predict_logistic_regression(X_test,theta)

    train_accuracy_BC = accuracy(predict_BC_train, y_train)
    test_accuracy_BC = accuracy(predict_BC_test, y_test)

    return theta, train_accuracy_BC, test_accuracy_BC, predict_BC_train, predict_BC_test

def pick_one_vs_all(y_input, class_label):
    """
    Gets labels(y_input) and class label(class_label) that we want to select 
    Return labels which our class label is 1 and other classes are 0
    """
    return np.array(y_input == class_label, dtype=float).reshape(-1,1)

def one_vs_rest(X_train, X_test, y_train, y_test, alpha, printer=0, plotter=0):
    """
    Gets matrix X_train, X_test, vector y_trarin, y_test, and alpha 
    then generate all N model one_vs_all and find maximum probability 
    for each sample in all N model
    Return train accuracy and test accurarcy
    """
    prob_i_rest_train = []
    prob_i_rest_test = []
    for i in range(3):
        y_i_rest_train = pick_one_vs_all(y_train,i)
        theta_i_rest = gradient_ascent_for_maximum_log_likelihood_logistic_regression(X_train, y_i_rest_train, alpha, printer, plotter)
        prob_i_rest_train.append(probability(X_train,theta_i_rest))
        prob_i_rest_test.append(probability(X_test,theta_i_rest))
    predict_MC_train_ova = np.argmax(prob_i_rest_train, axis=0)
    predict_MC_test_ova = np.argmax(prob_i_rest_test, axis=0)
    train_accuracy_ova = accuracy(predict_MC_train_ova,y_train)
    test_accuracy_ova = accuracy(predict_MC_test_ova,y_test)
    return train_accuracy_ova, test_accuracy_ova, predict_MC_train_ova, predict_MC_test_ova

def pick_one_vs_one(classes_input, class_0, class_1):
    """
    Get classes input and class0 and class1 
    Return matrix X contain samples of class0 and class1 
    and vector y contain label class0 as 0 and class1 as 1
    """
    X_return = np.concatenate((classes_input[class_0],classes_input[class_1]))
    y_return = np.concatenate((np.zeros(len(classes_input[class_0])), \
        np.zeros(len(classes_input[class_1]))+1)).reshape(-1,1)
    return X_return, y_return

def one_vs_one(X_train, X_test, y_train, y_test, classes_train, alpha, printer=0, plotter=0):
    """
    Gets matrix X_train, matrix X_test, vector y_trarin, vector y_test, 
    classes_train, and alpha then generate all c(N, 2) model one_vs_one 
    and for each sample find mode of all models predictions
    Return train accuracy and test accurarcy
    """
    class_i_j_train = []
    class_i_j_test = []
    for j in range(3):
        for i in range(j):
            X_i_j_train, y_i_j_train = pick_one_vs_one(classes_train, i, j)
            theta_i_j = gradient_ascent_for_maximum_log_likelihood_logistic_regression(X_i_j_train, y_i_j_train, alpha, printer, plotter)
            tmp_label_train = predict_logistic_regression(X_train, theta_i_j)
            tmp_label_train[tmp_label_train == 1] = j
            tmp_label_train[tmp_label_train == 0] = i
            class_i_j_train.append(tmp_label_train)
            tmp_label_test = predict_logistic_regression(X_test, theta_i_j)
            tmp_label_test[tmp_label_test == 1] = j
            tmp_label_test[tmp_label_test == 0] = i
            class_i_j_test.append(tmp_label_test)
    from scipy import stats
    predict_MC_train_ovo = stats.mode(class_i_j_train, axis=0)[0][0]
    predict_MC_test_ovo = stats.mode(class_i_j_test, axis=0)[0][0]
    train_accuracy_ovo = accuracy(predict_MC_train_ovo,y_train)
    test_accuracy_ovo = accuracy(predict_MC_test_ovo,y_test)
    return train_accuracy_ovo, test_accuracy_ovo, predict_MC_train_ovo, predict_MC_test_ovo