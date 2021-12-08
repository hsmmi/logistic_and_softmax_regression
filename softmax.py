import numpy as np
from matplotlib import pyplot as plt

from logistic_regression import accuracy, probability

def y_to_one_hot(y_input,number_of_class):
    """
    Gets vector of values and return one hot encode 
    """
    one_hot = np.zeros((len(y_input),int(number_of_class)))
    one_hot[np.arange(len(y_input)), y_input.astype(int).T] = 1
    return one_hot

def softmax(z):
    """
    Gets matrix prediction (m x c) and return matrix probability (m x c)
    which in row i is the probability of sample i belongs to class c 
    """
    z -= np.max(z) # for stability
    exp = np.exp(z)
    return (exp.T / (np.sum(exp,axis=1))).T

def probability(X_input, theta):
    """
    Gets matrix X_input(m x n) and matrix theta(n x c) then compute 
    probability matrix(m x c) which in i is the probability of sample i 
    belongs to class c 
    """
    return softmax(X_input @ theta)

def predict(X_input, theta):
    """
    Gets matrix X_input(m x n) and matrix theta(n x c) then compute 
    probability matrix(m x c) and for each sample find label with 
    maximum probability and return vector predicted label(m x 1)
    """
    return np.argmax(probability(X_input, theta), axis=1).reshape(-1,1)

def gradient_gradient_ascent_for_maximum_log_likelihood_softmax(X_input ,y_input, alpha, printer=0, plotter=0):
    """
    It gets matrix samples(X_input) and vector labels(y_input) which should be {0, 1}
    Compute learning parameters with updating all theta(i) to maximize the likelihood
    in each epoch and stop if #epochs exit the threshold or difference of
    mean square error was less than eps
    Return learned parameters(n x c)
    """
    number_of_class = int(y_input.max()+1)
    y_input = y_input.astype(int)
    y_input_one_hot = y_to_one_hot(y_input,number_of_class)
    normal_alpha = alpha/len(X_input)

    def gradient(X_input, theta, y_input_one_hot):
        return X_input.T @ (y_input_one_hot - probability(X_input, theta))

    def update_theta(X_input ,y_input_one_hot, theta, alpha):
        return theta + alpha * gradient(X_input, theta, y_input_one_hot)
    
    def step_decay(epoch,epochs_drop):
        """
        It 3/4 the learning rate every epochs_drop epochs
        """
        drop = 3/4
        return drop**((1+epoch)//epochs_drop)

    def log_likelihood(prob, y_input):
        return np.mean(np.log(prob[np.arange(len(y_input)).reshape(-1,1), y_input]))

    theta = np.zeros((len(X_input[0]),number_of_class))
    max_epoch = int(1e5)
    prob = probability(X_input, theta)
    log_likelihood_log = [log_likelihood(prob ,y_input)]
    epoch = 0
    eps = 1e-6

    for i in range(1,max_epoch):
        theta = update_theta(X_input ,y_input_one_hot, theta, normal_alpha * step_decay(i,10))
        epoch += 1
        prob = probability(X_input, theta)
        log_likelihood_log.append(log_likelihood(prob ,y_input))
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

def softmax_classification(X_train, X_test, y_train, y_test, alpha, printer=0, plotter=0):
    """
    Gets matrix X_train and X_test, vector labels y_train and y_test, 
    and learning rate alpha then find learning parameters and predict 
    train and test samples and compute accuracy train and test 
    Return learned parameters, accuracy train, accuracy test, predict train, predict test
    """
    theta = gradient_gradient_ascent_for_maximum_log_likelihood_softmax(X_train, y_train, alpha, printer, plotter)

    prediction_train_softmax = predict(X_train, theta)
    prediction_test_softmax = predict(X_test, theta)
    
    accuracy_train_softmax = accuracy(prediction_train_softmax,y_train)
    accuracy_test_softmax = accuracy(prediction_test_softmax,y_test)

    return theta, accuracy_train_softmax, accuracy_test_softmax, prediction_train_softmax, prediction_test_softmax