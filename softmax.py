import numpy as np
from matplotlib import pyplot as plt

from logistic_regression import MSE, probability, sigmoid

def y_to_one_hot(y_input,number_of_class):
    """
    Gets vector of values and return one hot encode 
    """
    one_hot = np.zeros((len(y_input),int(number_of_class)))
    one_hot[np.arange(len(y_input)), y_input.astype(int).T] = 1
    return one_hot

def softmax(z):
    z -= np.max(z) # for stability
    exp = np.exp(z)
    return (exp.T / (np.sum(exp,axis=1))).T

def probability(X_input, theta):
    return softmax(X_input @ theta)

def predict(X_input, theta):
    return np.argmax(probability(X_input, theta), axis=1).reshape(-1,1)

def gradient_gradient_ascent_for_maximum_likelihood_softmax(X_input ,y_input, alpha, printer=0, plotter=0):
    """
    It gets matrix samples(X_input) and vector labels(y_input) which should be {0, 1}
    Compute learning parameters with updating all theta(i) to maximize the likelihood
    in each epoch and stop if #epochs exit the threshold or difference of
    mean square error was less than eps
    Return learned parameters
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

    def loss(prob, y_input):
        return -np.mean(np.log(prob[np.arange(len(y_input)), y_input]))


    theta = np.zeros((len(X_input[0]),number_of_class))

    max_epoch = int(1e5)
    prob = softmax(X_input@theta)
    ce_loss_log = [loss(prob ,y_input)]
    epoch = 0
    eps = 1e-6

    for i in range(1,max_epoch):

        theta = update_theta(X_input ,y_input_one_hot, theta, normal_alpha * step_decay(i,10))
        epoch += 1
        prob = softmax(X_input@theta)
        ce_loss_log.append(loss(prob ,y_input))
        if(i > 1 and abs(ce_loss_log[-2]-ce_loss_log[-1]) < eps):
            break
    
    if(printer):
        print(f'MSE in each epochs are \n{ce_loss_log}')
        print(f'After {epoch} epochs')
  
    if(plotter):
        plt.plot(range(0,len(ce_loss_log)), ce_loss_log, ".--", label="cost function")
        plt.legend(loc="upper right")
        plt.xlabel('Iteratioin')
        plt.ylabel('cross-entropy(CE) loss')
        plt.title(f'Learning rate {alpha} and final MSE {round(ce_loss_log[-1],6)}')
        plt.show()
    
    return theta

