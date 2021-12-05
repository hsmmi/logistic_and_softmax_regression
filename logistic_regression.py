import numpy as np
from matplotlib import pyplot as plt

def sigmoid(z):
    """
    Gets vector or scaler z 
    Return sigmoid(z)
    """
    z = z.astype(float)
    return 1 / (1 + np.exp(-z)) 

def predict(X_input, theta):
    """
    Gets matrix X_input and theta and compute P(X_input | theta)
    Return 1 if P(X_input | theta) >= 0.5 and 0 otherwise
    """
    P = sigmoid(X_input @ theta)
    number_of_half = np.sum([P==0.5])
    random_lable = np.random.randint(0,2,size=number_of_half)
    P[P == 0.5] = random_lable
    return P > 0.5

def MSE(X_input ,y_input, theta):
    """
    Gets matrix X_input, vector y_input, and vector theta then compute mean square error 
    """
    return ((sigmoid(X_input @ theta) - y_input).T@ \
        (sigmoid(X_input @ theta) - y_input))[0][0] / len(X_input)

def accuracy(X_input ,y_input, theta):
    """
    Gets matrix X_input, vector y_input, and vector theta then compute 
    the percentage of the sample that was predicted correctly
    """
    return sum(predict(X_input,theta) == y_input)[0] / len(X_input) * 100

def gradient_ascent_for_maximum_likelihood(X_input ,y_input, alpha, printer=0, plotter=0):
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

    theta = np.zeros((len(X_input[0]),1))

    max_epoch = int(1e5)
    MSE_log = [MSE(X_input ,y_input, theta)]
    epoch = 0
    eps = 1e-6

    for i in range(1,max_epoch):
        theta = update_theta(X_input ,y_input, theta, normal_alpha * step_decay(i,10))
        # theta = update_theta(X_input ,y_input, theta, normal_alpha)
        epoch += 1
        MSE_log.append(MSE(X_input ,y_input, theta))
        if(i > 1 and abs(MSE_log[-2]-MSE_log[-1]) < eps):
            break
    
    if(printer):
        print(f'MSE in each epochs are \n{MSE_log}')
        print(f'After {epoch} epochs')
  
    if(plotter):
        plt.plot(range(0,len(MSE_log)), MSE_log, ".--", label="cost function")
        plt.legend(loc="upper right")
        plt.xlabel('Iteratioin')
        plt.ylabel('MSE')
        plt.title(f'Learning rate {alpha} and final MSE {round(MSE_log[-1],6)}')
        plt.show()
    
    return theta

def logistic_regression(X_train, y_train, alpha = None, printer = 0, plotter = 0):
    """
    It gets matrix samples train(X_train) and their labels(y_train) and method.
    alpha:
    .   If alpha be None it will use closed_form if not it use gradient_descent
        with your alpha
    Return learned parameters (θ0 , θ1 , ..., θn ) and the value of MSE 
    error on the train data.
    """
    if(printer):
        print(f'matrix X_train is\n{X_train}\n')
    if(printer):
        print(f'vector y_train is\n{y_train}\n')


    theta = gradient_ascent_for_maximum_likelihood(X_train,y_train,alpha,printer,plotter)

    if(printer):
        print(f'vector learned parameters (θ0 , θ1 , ..., θn ) is\n{theta}\n')

    prediction_train = X_train @ theta
    if(printer):
        print(f'vector prediction_train is\n{prediction_train}\n')

    error_train = prediction_train - y_train
    if(printer):
        print(f'vector error_train is\n{error_train}\n')

    MSE_train = (np.square(error_train)).mean(axis=0)
    if(printer):
        print(f'MSE on train data is\n{MSE_train}\n')
    
    return theta, MSE_train

def logistic_regression_evaluation(X_test, y_test, theta, printer = 0, plotter=0):
    """
    It gets matrix samples test(X_test) and their labels(y_test) and learned
    parameters(theta) and plotter(by default 0).
    If plotter be 1 then it'll plot the test sample and a regression line
    Return the value of MSE error on the test data.
    """
    if(printer):
        print(f'matrix X_test is\n{X_test}\n')
    if(printer):
        print(f'vector y_test is\n{y_test}\n')

    prediction_test = X_test @ theta
    # predict
    if(printer):
        print(f'vector prediction_test is\n{prediction_test}\n')

    error_test = prediction_test - y_test
    if(printer):
        print(f'vector error_test is\n{error_test}\n')

    MSE_test = np.square(error_test).mean(axis=0)
    if(printer):
        print(f'MSE on test data is\n{MSE_test}\n')

    if(plotter):
        sX_test = X_test.argmin(axis=0)[1]
        eX_test = X_test.argmax(axis=0)[1]

        plt.plot(list(zip(*X_test))[1], y_test, ".", label="sample")
        plt.plot([X_test[sX_test][1],X_test[eX_test][1]], \
            [prediction_test[sX_test],prediction_test[eX_test]], "-r", \
                label="regression line")
        plt.legend(loc="upper left")
        plt.xlabel('Feature')
        plt.ylabel('Lable')
        plt.show()

    return MSE_test


