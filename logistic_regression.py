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

def MSE(X_input ,y_input, theta):
    """
    Gets matrix X_input, vector y_input, and vector theta then compute mean square error 
    """
    return ((sigmoid(X_input @ theta) - y_input).T@ \
        (sigmoid(X_input @ theta) - y_input))[0][0] / len(X_input)

def MSE_prediction(predict_BC_input, y_input):
    """
    Gets prediction values and labels
    Return mean square error
    """
    return sum((predict_BC_input - y_input)**2)[0] / len(y_input)

def accuracy(predict_input ,y_input):
    """
    Gets vector predicted value and vector y_input theta then compute 
    the percentage of the sample that was predicted correctly
    """
    return sum(predict_input == y_input)[0] / len(predict_input) * 100

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

def pick_one_vs_all(y_input, class_label):
    """
    Gets labels(y_input) and class label(class_label) that we want to select 
    Return labels which our class label is 1 and other classes are 0
    """
    return np.array(y_input == class_label, dtype=float).reshape(-1,1)

def one_vs_rest(X_train, X_test, y_train, y_test, alpha):
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
        theta_i_rest = gradient_ascent_for_maximum_likelihood(X_train, y_i_rest_train, alpha, plotter= 0, printer = 0)
        prob_i_rest_train.append(probability(X_train,theta_i_rest))
        prob_i_rest_test.append(probability(X_test,theta_i_rest))
    predict_MC_train_ova = np.argmax(prob_i_rest_train, axis=0)
    predict_MC_test_ova = np.argmax(prob_i_rest_test, axis=0)
    train_accuracy_ova = accuracy(predict_MC_train_ova,y_train)
    test_accuracy_ova = accuracy(predict_MC_test_ova,y_test)
    return train_accuracy_ova, test_accuracy_ova

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

def one_vs_one(X_train, X_test, y_train, y_test, classes_train, alpha):
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
            theta_i_j = gradient_ascent_for_maximum_likelihood(X_i_j_train, y_i_j_train, alpha, plotter= 0, printer = 0)
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
    return train_accuracy_ovo, test_accuracy_ovo
