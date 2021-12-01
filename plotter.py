from matplotlib import pyplot as plt
import numpy as np

def plot_decision_boundary(X_input, classes_input, theta, title=None):
    min_x1_input = np.min(X_input[:,1])
    max_x1_input = np.max(X_input[:,1])
    class_0_input = np.array(classes_input[0])
    class_1_input = np.array(classes_input[1])
    plt.plot(class_0_input[:,1], class_0_input[:,2], '.', label='class 0')
    plt.plot(class_1_input[:,1], class_1_input[:,2], '.', label='class 1')
    plt.plot([min_x1_input,max_x1_input], [-theta[1]/theta[2]*min_x1_input-theta[0]/theta[2], \
    -theta[1]/theta[2]*max_x1_input-theta[0]/theta[2]], 'r-', label='decision boundary')
    plt.title(title)
    plt.legend()
    plt.show()