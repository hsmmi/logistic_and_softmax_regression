from my_io import read_dataset_with_pandas
from normalization import clipping, log_scaling, range_min_to_max, zero_mean_unit_variance

def read_dataset_to_X_and_y(file, range_atr= None, normalization = None, min_value = None, max_value = None):
    """
    Read the attribute(range_atr) that you want and put X0 = 1 and thoes attribute 
    of all samples in X and all samples lable in y 
    normalization:
    .   by default is None and can be "z_score", "scaling", "clipping"
        or "log_scaling"
    .   for "scaling", "clipping" must set min_value and max_value
    Return X and y as nparray
    """
    import numpy as np
    col_name, data = read_dataset_with_pandas(file)
    data = data.to_numpy()

    if(range_atr == None):
        samples = np.array(list(map(lambda x:np.concatenate(([1], x[:-1])),data)))
        lable = np.array(list(map(lambda x:[x[-1]],data)))
    else:
        if(range_atr[1] == len(col_name)):
            range_atr[1] -= 1
        samples = np.array(list(map(lambda x:np.concatenate(([1], x[range_atr[0]:range_atr[1]])),data)))
        lable = np.array(list(map(lambda x:[x[-1]],data)))

    if(normalization != None):
        if(normalization == 'z_score'):
            samples = zero_mean_unit_variance(samples)
        elif(normalization == 'scaling'):
            samples = range_min_to_max(samples, min_value, max_value)
        elif(normalization == 'clipping'):
            samples = clipping(samples, min_value, max_value)
        elif(normalization == 'logScaling'):
            samples = log_scaling(samples)
        else:
            print('method should be "z_score", "scaling", "clipping" or "logScaling"')
            return 
    samples[:,0] = float(1)
    return samples, lable

def train_test_split(file, range_atr= None, range_class=None, train_size = 0.75, normalization = None, min_value = None, max_value = None):
    """
    Read dataset from file and split to attribute and clases then slpit 
    it to train and test with first train_size of each class to train and
    rest to test in each class in range_class 
    normalization:
    .   by default is None and can be "z_score", "scaling", "clipping"
        or "log_scaling"
    .   for "scaling", "clipping" must set min_value and max_value
    Return X_train, X_test, y_train, y_test as nparray also classes_train
    and classe_test split by each class
    """
    import numpy as np
    
    col_name, data = read_dataset_with_pandas(file)
    if(range_atr == None):
        range_atr = (0, len(col_name))
    data = data.to_numpy()
    
    label_class, count_class = np.unique(data[:,-1],return_counts=True)
    if(range_class == None):
        range_class = (0, len(label_class))

    lable_start = np.zeros_like(count_class)
    for i in range(1,len(lable_start)):
        lable_start[i] = lable_start[i-1] + count_class[i-1]
      
    data_train = np.concatenate([data[lable_start[i]:lable_start[i]+int(count_class[i]*(train_size))] for i in range(range_class[0],range_class[1])])
    data_test = np.concatenate([data[lable_start[i]+int(count_class[i]*(train_size)):lable_start[i]+count_class[i]] for i in range(range_class[0],range_class[1])])

    if(range_atr[1] == len(col_name)):
        range_atr = (range_atr[0], range_atr[1]-1)
    X_train = np.array(list(map(lambda x:np.concatenate(([1], x[range_atr[0]:range_atr[1]])),data_train)))
    X_test = np.array(list(map(lambda x:np.concatenate(([1], x[range_atr[0]:range_atr[1]])),data_test)))
    y_train = np.array(list(map(lambda x:[x[-1]],data_train)))
    y_test = np.array(list(map(lambda x:[x[-1]],data_test)))

    X_train = X_train.astype(float)
    X_test = X_test.astype(float)

    if(normalization != None):
        if(normalization == 'z_score'):
            X_train = zero_mean_unit_variance(X_train)
            X_test = zero_mean_unit_variance(X_test)
        elif(normalization == 'scaling'):
            X_train = range_min_to_max(X_train, min_value, max_value)
            X_test = range_min_to_max(X_test, min_value, max_value)
        elif(normalization == 'clipping'):
            X_train = clipping(X_train, min_value, max_value)
            X_test = clipping(X_test, min_value, max_value)
        elif(normalization == 'log_scaling'):
            X_train = log_scaling(X_train)
            X_test = log_scaling(X_test)
        else:
            print('method should be "z_score", "scaling", "clipping" or "logScaling"')
            return 

    X_train[:,0] = float(1)
    X_test[:,0] = float(1)
    
    indices = np.unique(y_train, return_inverse=True)[1]
    y_train = np.array(([float(i) for i in indices])).reshape(-1,1)

    indices = np.unique(y_test, return_inverse=True)[1]
    y_test = np.array(([float(i) for i in indices])).reshape(-1,1)

    classes_train = {i:[] for i in range(range_class[0],range_class[1])}
    classes_test = {i:[] for i in range(range_class[0],range_class[1])}
    for i in range(len(X_train)):
        classes_train[int(y_train[i])].append(X_train[i])
    for i in range(len(X_test)):
        classes_test[int(y_test[i])].append(X_test[i])

    return X_train, X_test, y_train, y_test, classes_train, classes_test
