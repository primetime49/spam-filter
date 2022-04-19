import numpy as np

def split_data(k,n,X,y):
    start_test = int(n*k/10)
    end_test = int(n*(k+1)/10)
    X_test = X[start_test:end_test]
    y_test = y[start_test:end_test]
    X_train = np.concatenate((X[0:start_test],X[end_test:n]))
    y_train = np.concatenate((y[0:start_test],y[end_test:n]))
    return (X_train,y_train,X_test,y_test)