import numpy as np

class perceptron:

    def __init__(self , learning_rate = 0.01 , n = 1000):
        self.lr = learning_rate
        self.n_iters =  n
        self.activation_function = self.unit_step_function
        self.weights = None
        self.bias = None


    def fit(self , X,y):
        n_sample , n_feature = X.shape


        #init weight
        self.weights = np.zeros(n_feature)
        self.bias = 0

        y_ =[]
        for i in y:
            if (i>0):
                y_.append(1)
            else:
                y_.append(-1)
        # y_ = np.array([1 if i>0 else 0 for i in y])

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i , self.weights) + self.bias
                y_predicted = self.activation_function(linear_output)

                update = self.lr* (y_[idx]- y_predicted)
                self.weights += update * x_i
                self.bias +=update


    def prediction(self , X):
        linear_output = np.dot(X , self.weights) + self.bias
        y_predicted = self.activation_function(linear_output)
        return y_predicted

    
    def unit_step_function(self , x):
         return np.where(x>=0 ,1, 0)

