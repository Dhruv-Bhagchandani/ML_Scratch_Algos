import numpy as np


class GDLogisticRegression:
    def gd (X,y) : 
        # Function parameters : X : input features, y : output features
    
        # Insert 1s in the first column of your input features for transformation purposes
        np.insert(X,0,1,axis=1)

        # Initialize weights(coefficients) to any desired values : we'll take all ones 
        weights = np.ones(X.shape[1])

        # Define learning rate : 
        lr = 0.01  

        # loop to reach to our desired logistic regression line : 
        for i in (5000) : 
            
            # Our model prediction : 
            y_hat = sigmoid(np.dot(X,weights))
        
            # weights updation : 
            weights = weights + lr*((np.dot( y - y_hat ),X)/X.shape[0])

        return weights[0],weights[1:]  # constant,coefficients
    
    # Sigmoid function : 
    def sigmoid(z):
        return 1/( 1 + np.exp(-z) )
