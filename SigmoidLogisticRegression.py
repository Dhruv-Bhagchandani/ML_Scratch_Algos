import numpy as np 

class SigmoidLogisticRegression:
    def perceptron (X,y) : 
        # Function parameters : X : input features, y : output features
    
        # Insert 1s in the first column of your input features for transformation purposes
        np.insert(X,0,1,axis=1)

        # Initialize weights(coefficients) to any desired values : we'll take all ones 
        weights = np.ones(X.shape[1])

        # Define learning rate : 
        lr = 0.01  

        # loop to reach to our desired logistic regression line : 
        for i in (1000) : 
            j = np.random.randint(0,100) # here 100 is number of datapoints in the dataset 

            # Our model prediction : 
            y_hat = sigmoid(np.dot(X[j],weights))
        
            # weights updation : 
            weights = weights + lr*( y - y_hat )*X[j]

        return weights[0],weights[1:]  # constant,coefficients

    # Sigmoid function : 
    def sigmoid(z):
        return 1/( 1 + np.exp(-z) )

