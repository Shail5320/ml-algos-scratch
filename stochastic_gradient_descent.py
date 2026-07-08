import numpy as np
import matplotlib.pyplot as plt

# X = Years of Experience
X = np.array([1, 2, 3, 4, 5, 6, 7, 8])
# Y = Salary in LPA (Lakhs Per Annum)
Y = np.array([4, 5, 8, 8, 11, 13, 14, 16])

def my_model(X,Y, epochs, alpha):

    m = len(X)
    b0 = b1 = 0
    for e in range(1, epochs+1):

        random_indices = np.random.permutation(m)
        X_shuffled = X[random_indices]
        Y_shuffled = Y[random_indices]
        
        for i in range(len(X)):

            # x_i = X[i]
            # y_i = Y[i]

            x_i = X_shuffled[i]
            y_i = Y_shuffled[i]

            y_pred = b0 + b1*x_i
            error = (y_pred-y_i)
            
            b0 -= alpha*error
            b1 -= alpha*error*x_i
        
        plt.plot(X, b0+b1*X, color = "red")
        plt.show()

        
    
    final_predictions = b0 + b1 * X
    total_mse = np.mean((final_predictions - Y) ** 2)
    print(f"Total Error : {total_mse}")

my_model(X,Y,epochs=1000,alpha=0.001)