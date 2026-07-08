import numpy as np
import matplotlib.pyplot as plt

class PurePerceptron:
    def __init__(self, learning_rate=0.1, epochs=10):
        self.lr = learning_rate
        self.epochs = epochs
        self.W = None
        self.b = 0.0

    def step_function(self, z):
        return 1 if z >= 0 else 0

    def fit(self, X, Y):
        n_features = X.shape[1]
        self.W = np.zeros(n_features) 
        
        for e in range(self.epochs):
            mistakes_in_epoch = 0
            
            print(f"\nEpoch {e+1}")
            
            for i in range(len(X)):
                x_i = X[i]
                y_true = Y[i]
                
                z = np.dot(x_i, self.W) + self.b
                y_pred = self.step_function(z)
                
                error = y_true - y_pred
                
                if error != 0:
                    self.W += self.lr * error * x_i
                    self.b += self.lr * error
                    mistakes_in_epoch += 1
                    print(f"\nMistake on {x_i} (True:{y_true}, Pred:{y_pred}) \nUpdating Weights: W={self.W}, b={self.b:.2f}")
            
            if mistakes_in_epoch == 0:
                print(f"Convergence reached at Epoch {e+1}! 100% Accuracy.")
                break

    def predict(self, X):
        predictions = []
        for x_i in X:
            z = np.dot(x_i, self.W) + self.b
            predictions.append(self.step_function(z))
        return np.array(predictions)

# [Study Hours, Sleep Hours]
X = np.array([
    [2, 3], [1, 2], [2, 2], [3, 1],
    [6, 5], [7, 7], [5, 8], [8, 6]   
])
Y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

model = PurePerceptron(learning_rate=0.1, epochs=15)
model.fit(X, Y)

# To plot W1*X1 + W2*X2 + b = 0, we find X2:
# X2 = -(W1*X1 + b) / W2
x1_range = np.array([0, 10])
x2_range = -(model.W[0] * x1_range + model.b) / model.W[1]

plt.scatter(X[:4, 0], X[:4, 1], color='blue', label='Fail (0)', marker='o', s=100)
plt.scatter(X[4:, 0], X[4:, 1], color='red', label='Pass (1)', marker='^', s=100)
plt.plot(x1_range, x2_range, color='green', label='Decision Boundary')

plt.xlabel('Study Hours')
plt.ylabel('Sleep Hours')
plt.title('Perceptron Decision Boundary')
plt.legend()
plt.grid(True)
plt.xlim(0, 10)
plt.ylim(0, 10)
plt.show()