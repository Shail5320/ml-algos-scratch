import numpy as np

X = np.array([
    [1], [2], [3], [4], [5],
    [6], [7], [8], [9], [10],
    [11], [12], [13], [14], [15],
    [16], [17], [18], [19], [20]
])

y = np.array([
    8, 11, 13, 17, 20,
    22, 27, 29, 32, 34,
    39, 41, 44, 46, 50,
    53, 55, 59, 61, 65
])

b0 = b1 = 0

def my_mini_batch_model(X, y, epoch, alpha):
    y_pred = b0 + b1*X
    m = len(X)
    batch_size = 32
    
    for e in range(1, epoch+1):
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        for i in range(0,m,batch_size):
            X_batch = X_shuffled[i:i + batch_size]
            y_batch = y_shuffled[i:i + batch_size]
            batch_m = len(X_batch)

            y_pred = b0 + b1 * X_batch
            error = y_pred - y_batch

            b0 -= alpha * (1/batch_m) * np.sum(error)
            b1 -= alpha * (1/batch_m) * np.sum(error * X_batch)
    
        if e % 100 == 0:
                print(f"Epoch {e} | Last Point Error: {error:.4f} | Total b0: {b0:.4f}, b1: {b1:.4f}")


my_mini_batch_model(X,y,epoch=100,alpha=0.01)