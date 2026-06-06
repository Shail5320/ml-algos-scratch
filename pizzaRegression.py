import numpy as np
import matplotlib.pyplot as plt

# More Epochs More Consistency
# We have only one feature so the prediction model will be y = b0 + x*b1

# Show how the model shifts at every epoch, enhancing itself
def showGraph(X,Y):
    plt.scatter(X, Y, color='blue', label='Data Points')
    plt.xlabel('Distance (Miles)')
    plt.ylabel('Delivery Time (Minutes)')
    plt.title('Pizza Delivery Time vs Distance')
    plt.legend()
    return plt.show()

def my_linear_model(X,Y,lr,epochs):
    # My Weights
    b0, b1  = 0,0
    m = len(X)

    for e in range(epochs):
        y_pred = b0 + b1*X
        # Find the Loss with current weights (MSE)
        ms_error = 1/(2*m) * np.sum((y_pred-Y)**2)

        # Minimize the Loss by changing weights
        # calculating the d(lossFx)/d(b) :- gradient
        error = y_pred-Y
        db0 = 1/m * np.sum(error)
        db1 = 1/m * np.sum(error*X)

        # Updating the weights
        b0 -= lr*db0
        b1 -= lr*db1

        # Showing Results
        print(f"\nEpoch {e} ended Successfully")
        print(f"Error was {np.sum(error)}")
        print(f"Weight b0 = {b0} and Weight b1 = {b1}\n")

    plt.scatter(X, Y, color='blue', label='Data Points')
    plt.plot(X, b0+b1*X, color="red", label="Prediction Line") 
    plt.legend()
    plt.show()
        
    return(f"\nAfter {epochs} epochs, b0 = {b0}, b1 = {b1}\n")


# X = Independent variable (Distance), Y = Dependent variable (Time)
X = np.array([1.2, 2.5, 3.1, 4.8, 5.0, 6.2, 7.1, 8.5, 9.0, 10.4])
Y = np.array([15, 22, 25, 37, 33, 45, 49, 58, 60, 68])

ch = int(input("Do you wish to see Dataset Visualization [1] or Train Model [2] ? "))

if ch==1:
    plt.scatter(X, Y, color='blue', label='Data Points')
    plt.xlabel('Distance (Miles)')
    plt.ylabel('Delivery Time (Minutes)')
    plt.title('Pizza Delivery Time vs Distance')
    plt.legend()
    plt.show()
elif ch==2:
    epochs = int(input("Enter the number of Epochs : "))
    lr = float(input("Enter the value of learning rate (alpha) : "))
    print(my_linear_model(X,Y,lr,epochs))
else:
    raise ValueError("Invalid Choice")