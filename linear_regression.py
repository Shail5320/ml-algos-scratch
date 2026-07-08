import numpy as np
import time

# --- THE DATA ---
# X = Square footage of house (scaled), y = House Price (in hundreds of thousands)
X = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y = np.array([2.0, 3.9, 6.1, 8.0, 10.2]) # Notice it roughly follows: price = 2 * size

def train_linear_regression(X, y, epochs=5, lr=0.05):
    # Initialize our knobs randomly or at 0.0
    beta_0 = 0.0
    beta_1 = 0.0
    m = len(X)
    
    print("=" * 60)
    print(f"STARTING LINEAR REGRESSION | Initial Line: y = {beta_1}x + {beta_0}")
    print("=" * 60)
    # time.sleep(1)

    for epoch in range(1, epochs + 1):
        # 1. Forward Pass (Predicting)
        y_pred = beta_0 + beta_1 * X
        
        # 2. Grading (Calculate MSE Loss)
        loss = (1 / (2 * m)) * np.sum((y_pred - y) ** 2)
        
        # 3. Backward Pass (Calculate Slopes/Gradients)
        errors = y_pred - y
        db0 = (1 / m) * np.sum(errors)
        db1 = (1 / m) * np.sum(errors * X)
        
        # 4. Update Knobs (Gradient Descent Step)
        beta_0 -= lr * db0
        beta_1 -= lr * db1
        
        # --- Print Dashboard ---
        print(f"🎬 EPOCH {epoch}")
        print(f"   Current Equation : y = {beta_1:.4f}x + {beta_0:.4f}")
        print(f"   Total MSE Loss   : {loss:.6f}")
        print(f"   Gradients        : d(beta_0) = {db0:+.4f}, d(beta_1) = {db1:+.4f}")
        print("-" * 60)
        

# Run the training loop
train_linear_regression(X, y, epochs=500, lr=0.05)