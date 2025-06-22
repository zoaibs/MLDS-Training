#The point of linear regression is to find a straight line that best fits a set of data points
#We want to minimize "loss", the sum of residuals
#the residual is the difference between the actual data point and our prediction line at the same x coordinate
#we square this value so positive and negative error dont cancel out

#You can do a calculus based approach where you minimize the sum of the residuals.
#basically take the partial derivative of the loss fn wrt to b0, b1
#solve dJ/db0 = 0 for b0, then plug that into dJ/db1 = 0 and solve for b1

#However, modern implementations use matrix operations to achieve the same result a lot faster
#matrix ops are vectorized and very fast, with external libraries.
#Also, its more scalable for multivariate linear regression.

#y = Xb, where y (N x 1) is the target vector, X (N x p) is our matrix of x's and b (p x 1) is our vector of coefficients
#J(b) = ||y-Xb||^2 ... some matrix math and calculus later you arrive at:
#b=(X^T X)^−1(X^T y)

import numpy as np
import os

input_dir = 'linear_regression/data'
os.makedirs(input_dir, exist_ok=True)
file_path = os.path.join(input_dir, 'linreg.csv')

y = np.loadtxt(file_path, delimiter=',')
x = np.arange(len(y))

X = np.column_stack((np.ones(len(x)), x))

# beta = (X^T X)^-1 X^T y
beta = np.linalg.inv(X.T @ X) @ X.T @ y

print(f"Estimated b0 (intercept): {beta[0]:.3f}")
print(f"Estimated b1 (slope): {beta[1]:.3f}")

