#Generate linear data with noise
import numpy as np
import os
import matplotlib.pyplot as plt

#Number of data
N = 500

#the relationship we want to predict. y = b0+b1x
b0 = 3
b1 = 10

data = np.array([i*b0+b1 for i in range(N)])

noisiness = 30 * b0
noise = np.random.normal(loc=0, scale=noisiness, size=data.shape)
data = data + noise

EXPORT_DATA = True

if EXPORT_DATA:
    output_dir = 'linear_regression/data'
    os.makedirs(output_dir, exist_ok=True)

    # Save the file to that folder
    file_path = os.path.join(output_dir, 'linreg.csv')
    np.savetxt(file_path, data, delimiter=',', fmt='%.5f')


plt.xlabel(f"Noisiness: {noisiness}")
plt.scatter(list(range(len(data))), data)
plt.show()


