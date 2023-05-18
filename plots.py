import numpy as np
import matplotlib.pyplot as plt

#train = np.loadtxt('./training/training.txt')
val1= np.loadtxt('./training/validation_simple.txt')
val2= np.loadtxt('./training/validation.txt')

#plt.plot(np.log10(train[:,-1]),label='Training')
plt.plot(np.log10(val1[:,-1]),label = 'Validation - Simple')
plt.plot(np.log10(val2[:,-1]),label = 'Validation - CONN')

plt.legend()
plt.show
plt.show()
