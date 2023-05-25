import numpy as np
import matplotlib.pyplot as plt

train = np.loadtxt('./training/training.txt')
val = np.loadtxt('./training/validation.txt')
#val1= np.loadtxt('./training/validation.txt')
#val2= np.loadtxt('./training/validation_CONN.txt')


index = 0
plt.plot(np.log10(train[:,index]),label='Training')
plt.plot(np.log10(val[:,index]),label = 'Validation')
#plt.plot(np.log10(val1[:,index]),label = 'Validation - Simple')
#plt.plot(np.log10(val2[:,index]),label = 'Validation - CONN')

plt.legend()
plt.show
plt.show()
