import numpy as np
import matplotlib.pyplot as plt

train = np.loadtxt('./training.txt')
val = np.loadtxt('./validation.txt')

plt.plot(np.log10(train[:,-1]),label='Training')
plt.plot(np.log10(val[:,-1]),label = 'Validation')

plt.legend()
plt.show
plt.show()
