
import numpy as np
import matplotlib.pyplot as plt

def relu_activation(x):
    return max(0, x)

def relu_derivative(x):
    return 1 if x >= 0 else 0

x_values = np.linspace(-5, 5, 100)
y_values_activation = [relu_activation(x) for x in x_values]
y_values_derivative = [relu_derivative(x) for x in x_values]

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(x_values, y_values_activation, label='ReLU Activation')
plt.xlabel('Input')
plt.ylabel('Output')
plt.title('ReLU Activation Function')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(x_values, y_values_derivative, label='ReLU Derivative', color='orange')
plt.xlabel('Input')
plt.ylabel('Derivative')
plt.title('ReLU Derivative')
plt.legend()

plt.tight_layout()
plt.show()
