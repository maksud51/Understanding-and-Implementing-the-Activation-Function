
# Explanation of the Rectified Linear Unit (ReLU) Activation Function


## **Objective:**

**1.**	To comprehend the conceptual and mathematics underpinnings of the Activation Function.

**2.**	To execute the Activation Function in a programming language (such as Python).

**3.**The objective is to examine the attributes and consequences of using the Activation Function inside neural networks.

## **Task :**

## **Theoretical Understanding(ReLU):**

ReLU is an activation function introduced by [1], which has strong biological and mathematical underpinning. In 2011, it was demonstrated to further improve training of deep neural networks.
Glorot et al. [2] showed that ReLU activation function in the hidden layers could improve the learning speed of the various deep neural networks. Now the rectified linear unit is used as the standard activation function for deep neural networks.
All hidden layers are activated using the ReLU function; furthermore, the ReLU activation function increases the activation sum's sensitivity and prevents saturation. It seems and behaves like a linear function, but it is a nonlinear function that enables the network to discover intricate nonlinear correlations. It is a piecewise linear function in which half of the input domain is linear, and the other half is nonlinear since the last layer predicts the binary classification output. Due to its many benefits, the Sigmoid function is preferred over the Step and Tanh activation functions. For instance, the Sigmoid function has an "S-shaped" characteristic curve and limits the values to the interval, allows for smoother training than the Step function, and aids in preventing bias in the gradients. Tanh is a rescaled logistic Sigmoid function with outputs in the range and a zero centre.[3]

Traditional activation function is in the form of sigmoid function, denoted as `f_s`, is defined by the following equation:


`f_s = 1 / (1 + e^(-x))`

where `x` is the input to the function.

Gradient-based learning and its derivatives perform poorly in training neural networks because of the widespread saturation of sigmoidal function. It sped the convergence of stochastic gradient descent compared to the fs and rectified linear unit (ReLU)  have gained overall usage. In addition, its implementation is relatively simple, requiring just the thresholding of an activation map to zero [4]. ReLU is referred to as:

The Rectified Linear Unit (ReLU) function, denoted as `f_r(x)`, is defined by the following equation:

`f_r(x) = max(0, x)`

When adopting the ReLU activation function, we will not get very small values. Instead, it is either 0 (causing some gradients to return nothing) or 1. When we introduced the ReLU function into the neural network, we also introduced a lot of sparsity [5]. In a neural network, this means that the activated matrix contains many zeros. When a certain percentage (such as 50%) of activation is saturated, its call this neural network sparse. This can improve efficiency in terms of time and space complexity (constant values usually require less space and lower computational cost) [18]. Yoshua Bengio et al. found that this component of ReLU can actually make neural networks perform better, and it also has the aforementioned efficiency in terms of time and space [6]. ReLU can also be extended to Noisy Relu including Gaussian noise. It is utilized in the restricted Boltzmann machine to solve computer vision task [7]. Although the sparsity of the ReLU function solves the problem of the disappearance of the gradient caused by the "S-shaped" soft saturation activation function. However, the hard saturation of the negative half axis of ReLU is set to 0, which may lead to "neuronal dead" and also make its data distribution non-zero mean.The model may experience neuronal "dead" during the training process.

Figure 1 demonstrates that the gradients at positive values become constant and no longer disappear. This indicates that the problem of disappearing gradients can be avoided by employing the ReLU activation function. This is why the ReLU activation function can accelerate the learning rate of deep neural networks. [8]

![App Screenshot](https://github.com/maksud51/Screenshots/blob/main/Picture1.jpg?raw=true)

Figure 1 Non Linear Activation Function
The graph of the ReLU function is a piecewise linear function that outputs the input value for all positive inputs `(x≥0)`, and zero for all negative inputs `(x<0)`.

From this Figure 2, it is noticed that the gradients at the positive values becomes constant and do not vanish anymore. This means that the vanishing gradient problem can be avoided by using ReLU activation function. This is the reason why ReLU activation function can improve the learning speed of the deep neural networks.


![App Screenshot](https://github.com/maksud51/Screenshots/blob/main/Picture2.png?raw=true)

The ReLU activation function introduces non-linearity to the neural network, allowing it to learn complex patterns and relationships within the data. It is computationally efficient and helps in mitigating the vanishing gradient problem, which can occur with other activation functions like sigmoid or tanh.

**Example :**

Consider a simple artificial neuron with a linear combination of inputs and weights given by:

`Z = W_1 ⋅ X_1 + w_2 . X_2 + b`

where `w_1` and  `w_2` are weights, `ϰ_1` and `x_2` are input features, and b is the bias term.

Now, the output of this neuron with ReLU activation is computed as:

`f(z) = max(0,z)`

Let's take an example with `w_1 = 2`, `w_2 = -1`, `b = 1`, and input features `x_1 = ` and `x_2 = -2`.

**1.Linear combination**
`Z = 2 . 3 + (-1) . (-2) + 1 = 7`

**2. ReLu Activation**

`f(z) = max(0, 7) = 7`

So, the output of the neuron would be 7.
The ReLU activation function essentially outputs the input value if it is positive and zero otherwise. In this example, since the linear combination z is positive, the ReLU activation function outputs the same positive value.

This non-linearity introduced by ReLU is crucial for neural networks to learn complex relationships in data, as it allows the model to capture patterns that linear functions alone cannot represent. Additionally, during backpropagation, the derivative of ReLU helps update the weights efficiently, contributing to the training process.

## **Importance in Neural Networks:**

 Introduces non-linearity: Neural networks with linear activations wouldn't learn complex patterns. ReLU adds non-linearity, allowing networks to learn diverse features.  Faster training: Simpler calculations compared to other activation functions like sigmoid and tanh. This enables efficient training of deep neural networks.  Sparsity: Many units output 0, encouraging model interpretability and potentially reducing overfitting.


 ## **Derivation of the ReLU Activation Function Formula:**

The ReLU activation function is defined as:

 `f(x) = max(0,x)`

To derive this function, consider two cases:

**a.** If `x >= 0`, then `f(x) = x`                
**b.** If `x < 0`, then `f(x) = 0`

Therefore, the derivative `f'(x)` can be expressed as:

![App Screenshot](https://github.com/maksud51/Screenshots/blob/main/3.png?raw=true)

 ## **Output Range of the ReLU Activation Function:**

The output range of the ReLU activation function is ([0, + ∞)). This is because for all positive values of x, the function outputs the input value `(f(x) = x))`, and for all negative values of x, the function outputs zero `(f(x) = 0))`.

So, for any input x, the output `f(x)` is constrained to be non-negative, with the minimum output being 0.

**Example :**

**a.** For `x = 3`, `f(x) = max(0,3) = 3`                
**b.** If `x = -2`, `f(x) = max(0, -2) = 0`

So, as x varies, the output f(x) is always non-nagative, with a minimum value 0.


## **Derivative of the ReLU Activation Function and Backpropagation:**


The derivative of the ReLU activation function is crucial for the backpropagation process in training neural networks. The derivative f'(x) is given by:


![App Screenshot](https://github.com/maksud51/Screenshots/blob/main/3.png?raw=true)

The backpropagation process in a neural network. Suppose during training, the network output a positive value, and the ReLU function was the activation function. The derivative `f'(x) = 1` in this case, indicating that the gradient flows through, allowing for effective weight updates.

However, if the output was negative, the derivative `f'(x) = 0`, indicating that the gradient is effectively stopped from flowing backward, preventing unnecessary updates for neurons that are not contributing to the error. This selective updating is significant for mitigating the vanishing gradient problem.

The significance of this derivative lies in updating the weights during backpropagation. In the context of the gradient descent optimization algorithm, the derivative helps determine the direction and magnitude of the weight updates.

- For positive inputs `(x >= 0))`, the derivative is 1, allowing the gradient to flow through, aiding in weight updates.
- For negative inputs `(x < 0))`, the derivative is 0, effectively stopping the gradient from flowing backward, preventing unnecessary updates for inactive neurons.


This selective behavior of the ReLU derivative helps mitigate the vanishing gradient problem, making the training process more efficient and enabling the neural network to learn effectively from the data.

 ## **Programming Exercise:**

**ReLU Activation Function**

```python
def relu_activation(x):
    return max(0, x)
```

**ReLU Derivative**

```python

def relu_derivative(x):
    return 1 if x >= 0 else 0
```


**ReLU Activation Visualization(Small Dataset)**

```python
import numpy as np
import matplotlib.pyplot as plt

def relu_activation(x):
    return max(0, x)

def relu_derivative(x):
    return 1 if x >= 0 else 0

# Visualization with a larger dataset
x_values = np.linspace(-20, 20, 1000)
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

```
Please check the 'Activation_Functions(ReLu).py' file.

![App Screenshot](https://github.com/maksud51/Screenshots/blob/main/output.png?raw=true)


![App Screenshot](https://github.com/maksud51/Screenshots/blob/main/output2.png?raw=true)



# **Analysis**


## **Advantages and Disadvantages of ReLU Activation Function:** 

**1.Advantages:**

-Simplicity and computational efficiency. 
-Mitigates the vanishing gradient problem.
-Fast training, sparsity, biologically plausible.
-Introduces non-linearity, enabling the network to learn complex patterns. 

**2.Advantages:**

-Can suffer from the "dying ReLU" problem, where neurons may become inactive and stop learning during training.
-sensitive to initialization, limited output range.

## **Impact on Gradient Descent:** 

-Faster convergence due to simpler calculations.
-Zero gradients for dying ReLUs can hinder learning in certain cases.


## **Vanishing Gradients:**

 -Not as susceptible as sigmoid/tanh due to non-zero gradients for positive inputs.                        

 -Still a potential issue in very deep networks, requiring careful architecture design.


## **References**

1. Richard HR Hahnloser, Rahul Sarpeshkar, Misha A Mahowald, Rodney J Douglas, and H Sebastian Seung. 2000. Digital selection and analogue amplification coexist in a cortex-inspired silicon circuit. Nature 405, 6789 (2000), 947
2. X. Glorot, A. Bordes and Y. Bengio, ”Deep sparse rectifier neural networks”, AISTATS, 2011.
3. Reshi, A. A., Rustam, F., Mehmood, A., Alhossan, A., Alrabiah, Z., Ahmad, A., ... & Choi, G. S. (2021). An efficient CNN model for COVID-19 disease detection based on X-ray image classification. Complexity, 2021.
4. Alzubaidi, M. S., Shah, U., Dhia Zubaydi, H., Dolaat, K., Abd-Alrazaq, A. A., Ahmed, A., & Househ, M. (2021, June). The role of neural network for the detection of Parkinson’s disease: a scoping review. In Healthcare (Vol. 9, No. 6, p. 740). MDPI.
5. Wen, W., Wu, C., Wang, Y., Chen, Y., & Li, H. (2016). Learning structured sparsity in deep neural networks. Advances in neural information processing systems, 29, 2074-2082.
6. rpit, D., & Bengio, Y. (2019). The benefits of overparameterization at initialization in deep ReLU networks. arXiv preprint arXiv:1901.03611.
7. Gulcehre, C., Moczulski, M., Denil, M., & Bengio, Y. (2016, June). Noisy activation functions. In International conference on machine learning (pp. 3059-3068). PMLR.
8. Reshi, A. A., Rustam, F., Mehmood, A., Alhossan, A., Alrabiah, Z., Ahmad, A., ... & Choi, G. S. (2021). An efficient CNN model for COVID-19 disease detection based on X-ray image classification. Complexity, 2021.




 



 







