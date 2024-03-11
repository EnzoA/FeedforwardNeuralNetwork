import numpy as np
from typing import Callable, List
from matplotlib import pyplot as plt

class FeedforwardNeuralNetwork:
    def __init__(
            self,
            architecture: tuple[int, ...],
            activation_function: Callable[[np.ndarray[float]], float] = None,
            activation_function_derivative: Callable[[np.ndarray[float]], float] = None,
            loss_function: Callable[[np.ndarray[float], np.ndarray[float]], float] = None,
            loss_function_derivative: Callable[[np.ndarray[float], np.ndarray[float]], float] = None) -> None:
        
        # Validate the architecture of the network and randonly initialize the weights.
        assert ((architecture is not None
                and len(architecture) > 0
                and all([x > 0 for x in architecture])),
                f'{architecture} must be a non empty tuple of positive integers.')
        self._W = []
        self._b = []
        units_in_prior_layer = None
        for units_in_current_layer in architecture:
            if units_in_prior_layer is not None:
                self._W.append(np.random.random(size=(units_in_current_layer, units_in_prior_layer)))
                self._b.append(np.random.random(size=(units_in_current_layer, 1)))
            units_in_prior_layer = units_in_current_layer

        # Set the activation function and its derivative.
        assert ((activation_function is None and activation_function_derivative is None
                or activation_function is not None and activation_function_derivative is not None),
                f'Either both {activation_function} and {activation_function_derivative} should be provided or neither.')
        self._f = activation_function if activation_function is not None else lambda x: 1. / (1. + np.exp(-x))
        self._f_derivative = activation_function_derivative if activation_function_derivative is not None else lambda x: self._f(x).T @ (1. - self._f(x))

        # Set the loss function and its derivative.
        assert ((loss_function is None and loss_function_derivative is None
                or loss_function is not None and loss_function_derivative is not None),
                f'Either both {loss_function} and {loss_function_derivative} should be provided or neither.')
        self._loss = loss_function if loss_function is not None else lambda y, y_hat: 0.5 * np.square(np.subtract(y, y_hat))
        self._loss_derivative = loss_function_derivative if loss_function_derivative is not None else lambda y, y_hat: y_hat - y

    def fit(
            self,
            X: np.ndarray[float],
            y: np.ndarray[float],
            epochs: int=1000,
            lr: float=0.1,
            verbose: bool=False) -> List[float]:
        assert X is not None and X.shape[1] == self._W[0].shape[1], f'{X} must have as many columns as input units the architecture has.'
        assert y is not None and y.reshape((-1, 1)).shape[1] == self._W[-1].shape[0], f'{y} must have as many columns as output units the architecture has.'

        if len(y.shape) == 1:
            y = y.reshape((-1, 1))
        if len(X.shape) == 1:
            X = X.reshape((-1, 1))
        epochs_avg_losses = []
        n = X.shape[0]

        for epoch in np.arange(epochs):
            # Shuffle the observations.
            rearrange = np.random.permutation(X.shape[0])
            X = X[rearrange]
            y = y[rearrange]
            loss = 0
            # Iterate over the shuffled samples.
            for i in range(n):
                # Forward pass.
                y_i = y[i, :].reshape((1 if len(y.shape) == 1 else y.shape[1], 1))
                z = [X[i, :].reshape(X.shape[1], 1)]
                a = [z[0]]
                for l, W in enumerate(self._W):
                    z_i = W @ a[l] + self._b[l]
                    z.append(z_i)
                    a.append(self._f(z_i))
                loss += self._loss(y_i, a[-1])[0][0]

                # Backward pass.
                local_gradient = self._loss_derivative(y_i, a[-1]) * self._f_derivative(z[-1])
                for l, W in enumerate(self._W[::-1]):
                    self._W[-l - 1] = self._W[-l - 1] - lr * (local_gradient @ a[-l - 2].T)
                    self._b[-l - 1] = self._b[-l - 1] - lr * local_gradient
                    local_gradient = (W.T @ local_gradient) * self._f_derivative(z[-l - 2])

            # Get the average loss for this epoch.
            loss *= 1/n
            epochs_avg_losses.append(loss)
            
            if (verbose and epoch % (epochs/10)) == 0:
                print('Epoch ', epoch, ' Loss: ', loss)

        return epochs_avg_losses

    def predict(self, x):
        assert (x is not None and x.shape[0] == self._W[0].shape[1],
                f'{x} should be an single observations with the number of features according to the network\'s architecture')
        z = [x.reshape(x.shape[0], 1)]
        a = [z[0]]
        for l, W in enumerate(self._W):
            z_i = W @ a[l] + self._b[l]
            z.append(z_i)
            a.append(self._f(z_i))
        return a[-1]

if __name__ == '__main__':
    xor_nn = FeedforwardNeuralNetwork(architecture=(2, 2, 1))
    
    losses = xor_nn.fit(
        X=np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
        y=np.array([0, 1, 1, 0]),
        epochs=20000,
        lr=0.1,
        verbose=True
    )
 
    plt.plot(losses)
    plt.waitforbuttonpress()

    print(xor_nn.predict(np.array([0, 0])))
    print(xor_nn.predict(np.array([0, 1])))
    print(xor_nn.predict(np.array([1, 0])))
    print(xor_nn.predict(np.array([1, 1])))
