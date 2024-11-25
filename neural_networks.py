import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import os
from functools import partial
from matplotlib.patches import Circle

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

# Define a simple MLP class


class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(0)
        self.lr = lr  # learning rate
        self.activation_fn = activation  # activation function
        # TODO: define layers and initialize weights
        self.W1 = np.random.randn(
            input_dim, hidden_dim)
        self.b1 = np.ones(hidden_dim) * 0.1
        self.W2 = np.random.randn(
            hidden_dim, output_dim)
        self.b2 = np.ones(output_dim) * 0.1

    def forward(self, X):
        # TODO: forward pass, apply layers to input X
        # TODO: store activations for visualization
        self.z1 = X.dot(self.W1) + self.b1
        if self.activation_fn == 'tanh':
            self.a1 = np.tanh(self.z1)
        elif self.activation_fn == 'relu':
            self.a1 = np.maximum(0, self.z1)
        elif self.activation_fn == 'sigmoid':
            self.a1 = 1 / (1 + np.exp(-self.z1))

        self.z2 = self.a1.dot(self.W2) + self.b2
        self.a2 = self.z2  # No activation function for output layer I guess ?
        return self.a2

    def backward(self, X, y):
        # TODO: compute gradients using chain rule
        # TODO: store gradients for visualization
        m = y.shape[0]
        y = y.reshape(-1, 1)

        self.grad_z2 = (self.a2 - y) / m
        self.grad_W2 = self.a1.T.dot(self.grad_z2)
        self.grad_b2 = np.sum(self.grad_z2, axis=0)

        grad_a1 = self.grad_z2.dot(self.W2.T)
        if self.activation_fn == 'tanh':
            grad_z1 = grad_a1 * (1 - self.a1 ** 2)
        elif self.activation_fn == 'relu':
            grad_z1 = grad_a1 * (self.z1 > 0)
        elif self.activation_fn == 'sigmoid':
            grad_z1 = grad_a1 * self.a1 * (1 - self.a1)

        self.grad_W1 = X.T.dot(grad_z1)
        self.grad_b1 = np.sum(grad_z1, axis=0)

        # TODO: update weights with gradient descent
        self.W1 -= self.lr * self.grad_W1
        self.b1 -= self.lr * self.grad_b1
        self.W2 -= self.lr * self.grad_W2
        self.b2 -= self.lr * self.grad_b2


def generate_data(n_samples=100):
    np.random.seed(0)
    # Generate input
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int) * \
        2 - 1  # Circular boundary
    y = y.reshape(-1, 1)
    return X, y

# Visualization update function


def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y):
    ax_hidden.clear()
    ax_input.clear()
    ax_gradient.clear()

    # perform training steps by calling forward and backward function
    for _ in range(10):
        # Perform a training step
        mlp.forward(X)
        mlp.backward(X, y)

    # TODO: Plot hidden features
    hidden_features = mlp.a1
    ax_hidden.scatter(hidden_features[:, 0], hidden_features[:, 1],
                      hidden_features[:, 2], c=y.ravel(), cmap='bwr', alpha=0.7)
    ax_hidden.set_title(f'Hidden Space at Step {frame*10}')

    # TODO: Hyperplane visualization in the hidden space
    x_vals = np.linspace(hidden_features[:, 0].min(
    )-0.5, hidden_features[:, 0].max()+0.5, 50)
    y_vals = np.linspace(hidden_features[:, 0].min(
    )-0.5, hidden_features[:, 0].max()+0.5, 50)
    X_grid, Y_grid = np.meshgrid(x_vals, y_vals)
    Z_grid = - (mlp.W2[0, 0] * X_grid + mlp.W2[1, 0]
                * Y_grid + mlp.b2[0]) / mlp.W2[2, 0]
    ax_hidden.plot_surface(X_grid, Y_grid, Z_grid, color='yellow', alpha=0.3)

    # TODO: Distorted input space transformed by the hidden layer
    ax_hidden.set_xlim([np.floor(hidden_features[:, 0].min()) -
                       0.5, np.ceil(hidden_features[:, 0].max())+0.5])
    ax_hidden.set_ylim([np.floor(hidden_features[:, 1].min()) -
                       0.5, np.ceil(hidden_features[:, 1].max())+0.5])
    ax_hidden.set_zlim([np.floor(hidden_features[:, 2].min()) -
                       0.5, np.ceil(hidden_features[:, 2].max())+0.5])
    ax_hidden.set_xticks(np.arange(np.floor(hidden_features[:, 0].min()), np.ceil(
        hidden_features[:, 0].max()) + 0.5, 0.5))
    ax_hidden.set_yticks(np.arange(np.floor(hidden_features[:, 1].min()), np.ceil(
        hidden_features[:, 1].max()) + 0.5, 0.5))
    ax_hidden.set_zticks(np.arange(np.floor(hidden_features[:, 2].min()), np.ceil(
        hidden_features[:, 2].max()) + 0.5, 0.5))

    Z = np.dot(np.c_[X_grid.ravel(), Y_grid.ravel()], mlp.W1) + mlp.b1
    if mlp.activation_fn == 'tanh':
        A = np.tanh(Z)
    elif mlp.activation_fn == 'relu':
        A = np.maximum(0, Z)
    elif mlp.activation_fn == 'sigmoid':
        A = 1 / (1 + np.exp(-Z))

    ax_hidden.plot_surface(A[:, 0].reshape(X_grid.shape),  A[:, 1].reshape(
        X_grid.shape),  A[:, 2].reshape(X_grid.shape), color='lightblue', alpha=0.3)

    # TODO: Plot input layer decision boundary
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50),
                         np.linspace(y_min, y_max, 50))
    grid = np.c_[xx.ravel(), yy.ravel()]
    decision = mlp.forward(grid).reshape(xx.shape)
    levels = np.sort([decision.min(), 0, decision.max()])
    ax_input.contourf(xx, yy, decision, levels=levels, colors=['blue', 'red'], alpha=0.5)
    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='bwr', edgecolors='k')
    ax_input.set_title(f'Input Space at Step {frame*10}')

    # TODO: Visualize features and gradients as circles and edges
    ax_gradient.add_patch(Circle((1, 0.5), 0.05, color='blue'))
    ax_gradient.text(1, 0.5, 'y', fontsize=15,
                     ha='center', va='center', color='white')

    for i in range(mlp.W1.shape[0]):
        ax_gradient.add_patch(
            Circle((0,  1 - i / (mlp.W1.shape[0] - 1)), 0.05, color='blue'))
        ax_gradient.text(0, 1 - i / (mlp.W1.shape[0] - 1), f'x{i+1}', fontsize=15,
                         ha='center', va='center', color='white')
        for j in range(mlp.W1.shape[1]):
            ax_gradient.plot([0, 0.5], [1 - i / (mlp.W1.shape[0] - 1), 1 - j / (mlp.W1.shape[1] - 1)],
                             'purple', linewidth=np.abs(mlp.grad_W1[i, j]) * 40)

    for i in range(mlp.W1.shape[1]):
        ax_gradient.plot([0.5, 1], [1 - i / (mlp.W1.shape[1] - 1), 0.5],
                         'purple', linewidth=np.abs(mlp.grad_W2[i, 0]) * 40)
        ax_gradient.add_patch(
            Circle((0.5, 1 - i / (mlp.W1.shape[1] - 1)), 0.05, color='blue'))
        ax_gradient.text(0.5, 1 - i / (mlp.W1.shape[1] - 1), f'h{i+1}', fontsize=15,
                         ha='center', va='center', color='white')

    ax_gradient.set_xlim(-0.1, 1.1)
    ax_gradient.set_ylim(-0.1, 1.1)
    ax_gradient.set_title(f'Gradients at Step {frame*10}')

    # The edge thickness visually represents the magnitude of the gradient


def visualize(activation, lr, step_num):
    X, y = generate_data()
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1,
              lr=lr, activation=activation)

    # Set up visualization
    matplotlib.use('agg')
    fig = plt.figure(figsize=(21, 7))
    ax_hidden = fig.add_subplot(131, projection='3d')
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)

    # Create animation
    ani = FuncAnimation(fig, partial(update, mlp=mlp, ax_input=ax_input, ax_hidden=ax_hidden,
                        ax_gradient=ax_gradient, X=X, y=y), frames=step_num//10, repeat=False)

    # Save the animation as a GIF
    ani.save(os.path.join(result_dir, "visualize.gif"),
             writer='pillow', fps=10)
    plt.close()


if __name__ == "__main__":
    activation = "sigmoid"
    lr = 0.1
    step_num = 1000
    visualize(activation, lr, step_num)
