import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import streamlit as st
from io import StringIO
import pandas as pd

class Optimizer:
    """Base class for all optimizers with logging and gradient clipping support."""
    def __init__(self, learning_rate=0.01, clip_norm=None):
        self.learning_rate = learning_rate
        self.clip_norm = clip_norm
        self.logs = {"gradients": [], "params": [], "loss": []}

    def step(self, params, gradients):
        """Apply the optimizer's update rule to the parameters."""
        raise NotImplementedError("Derived classes must implement this method.")

    def clip_gradients(self, gradients):
        """Apply gradient clipping if clip_norm is set."""
        if self.clip_norm:
            total_norm = np.sqrt(sum((g ** 2).sum() for g in gradients))
            if total_norm > self.clip_norm:
                scale = self.clip_norm / (total_norm + 1e-6)
                gradients = [g * scale for g in gradients]
        return gradients

    def log(self, gradients, params, loss):
        """Log gradients, parameters, and loss for analysis."""
        self.logs["gradients"].append([np.linalg.norm(g) for g in gradients])
        self.logs["params"].append([np.linalg.norm(p) for p in params])
        self.logs["loss"].append(loss)

    def visualize_logs(self):
        """Plot logged data."""
        plt.figure(figsize=(12, 4))
        
        # Plot Gradient Norms
        plt.subplot(1, 3, 1)
        plt.plot([np.mean(g) for g in self.logs["gradients"]])
        plt.title("Gradient Norms")
        plt.xlabel("Iterations")
        plt.ylabel("Norm")
        
        # Plot Parameter Norms
        plt.subplot(1, 3, 2)
        plt.plot([np.mean(p) for p in self.logs["params"]])
        plt.title("Parameter Norms")
        plt.xlabel("Iterations")
        plt.ylabel("Norm")
        
        # Plot Loss
        plt.subplot(1, 3, 3)
        plt.plot(self.logs["loss"])
        plt.title("Loss Curve")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        
        plt.tight_layout()
        plt.show()


class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer."""
    def step(self, params, gradients):
        gradients = self.clip_gradients(gradients)
        for i in range(len(params)):
            params[i] -= self.learning_rate * gradients[i]


class Momentum(Optimizer):
    """Momentum-based optimizer."""
    def __init__(self, learning_rate=0.01, momentum=0.9, clip_norm=None):
        super().__init__(learning_rate, clip_norm)
        self.momentum = momentum
        self.velocity = None

    def step(self, params, gradients):
        if self.velocity is None:
            self.velocity = [np.zeros_like(p) for p in params]
        gradients = self.clip_gradients(gradients)
        for i in range(len(params)):
            self.velocity[i] = self.momentum * self.velocity[i] + self.learning_rate * gradients[i]
            params[i] -= self.velocity[i]


class Adam(Optimizer):
    """Adam optimizer with weight decay."""
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, clip_norm=None):
        super().__init__(learning_rate, clip_norm)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def step(self, params, gradients):
        if self.m is None or self.v is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]
        gradients = self.clip_gradients(gradients)
        self.t += 1
        for i in range(len(params)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * gradients[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * gradients[i]**2
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)
            params[i] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)


class LearningRateScheduler:
    """Base class for learning rate schedulers."""
    def __init__(self, initial_lr):
        self.initial_lr = initial_lr

    def get_lr(self, epoch):
        """Get the learning rate for the current epoch."""
        raise NotImplementedError("Derived classes must implement this method.")


class StepDecay(LearningRateScheduler):
    """Step decay learning rate scheduler."""
    def __init__(self, initial_lr, drop_rate, step_size):
        super().__init__(initial_lr)
        self.drop_rate = drop_rate
        self.step_size = step_size

    def get_lr(self, epoch):
        return self.initial_lr * (self.drop_rate ** (epoch // self.step_size))


class ExampleModel:
    """Dummy model for testing the optimizer library."""
    def __init__(self):
        self.params = [np.random.randn(10, 10), np.random.randn(10)]

    def forward(self, x):
        """Dummy forward pass."""
        return x.dot(self.params[0]) + self.params[1]

    def compute_loss(self, preds, targets):
        """Dummy loss function."""
        return np.mean((preds - targets) ** 2)

    def backward(self, x, preds, targets):
        """Dummy gradient computation."""
        grads = [
            x.T.dot(preds - targets) / x.shape[0],
            np.mean(preds - targets, axis=0)
        ]
        return grads


# Streamlit App
def main():
    st.title("Optimizer Visualization and Real-time Training")

    # Sidebar for dataset selection
    st.sidebar.header("Dataset Selection")
    dataset_source = st.sidebar.selectbox(
        "Choose dataset source:",
        ("Preset Dataset", "Upload Dataset", "Provide URL")
    )

    if dataset_source == "Preset Dataset":
        dataset_name = st.sidebar.selectbox("Choose a preset dataset:", ("Synthetic", "Random Gaussian"))
        if dataset_name == "Synthetic":
            x = np.random.randn(100, 10)
            y = np.random.randn(100, 10)
        elif dataset_name == "Random Gaussian":
            x = np.random.normal(0, 1, (100, 10))
            y = np.random.normal(0, 1, (100, 10))

    elif dataset_source == "Upload Dataset":
        uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type="csv")
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            st.write("Data preview:", data.head())
            x = data.iloc[:, :-1].values
            y = data.iloc[:, -1:].values
        else:
            st.warning("Please upload a file.")

    elif dataset_source == "Provide URL":
        dataset_url = st.sidebar.text_input("Enter dataset URL:")
        if dataset_url:
            data = pd.read_csv(dataset_url)
            st.write("Data preview:", data.head())
            x = data.iloc[:, :-1].values
            y = data.iloc[:, -1:].values
        else:
            st.warning("Please provide a valid URL.")

    optimizer_choice = st.sidebar.selectbox("Select Optimizer:", ("SGD", "Momentum", "Adam"))
    learning_rate = st.sidebar.slider("Learning Rate:", min_value=0.0001, max_value=1.0, step=0.0001, value=0.01)

    # Instantiate the selected optimizer
    if optimizer_choice == "SGD":
        optimizer = SGD(learning_rate=learning_rate, clip_norm=5.0)
    elif optimizer_choice == "Momentum":
        momentum = st.sidebar.slider("Momentum:", min_value=0.0, max_value=1.0, step=0.01, value=0.9)
        optimizer = Momentum(learning_rate=learning_rate, momentum=momentum, clip_norm=5.0)
    elif optimizer_choice == "Adam":
        optimizer = Adam(learning_rate=learning_rate, clip_norm=5.0)

    scheduler = StepDecay(initial_lr=learning_rate, drop_rate=0.5, step_size=10)

    model = ExampleModel()

    if st.button("Run Training"):
        for epoch in range(50):
            # Forward pass
            preds = model.forward(x)
            loss = model.compute_loss(preds, y)

            # Backward pass
            gradients = model.backward(x, preds, y)

            # Optimization step
            optimizer.step(model.params, gradients)
            optimizer.log(gradients, model.params, loss)

            # Adjust learning rate
            optimizer.learning_rate = scheduler.get_lr(epoch)
            st.write(f"Epoch {epoch+1}, Loss: {loss:.4f}, LR: {optimizer.learning_rate:.5f}")

        st.write("Training Complete.")
        st.pyplot(optimizer.visualize_logs())

if __name__ == "__main__":
    main()
