# Optimizer Visualizer

Optimizer Visualizer is a dynamic application that allows users to explore and understand the impact of various optimization techniques on machine learning models in real-time. This tool provides an intuitive interface to load datasets, train models using popular optimizers, and visualize their performance.

## Features

- **Dataset Options:**
  - Load datasets via URL.
  - Upload your own dataset.
  - Choose from preset datasets.
- **Optimizers Supported:**
  - Stochastic Gradient Descent (SGD)
  - Momentum
  - Adam
- **Learning Rate Scheduling:**
  - Step Decay Scheduler
- **Visualizations:**
  - Gradient norms
  - Parameter norms
  - Loss curve
- **Real-Time Feedback:**
  - Watch training progress dynamically.
  - Observe the optimizer’s effect on model parameters and loss.

## Try it Out

Explore the application here: [Optimizer Visualizer](https://optimizervisualizer.streamlit.app/)

## Installation

To run this project locally:

1. Clone the repository:
   ```bash
   git clone https://github.com/RohanSai22/optimizervisualizer
   cd optimizer-visualizer
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Launch the application:
   ```bash
   streamlit run streamlit_app.py
   ```

4. Access the app locally at `http://localhost:8501`.

## Usage

1. **Load a Dataset:**
   - Enter a URL to fetch a dataset.
   - Upload a local CSV file.
   - Select from a list of preloaded datasets.

2. **Choose Optimizer and Scheduler:**
   - Select one of the supported optimizers.
   - Configure learning rate scheduler parameters.

3. **Train the Model:**
   - Start training to visualize real-time metrics.

4. **Analyze Results:**
   - Observe the impact of optimizer settings on model training through interactive plots.
  

Each optimizer is defined by a mathematical update rule that adjusts model parameters (\(\theta\)) to minimize the loss function (\(J(\theta)\)).

1. **SGD**: Basic gradient descent with a constant learning rate.  
   \[
   \theta_{t+1} = \theta_t - \eta \nabla_\theta J(\theta)
   \]

2. **Momentum**: Adds momentum to smooth updates and overcome oscillations.
   \[
   v_t = \beta v_{t-1} + (1 - \beta) \nabla_\theta J(\theta), \quad \theta_{t+1} = \theta_t - \eta v_t
   \]

3. **Nesterov Accelerated Gradient (NAG)**: Looks ahead before calculating the gradient.
   \[
   v_t = \beta v_{t-1} + \eta \nabla_\theta J(\theta - \beta v_{t-1})
   \]

4. **RMSProp**: Scales learning rates based on recent gradient magnitudes.
   \[
   G_t = \beta G_{t-1} + (1 - \beta)(\nabla_\theta J(\theta))^2, \quad \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t + \epsilon}} \nabla_\theta J(\theta)
   \]

5. **AdaGrad**: Adjusts learning rates based on the cumulative gradient history.
   \[
   \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t + \epsilon}} \nabla_\theta J(\theta)
   \]

6. **Adam**: Combines momentum and RMSProp for adaptive learning rates.
   \[
   m_t = \beta_1 m_{t-1} + (1 - \beta_1) \nabla_\theta J(\theta), \quad v_t = \beta_2 v_{t-1} + (1 - \beta_2)(\nabla_\theta J(\theta))^2
   \]
   \[
   \hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}, \quad \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t
   \]

7. **AdamW**: Combines Adam with weight decay to prevent overfitting.
   \[
   \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t - \lambda \theta_t
   \]


## Folder Structure

```
optimizer-visualizer/
├── streamlit_app.py      # Main Streamlit app
├── requirements.txt      # Dependencies
├── README.md             # Project documentation
```

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new branch for your feature or bugfix:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add new feature"
   ```
4. Push to your branch:
   ```bash
   git push origin feature-name
   ```
5. Create a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contact

For questions or feedback, feel free to reach out:

- **Author:** Rohan Sai
- **Email:** maragonirohansai@gmail.com
