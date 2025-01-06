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
   streamlit run app.py
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
