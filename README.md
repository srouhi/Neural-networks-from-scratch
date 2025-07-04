# TensorFlow vs NumPy Performance and MNIST Classification

This project explores the performance difference between TensorFlow and NumPy for vector operations and implements two feedforward neural networks using TensorFlow on the MNIST and Fashion MNIST datasets.

---

## Features

- **Performance Comparison**: Compare element-wise vector multiplication through TensorFlow vs NumPy with timing benchmarks.
- **Tensor Operations**: Perform matrix operations using TensorFlow.
- **Tensor <-> NumPy Conversion**: Demonstrate how to convert between TensorFlow tensors and NumPy arrays.
- **MNIST Digit Classification**: Train a simple neural network on MNIST.
- **Fashion MNIST Classification**: Train a deeper neural network on the Fashion MNIST dataset to achieve >90% accuracy.

---

## Project Structure

```bash
tensorflow-vs-numpy-perf-mnist/
├── notebook.ipynb           # Jupyter Notebook version (recommended for walkthrough + visuals)
├── script.py                # Python script version of the project
├── README.md                # Project overview and usage
└── requirements.txt         # Python dependencies (optional but recommended)
```

---

## Model Architectures

### MNIST Model
- `784 → 100 → 10`
- Activation: ReLU in hidden, Softmax in output
- Loss: MSE
- Optimizer: SGD

### Fashion MNIST Model
- `784 → 128 → 64 → 32 → 10`
- Activation: ReLU (all hidden), Softmax (output)
- Optimizer: SGD
- Loss: MSE
- Epochs: 300
- **Final Training Accuracy**: ~92.3%

---

## Sample Visualizations

The project displays the first few images in both MNIST and Fashion MNIST, along with their labels using `matplotlib`.

---
## Author

**Shay - Shaghayegh Rouhi**  
Data Science | AI/ML Development | NLP Applications  
[LinkedIn Profile](https://www.linkedin.com/in/Shay-shaghayegh-rouhi-aba3892a1)
