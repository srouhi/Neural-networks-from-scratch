# Neural Networks From Scratch & With TensorFlow

This repository contains two related machine learning projects focused on training feedforward neural networks (FNNs) for both regression and classification tasks.

---

## Project Structure

### `home_price_regression/`
Implements a feedforward neural network **from scratch in NumPy** to predict house prices based on real estate features from Lincoln, Nebraska.

- ReLU activation and Mean Squared Error loss
- Custom backpropagation and training loop
- Mini-batch Stochastic Gradient Descent (SGD)

### `mnist_fashion_classification/`
Trains deep neural networks on the MNIST and Fashion-MNIST datasets using **TensorFlow/Keras**.

- Uses ReLU + Softmax activation layers
- Compares optimizers: SGD vs Adam
- Monitors validation loss to avoid overfitting

---

## Datasets Used

- **Lincoln Home Sales**: Structured data (CSV) with 16 features + price.
- **MNIST**: Handwritten digit classification (0–9)
- **Fashion MNIST**: 10-class clothing image dataset

---

## Sample Results

| Task | Dataset | Final Accuracy |
|------|---------|----------------|
| Regression | Lincoln Home Sales | ~85% near-accurate within tolerance |
| Classification | MNIST | ~98.2% (Adam) |
| Classification | Fashion MNIST | ~89.4% (Adam, 7 epochs) |

---

## 🛠Technologies Used

- Python
- NumPy (for from-scratch FNN)
- TensorFlow & Keras (for MNIST/Fashion)
- Matplotlib (for visualizations)

---

## How to Run

1. Clone the repository.
2. Open each `.ipynb` in Jupyter or VS Code.
3. For `home_price_nn.ipynb`, make sure `LincolnHomeSales.csv` is in the same folder.
4. Run all cells sequentially to see the training process.

---

## Learnings

- Implementing neural networks from scratch deepened understanding of forward/backward passes.
- Loss functions and optimizer choices have significant effects on convergence.
- Validation loss is essential to detect overfitting.

---

**Shay - Shaghayegh Rouhi**  
Data Science, Machine Learning, AI  
[LinkedIn](https://www.linkedin.com/in/Shay-shaghayegh-rouhi-aba3892a1) | [Email](mailto:Shaghayegh.rouhi.sr@gmail.com)


