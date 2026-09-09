# Normalization in Neural Networks

This project was developed as part of COMPSCI 309, Principles of Machine Learning. The goal is to study how different normalization strategies affect training dynamics, convergence, and final performance in neural networks for image classification.

Instead of using a CNN, this project uses a simple multilayer perceptron (MLP) to isolate the effect of normalization itself. That keeps the architecture intentionally straightforward, so the comparison is focused on how normalization changes optimization behavior rather than on a more complex vision model.

## Project Objective

The project compares three training setups on the CIFAR-10 dataset:

- No normalization
- Layer Normalization
- RMS Normalization

The goal is to observe how normalization affects:

- training stability
- learning speed
- final validation accuracy
- activation distribution and optimization behavior

## Dataset

The experiments use the CIFAR-10 dataset from torchvision.

- 10 classes
- 50,000 training images
- 10,000 test images
- each image is converted to a tensor with shape (3, 32, 32)

The data loader is configured with:

- batch size: 64
- shuffle enabled for training data
- standard PyTorch `DataLoader`

## Model Architecture

The network is a fully connected MLP rather than a convolutional network.

The implemented architecture is:

1. Flatten input image: 3 × 32 × 32 → 3072 features
2. Hidden layer 1: 3072 → 128
3. Hidden layer 2: 128 → 128
4. Hidden layer 3: 128 → 128
5. Hidden layer 4: 128 → 128
6. Hidden layer 5: 128 → 128
7. Output layer: 128 → 10 classes

Each hidden layer applies:

- linear transformation
- optional normalization (`LayerNorm` or `RMSNorm`)
- ReLU activation

The model is implemented in the notebook as `ClassiModel`, with a configurable `norm` argument:

- `None` for no normalization
- `"layer"` for `nn.LayerNorm`
- `"rms"` for `nn.RMSNorm`

## Normalization Variants

### 1. NoNorm

This version uses plain linear layers with ReLU and no normalization between layers. This is useful as a baseline to show the effect of unnormalized activations.

### 2. LayerNorm

Layer normalization normalizes activations across features for each sample. In this setup, it is applied independently to each hidden layer.

### 3. RMSNorm

RMS normalization scales the activations using the root mean square of the input vector, which can help stabilize training without requiring full mean/variance normalization.

## Training Setup

The training loop includes:

- cross-entropy loss
- SGD optimizer with learning rate `1e-3`
- 20 training epochs
- train/validation evaluation each epoch
- TensorBoard logging for loss and accuracy

The notebook saves logs under:

- `runs/CIFAR10/NoNorm`
- `runs/CIFAR10/LayerNorm`
- `runs/CIFAR10/RMSNorm`

## Observed Results

From the recorded training logs in the notebook, the final validation accuracies after 20 epochs were:

| Model | Final Validation Accuracy | Final Validation Loss |
| --- | ---: | ---: |
| NoNorm | 0.1000 | 2.3027 |
| LayerNorm | 0.3120 | 1.9989 |
| RMSNorm | 0.1727 | 2.2957 |

This suggests that under this architecture and training setup, `LayerNorm` gives the strongest improvement over the baseline, while `RMSNorm` is better than no normalization but still underperforms LayerNorm.

## Interpretation

The experiment illustrates a key idea in deep learning: without normalization, activations can drift and training can become unstable or plateau. In this project, the effect is visible in the poor performance of the unnormalized model. Normalization helps keep activations in a healthier range, which improves optimization and final accuracy.

The main takeaway is that even a simple MLP can benefit noticeably from normalization when used on a moderately difficult image classification problem.

## Files in the Project

- `class.ipynb` — full experiment and training code
- `data/` — CIFAR-10 dataset cache
- `runs/` — TensorBoard event logs for each model variant

## How to Run

Open the notebook in Jupyter or VS Code and run the cells in order:

1. load CIFAR-10
2. build the data loaders
3. define the model
4. train each normalization variant
5. compare results

A typical flow is:

```python
model = ClassiModel(
    in_dims=3*32*32,
    hid_dim=128,
    out_dims=10,
    n_hidden=5,
    norm="layer"
)
```

Then train with the provided `train()` function and monitor the logged metrics.

## Conclusion

This project shows that normalization is not just a minor implementation detail—it plays a significant role in learning dynamics. Even for a simple MLP, the difference between no normalization and normalized training is substantial, making this an effective teaching example for understanding the practical value of normalization in neural networks.

---

This notebook is a compact experimental study designed to compare normalization strategies in a controlled setting, with the main focus on understanding their impact on training behavior and model performance.

