# Teo_BackProp_DIY
A toy implementation of backpropagation on a simple MLP. By Teo Asinari

## Documentation
It is a simple MLP with 3 layers. The first two are linear with ReLU activation (2 neurons each) and the last is a single neuron sigmoid layer. 
It is being trained on a simplified 2-D version of the iris dataset, found here data/iris2d.csv.

The full derivation of the backward pass updates is explained in docs/models/MultiLayerPerceptron/BackpropagationDerivation.pdf.

## How to run
I ran on Python 3.8.10 on WSL 2.0 (Windows Subsystem For Linux) using Pycharm. Assuming you're in a POSIX terminal with python 3.8.10 or higher, you ought to be able to just run:
<your_python_executable> <path to BasicMLP_Iris2D_Sandbox\.py>
