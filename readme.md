# Multilayer Perceptron (school 42 project)

## Description
This project involves training a neural network using a dataset, validating the model, and making predictions. The neural network structure and training parameters are defined in a JSON file.
This project presents several challenges, particularly in understanding the mathematical processes involved. Key concepts include:

- **Partial Derivatives**: Essential for calculating gradients in backpropagation.
- **Matrix Multiplication**: Used in forward and backward passes through the network.
- **Cost Functions**: Such as binary cross-entropy, which measures the error of the model.
- **Gradient Descent**: An optimization algorithm used to minimize the cost function by updating the model's parameters.
- **Activation Functions**: Functions like sigmoid, relu, and softmax that introduce non-linearity into the model.

A solid grasp of these concepts is crucial for successfully training and optimizing the neural network.

## Installation
To install the necessary dependencies, run:
```bash
pip install -r requirements.txt
```

## Dataset Separation
Randomly split the dataset into two parts. One for training and one for validation.
```bash
python3 separate.py <dataset.csv>
```

## Program Training
Start the training of the neural network and save its parameters in a JSON file.
```bash
python3 train.py <dataset.csv> <model.json> <json output name> <target column name> <validation dataset.csv>
```

## Prediction Program
Load a dataset to predict, the neural network parameter file, and save the results in a CSV file.
```bash
python3 predict.py <validation_dataset.csv> <modelParameters.json> <csv output name>
```

## Guide to Creating a `network.json` File
This JSON file is used to define the structure of an artificial neural network and the training parameters. Multiple networks can be defined in the file, but only the one specified in the `model_fit` section will be considered.

---

### Structure of the `network.json` File

Here is the general structure of the file:

```json
{
    "network_1": {
        "input_layer": {
            "activation": "default"
        },
        "hidden_layer_1": {
            "neurons": 20,
            "activation": "sigmoid",
            "weights_init": "heUniform"
        },
        "hidden_layer_2": {
            "neurons": 7,
            "activation": "sigmoid",
            "weights_init": "heUniform"
        },
        "output_layer": {
            "activation": "softmax",
            "weights_init": "heUniform"
        }
    },
    "model_fit": {
        "network": "network_1",
        "loss": "binaryCrossEntropy",
        "learning_rate": 0.08,
        "batch_size": 32,
        "epochs": 100
    }
}
```

---

### Element Details

#### 1. Network Definition

A network is defined by several layers:

- **`input_layer`**:
  - `activation`: Specifies the activation function of the input layer (default value "default").

- **Hidden layers (`hidden_layer_X`)**:
  - `neurons`: Number of neurons in the layer.
  - `activation`: Activation function (e.g., "sigmoid", "relu", etc.).
  - `weights_init`: Weight initialization method (e.g., "heUniform").

- **`output_layer`**:
  - `activation`: Activation function (e.g., "softmax").
  - `weights_init`: Weight initialization method.

**Note:** 
1. The number of hidden layers is unlimited and they must be named according to the convention `hidden_layer_1`, `hidden_layer_2`, etc.
2. The different activations possible are : 
"sigmoid", "relu", "tanh", "leakyRelu"
3. Softmax activation only possible for the output layer

#### 2. Training Parameters (`model_fit`)

This section defines the training parameters of the model. The available fields are:

- `network`: The name of the network to use.
- `loss`: Cost function (must be **binaryCrossEntropy**).
- `learning_rate`: Learning rate (e.g., 0.08).
- `batch_size`: Batch size for training (e.g., 32).
- `epochs`: Number of training epochs (e.g., 100).

---

### Important Notes

1. Only the network mentioned in the `model_fit` section will be considered for training.
2. It is possible to define multiple networks, but only one can be used at a time.
3. The only supported cost function is `binaryCrossEntropy`.
4. The number of hidden layers is unlimited, but they must be named consecutively.

---

### Example of Another Network

```json
{
    "network_2": {
        "input_layer": {
            "activation": "default"
        },
        "hidden_layer_1": {
            "neurons": 15,
            "activation": "relu",
            "weights_init": "xavier"
        },
        "hidden_layer_2": {
            "neurons": 10,
            "activation": "tanh",
            "weights_init": "heNormal"
        },
        "output_layer": {
            "activation": "softmax",
            "weights_init": "xavier"
        }
    }
}
```

## Acknowledgments
I would like to thank Ecole 42 for my training, Nicolas Pantano, and Martin Detournay for their help in data science and machine learning.

## License
This project is licensed under the MIT License.
