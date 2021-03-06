# Classifying Circles & Squares

This project contains implementations of several machine learning methods for classifying a dataset of noisy input images as either circles or squares. Each method outputs  testing accuracy to stdout

| [![Square](https://raw.githubusercontent.com/nicrobertson14/CircleSquareClassifiers/master/img-data/Square.png)](Square) |
[![Circle](https://raw.githubusercontent.com/nicrobertson14/CircleSquareClassifiers/master/img-data/Circle.png)](Circle) |
|:------:|:------:|
| Square | Circle |

## Dataset
The image dataset contains 200 training images and 200 test images

## Dependencies
CNN.py requires keras, tensorflow packages

## Usage
To build a kd-tree and perform 'k' nearest neighbour search simply run ```python src/knn.py k``` where 0 < k < 200

To build decision trees run ```python src/DT.py ```

To build a convolutional neural network run ```python src/CNN.py```

