# C-CORE-Iceberg-Classifier-Challenge
My solution for C-CORE Iceberg Classifier Challenge on kaggle.

## General results table
|Model name|Accuracy (%)|Min accuracy (%)|Max accuracy (%)|Logloss|Min logloss|Max logloss|
|---|---|---|---|---|---|---|
|Base model|86.66 +/- 1.67|84.11|88.16|0.276012 +/- 0.016718|0.247729|0.297221|

## Common models properties
All models use common callbacks: EarlyStopping, ReduceLROnPlateau. 

## Base model
AlexNet-like architecture (for more details see [realiztion](./Research/base_model.py)).

|Property|Value|
|---|---|
|Optimizer|SGD(lr=0.001, momentum=0.9)|
|Weight decay|1e-6|
