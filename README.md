# C-CORE-Iceberg-Classifier-Challenge
My solution for [C-CORE Iceberg Classifier Challenge](https://www.kaggle.com/c/statoil-iceberg-classifier-challenge) 
on kaggle.

# Common models properties
##  Keras Callbacks
### EarlyStopping
All models use EarlyStopping callback that stop training if it was not significant improvement (minimum delta = 
1e-3) after 45 epoches. 
### ReduceLROnPlateau
All models use ReduceLROnPlateau callback that reduce learning rate (multiply by 0.3) if it was not 
significant improvement (minimum delta = 5e-3) after 15 epoches.
### TensorBoard
All models use TensorBoard callback for visualizing training results.
### ModelCheckpoint
All models use ModelCheckpoint callback for save best (by validation loss) model. 
## Data augmentation
For data augmentation was used Keras ImageDataGenerator with vertical_flip, width_shift_range and height_shift_range.
## Activations
Activation function change (ReLU -> Leaky-ReLU -> PReLu -> ELU) give significant loss improvement. All models use ELU
 activation (except SqueezeNet that use SeLU activation).
## Optimization
All models used RMSProp optimizer with initial learning rate 1e-3.
## k-Fold validation
All models used 5 fold validation with fixed random seed (0xCAFFE).



# Results table
|Model name|Details|Accuracy|ROC AUC|Logloss|Public LB|
|---|---|---|---|---|---|
|ResNeXt-11|Width=8, Cardinality=4, Noise=1e-2|0.930800 +/- 0.010850|0.980419 +/- 0.005398|0.179810 +/- 0.025678|0.1313|
|ResNet-18|Filters=8|?|0.974473 +/- 0.005401|?|0.1595|
|SqueezeNet|Filters=8|?|0.970117 +/- 0.007891|0.223358 +/- 0.30510|?|
