# -*- coding: utf-8 -*-
"""Training a dense neural network to detect arrhythmias from ecg signals.

This script uses funstions and classes from 'arrhythmia_analysis.py'
to import and process ecg signal data. A dense neural network with
optimized hyperparamters is trained and saved for further use.


Explore this repository at:
    https://github.com/chance-alvarado/arrhythmia-detector

Author:
    Chance Alvarado
        LinkedIn: https://www.linkedin.com/in/chance-alvarado/
        GitHub: https://github.com/chance-alvarado/
"""
# Set random seeds for reproducability
from numpy.random import seed
seed(1)
import tensorflow
tensorflow.random.set_seed(2)

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.utils import plot_model

import itertools
import numpy as np
import matplotlib.pyplot as plt

import arrhythmia_analysis

# Data path
train_dir = '../data/mitbih_train.csv'

# Create DataFrame
df = arrhythmia_analysis.create_dataframe(train_dir)

# Create DataProcessing object
DataProcessing = arrhythmia_analysis.DataProcessing()

# Resample the data to have 20000 of each class
df_resample = DataProcessing.resample(20000, df)

# Randomly shuffle the dataset
df_resample = DataProcessing.shuffle(df_resample)

# Add noise to features
df_resample_noise = DataProcessing.add_noise(df_resample)

# Split our data into the feature matrix and target vector
feature_mat, target_vect = DataProcessing \
    .feature_target_split(df_resample_noise)

# One hot encode target vector
target_vect_enc = DataProcessing.one_hot_encoder(target_vect)

# Create callback object to stop training
early_stopper = EarlyStopping(monitor='val_accuracy', verbose=0,
                              patience=2,
                              )

# Print message for user
print('Data has been processed. Model construction has begun. \n')


# Define function for model creation and training
def model_shell(layer_units, dropout_1_rate, dropout_2_rate, final_eval=False):
    # Build model
    model = Sequential()
    model.add(Dense(layer_units[0], activation='relu',
                    input_shape=(187,),
                    kernel_initializer='random_uniform',
                    bias_initializer='zeros')
              )
    model.add(Dropout(dropout_1_rate))
    model.add(Dense(layer_units[1], activation='relu'))
    model.add(Dropout(dropout_2_rate))
    model.add(Dense(5, activation='softmax'))

    # Compile model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy']
                  )

    # If final model evaluation, let run for 100 epochs
    if final_eval:
        callbacks_list = []
    else:
        callbacks_list = [early_stopper]

    # Fit model
    history = model.fit(feature_mat, target_vect_enc,
                        validation_split=0.3,
                        callbacks=callbacks_list,
                        verbose=0,
                        epochs=100
                        )

    return history, model


# Specify list of hyperparamters to tune
# Layer combinations
layer_sequences = [
    (128, 96),
    (128, 128),
    (216, 128)
]

# Dropout rates
dropout_1_list = [0, 0.25, 0.5]
dropout_2_list = [0, 0.25, 0.5]

# Parameter storage
max_val_acc = 0

# Iterate through permuted paramters
for layer_units, dropout_1_rate, dropout_2_rate \
        in itertools.product(layer_sequences, dropout_1_list, dropout_2_list):

    # Print user message
    print('Testing: \n',
          'layer_units: ', layer_units, '\n',
          'dropout_1_rate :', dropout_1_rate, '\n',
          'droupout_2_rate: ', dropout_2_rate
          )

    # Evaluate model with current hyperparamters
    history, _ = model_shell(layer_units, dropout_1_rate, dropout_2_rate)

    # Current accuracy
    current_val_acc = max(history.history['val_accuracy'])

    # Print user message
    print('Validation accuracy: ', current_val_acc, '\n')

    # Update current best
    if current_val_acc > max_val_acc:
        max_val_acc = current_val_acc

        # Fetch number of epochs the model has run for
        epochs = len(history.history['val_accuracy'])

        # Store parameters
        best_params = {
            'layer_units': layer_units,
            'dropout_1_rate': dropout_1_rate,
            'dropout_2_rate': dropout_2_rate}

        # Store scores
        best_scores = {
            'loss': history.history['loss'][epochs - 1],
            'accuracy': history.history['accuracy'][epochs - 1],
            'val_loss': history.history['val_loss'][epochs - 1],
            'val_accuracy': history.history['val_accuracy'][epochs - 1]}

# Print summary of model with best parameters
print('Best model summary:')
print(best_params)
print(best_scores, '\n')

# Print user maggeage
print('Retraining with optimized parameters. \n')

# Retrain model with best hyperparamters
history, model = model_shell(**best_params, final_eval=True)

# Print user message
print('Training complete. Saving model and visualizing training data.')

# Retrieve metrics from history object
val_loss = history.history['val_loss']
val_acc = history.history['val_accuracy']

# Plot model architecture
plot_model(model, to_file='../plots/model_architecture.png', 
           show_shapes=True, show_layer_names=False, rankdir='LR')

# Plot final training metrics
# Create figure
fig, ax = plt.subplots(1, 1, figsize=(8, 5), facecolor='k')

# Create tick lists
x_ticks = np.linspace(0, 100, 11)
x_labels = [1, 10, 20, 30, 40, 50, 60,
            70, 80, 90, 100]

y_ticks = np.linspace(0, 1., 11)
y_labels = [round(val, 2) for val in y_ticks]

# Set suptitle
fig.suptitle('Validation Metrics vs. Epochs', fontsize=18, c='w',
             y=0.95
             )

# Set facecolor to black
ax.set_facecolor('k')

# Plot with subtle glow effect
main_line_l = ax.plot(val_loss, color='r', linewidth=2, alpha=.7)
ax.plot(val_loss, color='r', linewidth=3, alpha=.4)
ax.plot(val_loss, color='w', linewidth=5, alpha=.2)

main_line_a = ax.plot(val_acc, color='b', linewidth=2, alpha=.7)
ax.plot(val_acc, color='b', linewidth=3, alpha=.4)
ax.plot(val_acc, color='w', linewidth=5, alpha=.2)

# Set axis labels
ax.set_xlabel('Epochs', fontsize=14, c='w')
ax.set_ylabel('Accuracy / Loss', fontsize=14, c='w')

# Set axis ticks
ax.set_xticks(ticks=x_ticks)
ax.set_yticks(ticks=y_ticks)

# Set axis tick labels
ax.set_xticklabels(labels=x_labels, color='w')
ax.set_yticklabels(labels=y_labels, color='w')

# Add grid
ax.grid(linestyle='-', color='w', alpha=.3)

# Add legend
leg = ax.legend([main_line_a[0], main_line_l[0]],
                ['Validation Accuracy', 'Validation Loss'],
                facecolor='k', framealpha=1, fancybox=True
                )

# Change legend text color
for text in leg.get_texts():
    plt.setp(text, color='w')

# Save plot
plt.savefig('../plots/model_training.jpg', facecolor='k')
plt.close()

# Save model
model.save('../model/best_model.h5')
