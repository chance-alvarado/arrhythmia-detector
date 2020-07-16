# -*- coding: utf-8 -*-
"""Defining functions and classes for arrhythmia detection through ECG signals.

The contents of this module define functions and classes for analyzing,
visualizing, and making predictions based on data from the
MIT-BIH Arrhythmia Database.

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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib import colors
from sklearn.metrics import confusion_matrix
from matplotlib.animation import FuncAnimation
from keras.models import load_model


def create_dataframe(path):
    """Alias of Pandas' read_csv without an additional import."""
    df = pd.read_csv(path, header=None)
    return df


def sample_dataframe(path):
    """Preview 5 rows of DataFrame."""
    df_sample = pd.read_csv(path, nrows=5, header=None)

    return df_sample


class DataVisualization:
    """Class for data exploration through visualization."""

    def plot_setup(self, axs):
        """Set up general plot attributes."""
        # Loop through all axis objects
        for ax in axs:
            # Set facecolor to black
            ax.set_facecolor('k')

            # Remove spines
            ax.spines["top"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_visible(False)

            # Add grid
            ax.grid(linestyle='-', color='w', alpha=.2)

    def save_plot(self, save_location):
        """Save plot based on user's preference."""
        # Try to save plot if user speciefies a save location
        if save_location:
            plt.savefig(save_location, facecolor='k')
            plt.close()

        # Else show plot
        else:
            plt.show()

    def label_counts(self, df):
        """Create vectors of unique labels and their counts."""
        # Find target column
        target = df.iloc[:, -1]

        # Unique labels
        unique_labels = target.unique()

        # Count number of unique occurances for each label
        unique_count = []
        for label in unique_labels:
            unique_count.append(target[target == label].count())

        return unique_labels, unique_count

    def class_bar(self, df, save_location=None):
        """Create bar chart for showing classs balance."""
        # Collect necessary data
        unique_labels, unique_count = self.label_counts(df)

        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(7, 4), facecolor='k')

        # General plot setup
        self.plot_setup([ax])

        # Title
        fig.suptitle('Arrhythmia Type Breakdown', c='w', fontsize=18, y=.95)

        # Set proper color
        ax.tick_params(colors='w')

        # Add x label
        ax.set_xlabel('Arrhythmia Type', c='w', fontsize=14, alpha=0.8)

        # Change scale of y
        ax.set_yticks(np.arange(0, sum(unique_count),
                                sum(unique_count)/10)
                      )

        # Plot with glow
        ax.bar(unique_labels, unique_count, width=.9, color='r', alpha=0.75)
        ax.bar(unique_labels, unique_count, width=.93, color='r', alpha=0.4)
        ax.bar(unique_labels, unique_count, width=.95, color='w', alpha=0.2)

        # Save plot
        self.save_plot(save_location)

    def ecg_scatter(self, df, save_location=None):
        """Create scatter plot of 100 of each type of arrhythmia."""
        # Collect necessary data
        unique_labels, _ = self.label_counts(df)
        target_vect = df.iloc[:, -1]

        # Create figure
        fig, axs = plt.subplots(nrows=5, ncols=1, figsize=(8, 12),
                                facecolor='k'
                                )

        # General plot setup
        self.plot_setup(axs)

        # Add title
        fig.suptitle('Averaged ECG Signals', c='w', fontsize=16, y=0.92)

        # Iterate through all labels
        for col, label in enumerate(unique_labels):

            # Plot text box with arrhythmia type
            axs[col].text(df.shape[1], .95,
                          ('Arrhythmia Type: %s' % (str(int(label)))),
                          size=14, ha="right", va="top", c='w',
                          bbox=dict(boxstyle="round", ec=(1., 0.5, 0.5),
                                    fc='r', alpha=.7
                                    )
                          )

            # Scatter plot for arrhythmia
            matching_rows = (target_vect == label)
            for i in range(100):
                # Dataframe of only relevant rows
                temp_df = df.iloc[:, :-1][matching_rows].round(decimals=1)

                # Data to plot
                data = temp_df.iloc[i, :]
                t_span = range(len(data))

                # Plot iteration
                axs[col].scatter(t_span, data, alpha=0.05, c='r', s=2)

        # Save plot
        self.save_plot(save_location)

    def ecg_line(self, row, viz_type='static', save_location=None):
        """Create a line plot of an individual ecg signal."""
        # Get relevant data
        signal = row[:-1]
        target = row.iloc[-1]

        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(7, 3),
                               facecolor='k')

        # Create title
        fig.suptitle('ECG Signal',
                     fontsize=18,
                     color='white',
                     )

        # General plot setup
        self.plot_setup([ax])

        # Hide tick labels
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        # Add titles
        ax.set_xlabel('Time', c='w', fontsize=14, alpha=0.8)
        ax.set_ylabel('Amplitude', c='w', fontsize=14, alpha=0.8,)

        # Plot text box with arrhythmia type
        plt.text(len(signal), .95,
                 ('Arrhythmia Type: %s' % (str(int(target)))),
                 size=14, ha="right", va="top", c='w',
                 bbox=dict(boxstyle="round",
                           ec=(1., 0.5, 0.5),
                           fc='r',
                           alpha=.7
                           )
                 )

        # Check type
        if viz_type == 'static':
            # Plot with subtle glow effect
            ax.plot(signal, color='r', linewidth=2, alpha=.7)
            ax.plot(signal, color='r', linewidth=3, alpha=.4)
            ax.plot(signal, color='w', linewidth=5, alpha=.2)

            # Save plot
            self.save_plot(save_location)

        # Check type
        elif viz_type == 'dynamic':
            # Time vector
            time_vect = list(range(len(signal)))

            # Create line objects
            line, = ax.plot(time_vect, signal, color='r',
                            linewidth=2, alpha=.7
                            )
            line_g1, = ax.plot(time_vect, signal, color='r',
                               linewidth=3, alpha=.4
                               )
            line_g2, = ax.plot(time_vect, signal, color='w',
                               linewidth=5, alpha=.2
                               )

            # Update function
            def update(num, time_vect, signal, line):
                """Define function to update plot every frame."""
                # Scaling value
                scaling_factor = 10
                end = num*scaling_factor

                if end > 100:
                    start = end-100
                else:
                    start = 0

                for line_obj in [line, line_g1, line_g2]:
                    line_obj.set_data(time_vect[start:end],
                                      signal[start:end]
                                      )

                return [(line,), (line_g1,), (line_g2,)]

            # Create animation
            anim = FuncAnimation(fig, update, interval=40, frames=40,
                                 fargs=[time_vect, signal, line]
                                 )

            # Save animation
            anim.save(save_location, writer='imagemagick', fps=20,
                      savefig_kwargs={'facecolor': 'k', 'transparent': True})
            plt.close()


class DataProcessing:
    """Class for processing ecg data before training model."""

    def resample(self, num_samples, df):
        """Resample data to have 'num_samples' of each label."""
        # New DataFrame
        df_resample = pd.DataFrame()

        # Define target vector
        target = df.iloc[:, -1]

        # Resample for each unique value in target
        for t in target.unique():
            temp_df = df[target == t].sample(num_samples, replace=True)
            df_resample = pd.concat([df_resample, temp_df], ignore_index=True)

        return df_resample

    def shuffle(self, df):
        """Randomly shuffle data."""
        df = df.sample(frac=1).reset_index(drop=True)

        return df

    def add_noise(self, df, noise_level=0.05):
        """Add normal noise with standard deviation 'noise_level'."""
        # Get shape
        rows, cols = df.shape

        # Iterate through rows
        for index in range(rows):
            # Create new noise
            noise = np.random.normal(0, 0.05, cols-1)
            noise = np.append(noise, 0.)

            # Add noise
            df.iloc[index, :] += noise

            # Keep all values between 0 and 1
            for ind, val in enumerate(df.iloc[index, :-1]):
                if val > 1:
                    df.iloc[index, ind] = 1
                elif val < 0:
                    df.iloc[index, ind] = 0

        return df

    def feature_target_split(self, df):
        """Split DataFrame intto a feature matrix and target vector."""
        feature_mat = df.iloc[:, :-1].to_numpy()
        target_vect = df.iloc[:, -1].to_numpy()

        return feature_mat, target_vect

    def one_hot_encoder(self, vect):
        """One hot encode categorical numerical values given Pandas Series."""
        # New target list
        target_vect_enc = []

        # Number of columns in encoded vector
        num_cols = len(np.unique(vect))

        # Iterate through each value in vector
        for val in vect:
            # Create vector to append
            bin_vect = np.zeros(num_cols)
            bin_vect[int(val)] = 1

            # Append
            target_vect_enc.append(bin_vect)

        return np.array(target_vect_enc)


class ModelEvaluation:
    """Class for evaluation of predictive model's metrics."""

    def undo_encode(self, vect):
        """Undo one hot encoding used in training and predictions."""
        # New target list
        unencoded_target_vect = []

        # Add array index to list
        for val in vect:
            unencoded_target_vect.append(np.argmax(val))

        return unencoded_target_vect

    def import_best_model(self):
        """Import best model saved in directory."""
        model = load_model('resources/model/best_model.h5')

        return model

    def best_parameters(self, model):
        """Print the best parameters for each layer of model."""
        # Get configuration json
        config = model.get_config()

        # Iterate through all layers and print relevant info
        for layer in config['layers']:
            layer_type = layer['class_name']
            if layer_type == 'Dense':
                print('Dense Layer Nodes: %d' % (layer['config']['units']))
            elif layer_type == 'Dropout':
                print('Dropout Rate: %d' % (layer['config']['rate']))
            elif layer_type == 'InputLayer':
                print('Input Layer Nodes: %d'
                      % (layer['config']['batch_input_shape'][1])
                      )

    def evaluate_model(self, model, test_X, test_y):
        """Evaluate model on the test data."""
        acc = model.evaluate(test_X, test_y, verbose=0)[1]
        print('Accuracy on testing data: ', acc)

    def plot_confusion_matrix(self, model, test_X, y_true):
        """Plot confusion matrix with custom colormap."""
        # List of target labels
        labels = [0, 1, 2, 3, 4]

        # Make predictions
        y_pred = model.predict(test_X)

        # Unencode target vector
        y_pred = self.undo_encode(y_pred)

        # Get number of samples
        num_samples = len(y_pred)

        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Normalize confusion matrix and round
        cm_norm = np.zeros(shape=cm.shape)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                val = round((cm[i][j] / num_samples), ndigits=2)
                cm_norm[i, j] = val

        # Create figure
        fig, ax = plt.subplots(facecolor='k', figsize=(7, 6))

        # Create black to red color gradient
        # Thanks to SpghttCd on stackoverflow for this code
        def NonLinCdict(steps, hexcol_array):
            cdict = {'red': (), 'green': (), 'blue': ()}
            for s, hexcol in zip(steps, hexcol_array):
                rgb = colors.hex2color(hexcol)

                cdict['red'] = cdict['red'] + ((s, rgb[0], rgb[0]),)
                cdict['green'] = cdict['green'] + ((s, rgb[1], rgb[1]),)
                cdict['blue'] = cdict['blue'] + ((s, rgb[2], rgb[2]),)

            return cdict

        hc = ['#000000', '#5b0000', '#ac0000', '#c80000', '#ff0000']
        th = [0, 0.01, 0.03, 0.05, 1]

        cdict = NonLinCdict(th, hc)
        black_red_cmap = colors.LinearSegmentedColormap('black_red_cmap',
                                                        cdict
                                                        )

        # Plot
        sns.heatmap(cm_norm, annot=True, cmap=black_red_cmap,
                    ax=ax, fmt="g", cbar=False,
                    annot_kws={"size": 14},
                    linewidths=1, linecolor='w'
                    )

        # Add suptitle
        fig.suptitle('Confusion Matrix', c='w', y=.95, fontsize=18)

        # Set axis labels
        ax.set_xlabel('Predicted Arrhythmia Type', fontsize=14, c='w')
        ax.set_ylabel('Actual Arrhythmia Type', fontsize=14, c='w')

        # Set tick parameters
        ax.tick_params(axis='both', which='major', labelsize=12, )
        ax.set_xticklabels(labels=labels, color='w')
        ax.set_yticklabels(labels=labels, color='w', rotation=0)

        # Show plot
        plt.show()
