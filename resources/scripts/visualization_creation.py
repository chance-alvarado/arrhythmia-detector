# -*- coding: utf-8 -*-
"""Create and save visualizationss for data presentation.

This script creates static and dynamic visualizations of
ecg data found in the MIT-BIH Arrhythmia Database. Visualizations
are then saved for later interpretation and use.


Explore this repository at:
    https://github.com/chance-alvarado/arrhythmia-detector

Author:
    Chance Alvarado
        LinkedIn: https://www.linkedin.com/in/chance-alvarado/
        GitHub: https://github.com/chance-alvarado/
"""
import arrhythmia_analysis

# Data path
train_dir = '../data/mitbih_train.csv'

# Create DataFrame
df = arrhythmia_analysis.create_dataframe(train_dir)

# Create visualization object
DataVisualization = arrhythmia_analysis.DataVisualization()

# Create processing object
DataProcessing = arrhythmia_analysis.DataProcessing()

# Create class bar graph
DataVisualization.class_bar(df, save_location='../plots/class_bar.jpg')

# Create gifs of each type of arrhythmia
# Target vector
target_vect = df.iloc[:, -1]

# Iterate through each unique label
for label in target_vect.unique():
    # Parse title of visualization
    save_location = ('../plots/ecg_signal_%d.gif' % (int(label)))

    # Grab row
    index = df[target_vect == label].index[0]
    row = df.iloc[index, :]

    # Create and save visualizations
    DataVisualization.ecg_line(row, viz_type='dynamic',
                               save_location=save_location
                               )

# Create static signal of pre and post noise signal
signal = df.sample(n=1)

# Create visualizations
DataVisualization.ecg_line(signal.iloc[0, :],
                           save_location='../plots/ecg_no_noise.jpg'
                           )

# Add noise
signal_noise = DataProcessing.add_noise(signal, noise_level=0.05)

# Create visualizations
DataVisualization.ecg_line(signal_noise.iloc[0, :],
                           save_location='../plots/ecg_noise.jpg'
                           )

# Create scatter plot of each arrhythmia type
DataVisualization.ecg_scatter(df, save_location='../plots/ecg_scatter.jpg')

print('All plots successfully created.')
