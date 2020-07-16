# arrhythmia-detector

Training a dense neural network on ECG signals to detect cardiac arrhythmias.

**View the properly rendered notebook with *nbviewer* [here](https://nbviewer.jupyter.org/github/chance-alvarado/dementia-classifier/blob/master/arrhythmia_detector.ipynb).**

---

## Introduction

Cardiac arrhythmias occur when the electrical impulses of the heart don't function properly. These irregular impulses may manifest themselves as anxiety, fatigue, and dizziness. While some cases of cardiac arrhythmias are relatively harmless, others are indicative of learger issues including high blood pressure, diabetes, and heart attacks. An electrocardiogram (ECG or EKG) is a tool medical professionals can use to visualize the electrical workings of the heart. Analsyis of these ECG signals accomponied by other testing measures allows for the diagnosis of cardiac problems such as arrhythmias. Quickly and accurately detecting cardiac arrhythmias through analyzing ECG signals would allow healthcare professionals to better apply appropriate intervention techniques

---

## Emphasis

This project emphasizes the following skills:

- Creating static and dynamic signal visualizations using *Matplotlib* and *Seaborn*.
- Process large, high-dimensionality datasets for machine learning applications using *Pandas* and *Numpy*.
- Develop and tune a deep neural network on a validation set suing *TensorFlow's* *Keras*.
- Evaluate the effectiveness of a neural network using *Scikit-Learn* and *Keras*.
- Providing instructions for easily reproducable results.


---

## Prerequisites

This repository is written using [Python](https://www.python.org/) v3.7.4 and [Jupyter Notebook](https://jupyter-notebook.readthedocs.io/) v6.0.3. 

The following packages are recommended for proper function:

Requirement | Version
------------|--------
[Pandas](https://pandas.pydata.org/) | 1.0.1
[Matplotlib](https://matplotlib.org/) | 3.1.3
[Numpy](https://numpy.org/) | 1.18.1
[Seaborn](https://seaborn.pydata.org/) | 0.10.0
[Scikit-learn](https://scikit-learn.org/) | 0.22.1
[TensorFlow](https://www.tensorflow.org/) | 2.3.0
[Keras](https://keras.io/) | 2.4.3


Installation instructions for these packages can be found in their respective documentation.

---

## Project Struture/Replicating Results

This project has the folowing architecture:
```
arrhythmia-detector
├─ arrhythmia_detector.ipynb
└─ resources
   ├─ scripts
   ├─ model 
   ├─ data
   ├─ images
   └─ plots
```

The results of this analysis (i.e. the model, plots, and metric) have been included for ease of use. However, all scripts neccessary to replicate the results have been included. To validate the results of this analysis do the following:

- Remove `best_model.h5` from the `model` folder
- Remove all plots from the `plots` folder
- In the `scripts` folder execute `model_training.py` and `visualization_creation.py`.
  - This can be done through the terminal as follows:
  ```
  $pwd
  ../resources/scripts
  
  $python model_training.py
  Data has been processed. Model construction has begun. 

  Testing: 
  layer_units:  (128, 96) 
  dropout_1_rate : 0 
  dropout_2_rate:  0
  ...
  
  Training complete. Saving model and visualizing training data.
  
  $python visualization
  All plots successfully created.
  
  ```
  
- Run `arrhythmia_detector.ipynb` in your preferred notebook viewer - [Jupyter Notebook](https://jupyter-notebook.readthedocs.io/) is reccomended. 
 
---

## Data

The foundation of this analysis is built on data collected and processed by the Beth Israel Deaconess Medical Center and MIT. Their MIT-BIH Arrhythmia database has acted as the foundation for many influential cardiac arrhythmia studies.

The data used in this analysis is divided among two csv files:

- `mitbih_test.csv`
  - This dataset contains 87553 ECG signals of a single heartbeat measured at 187 instances. Each instance notes the signal's normalized amplitude ranging between 0 and 1. The signal's respective arrhythmia type is also noted.
- `mitbih_train.csv`
  - This dataset contains 21891 instances of ECG signals in the same fashion as `mitbih_train`.


More information about the data used in this analysis can be found [here](https://www.physionet.org/content/mitdb/1.0.0/).

- Acknowledgments:
  - Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database. IEEE Eng in Med and Biol 20(3):45-50 (May-June 2001). (PMID: 11446209)
  - Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215–e220.

---

## Cloning

Clone this repository to your computer [here](https://github.com/chance-alvarado/arrhythmia-detector/).

---

## Author

- **Chance Alvarado** 
    - [LinkedIn](https://www.linkedin.com/in/chance-alvarado/)
    - [GitHub](https://github.com/chance-alvarado/)

---

## License

- **[MIT license](http://opensource.org/licenses/mit-license.php)**
