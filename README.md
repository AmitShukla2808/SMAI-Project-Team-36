# Table Tennis Performance Analysis System

This project aims to develop a comprehensive system for table tennis performance analysis, moving beyond traditional subjective evaluations by leveraging both sensor data (from the TTSWING dataset) and deep learning for trajectory prediction. The system focuses on providing objective, data-driven insights for players and coaches.

## Table of Contents
1.  [Project Overview](#project-overview)
2.  [Analytical Components](#analytical-components)
3.  [Datasets](#datasets)
4.  [Data Preprocessing Overview](#data-preprocessing-overview)
5.  [Methodology & Experiments](#methodology--experiments)
6.  [Repository Structure (Jupyter Notebooks)](#repository-structure-jupyter-notebooks)
7.  [Setup Instructions](#setup-instructions)
8.  [Running the Code (Notebooks)](#running-the-code-notebooks)
    *   [Training Models](#training-models)
    *   [Evaluating Models/Outputs](#evaluating-modelsoutputs)
    *   [Making Inferences/Generating Outputs](#making-inferencesgenerating-outputs)
9.  [Model Weights & Architectures](#model-weights--architectures)
10. [Detailed Results, Discussion & Conclusion](#detailed-results-discussion--conclusion)

## 1. Project Overview

Table tennis lacks advanced, data-driven tools for performance analysis common in other sports. Manual evaluation is often subjective. This project aims to bridge this gap by developing modules for:
*   **Ball Trajectory Prediction using LSTM** (Based on initial launch parameters from HDF5 data)
*   **Player Profiling from Swing Data** (Using clustering on TTSWING sensor data)
*   **Stroke Power Prediction** (Using regression on TTSWING sensor data)
*   **Bad Stroke Identification** (Using anomaly detection on TTSWING sensor data)
*   **Synthetic Swing Generation** (Using VAEs on TTSWING sensor data)

The goal is to enable objective feedback, explore novel integrations of ML/DL techniques, and demonstrate potential applications in training, simulation, and scouting.

## 2. Analytical Components

This project is comprised of the following key analytical modules:

1.  **Table Tennis Ball Trajectory Prediction using LSTM (Implemented):**
    *   **Goal:** To predict the 3D trajectory of a table tennis ball based on its initial launch parameters using an LSTM network.
    *   **Methodology:** Involves loading trajectory data from HDF5 files, preprocessing (padding variable length sequences), building an LSTM-based sequence generation model, training, and evaluating using MSE, MAE, and RMSE.
    *   **Code Reference:** `Ball_trajectory_prediction.ipynb` 

2.  **Player Profiling from Swing Data (Implemented):**
    *   **Goal:** To identify distinct player swing styles or stroke characteristics from sensor data using unsupervised clustering.
    *   **Methodology (TTSWING Data):** Involves preprocessing of motion features, PCA for dimensionality reduction/visualization, and application of clustering algorithms (K-Means, DBSCAN, HDBSCAN).
    *   **Code Reference:** `player_profiling_clustering.ipynb`

3.  **Stroke Power Prediction (Implemented):**
    *   **Goal:** To predict the power of a table tennis stroke based on sensor and demographic data.
    *   **Methodology (TTSWING Data):** Involves engineering a 'power' target variable, extensive preprocessing, and training various regression models.
    *   **Code Reference:** `Stroke_Power_Prediction.ipynb`

4.  **Bad Stroke Identification (Implemented):**
    *   **Goal:** To identify anomalous or potentially "bad" strokes from swing sensor data.
    *   **Methodology (TTSWING Data):** Uses unsupervised anomaly detection techniques (Isolation Forest, One-Class SVM, Local Outlier Factor) on preprocessed motion features.
    *   **Code Reference:** `Bad_Stroke_Finder.ipynb`

5.  **Synthetic Swing Generation (VAE) (Implemented):**
    *   **Goal:** To generate new, realistic synthetic swing data samples by learning the underlying distribution of existing swing data.
    *   **Methodology (TTSWING Data):** Employs a Variational Autoencoder (VAE) trained on statistical motion features.
    *   **Code Reference:** `synthetic-swing-generation-with-vae.ipynb`

## 3. Datasets

*   **Ball Trajectory Dataset (HDF5):**
    *   **Description:** Used for Component 1 (Ball Trajectory Prediction). Data is in HDF5 format. Each file contains groups (within `/originals` path) representing a single trajectory, including `launch_param` (5 launch parameters) and `positions` (sequence of 3D coordinates [x, y, z]).
    *   **Source/Access:** Update `BASE_DATA_PATH` and `SUBDIRECTORIES_TO_PROCESS` variables in the `ball_trajectory_prediction_lstm.ipynb` notebook to point to your data location (e.g., Google Drive path if using Colab).
    *   **Link to  Dataset ):** `[[Insert Google Drive/Kaggle Link to HDF5 Trajectory Data Here]](https://drive.google.com/drive/folders/1vyXtYarvPisPjX19c3P1k2W7U1b__Cc9?usp=sharing)`
*   **TTSWING Dataset (CSV):**
    *   **Description:** This is the primary dataset used for components 2, 3, 4, and 5. It contains sensor data (accelerometer, gyroscope) from table tennis swings, along with player demographics and metadata.
    *   **Source:** `/kaggle/input/ttswing-data/TTSWING.csv` or `/content/TTSWING.csv` (as used in your notebooks).
    *   **Link to Dataset :** `[[Insert Google Drive/Kaggle Link to TTSWING.csv Here]](https://github.com/DEPhantom/TTSWING/blob/main/dataset/TTSWING.csv)`

## 4. Data Preprocessing Overview

**A. Ball Trajectory Prediction (HDF5 Data):**
*   Loading data from multiple HDF5 files across specified subdirectories.
*   Extracting `launch_param` (input features) and `positions` (target sequences).
*   Handling variable sequence lengths by padding shorter trajectory sequences to `max_seq_length` (e.g., 629 as per model summary) using post-padding.
*   Splitting data into training, validation, and test sets.
*   Scaling input launch parameters (`X_scaler`) and output trajectory positions (`y_scaler`) using `StandardScaler`.

**B. Sensor-Based Swing Analysis (TTSWING CSV Data):**
*   Loading the `TTSWING.csv` file.
*   Removing irrelevant metadata and identifiers.
*   Encoding categorical features (Label Encoding or One-Hot Encoding).
*   Selecting motion-specific features for swing analysis.
*   Standardizing numerical features using `StandardScaler`.
*   For Stroke Power Prediction: Engineering a 'power' target variable, normalizing it, imputing missing values, and engineering date features.

*Specific preprocessing steps for each component are detailed within the respective Jupyter Notebooks. A comprehensive overview is also provided in the main project report.*

## 5. Methodology & Experiments

The methodologies for each implemented analytical component are detailed within their respective Jupyter Notebooks.

**The main project report contains a comprehensive "Methodology" section for each component and an "Experiments" section that includes:**
*   **Qualitative analysis** (e.g., visual comparison of predicted vs. actual trajectories, case-by-case review of "bad strokes").
*   **Quantitative analysis** (e.g., MSE/MAE/RMSE for trajectory prediction, Silhouette scores, Power Prediction RMSE/R2, Anomaly counts, VAE Loss).

*All figures, tables, statistics, and claims in the main project report are backed by reasonable arguments and cited via links to the relevant code sections in these notebooks (or training logs where necessary) in the footnotes of the report.*

## 6. Repository Structure (Jupyter Notebooks)

The codebase is organized into the following Jupyter Notebooks:
Code Files:
*   `Ball_trajectory_prediction.ipynb.ipynb`: Loads HDF5 trajectory data, preprocesses, builds, trains, and evaluates an LSTM model for trajectory prediction. Includes visualization of results.
*   `player_profiling_clustering.ipynb`: Performs data preprocessing, PCA, trains and visualizes clustering models (K-Means, DBSCAN, HDBSCAN) for player profiling from TTSWING data.
*   `Stroke_Power_Prediction.ipynb`: Implements data loading, feature engineering, preprocessing, trains multiple regression models, evaluates them, and saves the best performing model from TTSWING data.
*   `Bad_Stroke_Finder.ipynb`: Preprocesses motion data from TTSWING, applies anomaly detection models, and visualizes detected anomalies.
*   `synthetic-swing-generation-with-vae.ipynb`: Implements data preparation, VAE architecture, training, and generation of synthetic swing statistical features from TTSWING data.
*   `README.md`: This file.

*(Note: `train`, `eval`, `infer` functionalities are contained within these notebooks.)*

## 7. Setup Instructions

1.  **Environment:** This project is designed to run in Google Colab or Kaggle Notebooks, or a local Python environment.
2.  **Python Version:** Python 3.x.
3.  **Dependencies:**
    ```bash
    pip install numpy matplotlib tensorflow scipy scikit-learn h5py glob pandas seaborn hdbscan joblib tqdm
    ```
    *(Ensure `pip install -U scikit-learn` is run as per notebook instructions if needed for specific library versions.)*
    For Colab/Kaggle, most of these are pre-installed. Tensorflow is used for the LSTM model. Torch is used for the VAE.
4.  **Dataset Access:**
    *   **Trajectory Data (HDF5):**
        *   Upload your HDF5 files to Google Drive if using Colab, or as a Kaggle Dataset.
        *   Update `BASE_DATA_PATH` and `SUBDIRECTORIES_TO_PROCESS` in `ball_trajectory_prediction_lstm.ipynb` to point to your data.
    *   **TTSWING Data (CSV):**
        *   Upload `TTSWING.csv` to your Colab/Kaggle environment or provide the Kaggle dataset path (e.g., `/kaggle/input/ttswing-data/TTSWING.csv`).
        *   Update file paths in the relevant notebooks if necessary.
    *   **Links to Datasets (Host and provide for replication):**
        *   HDF5 Trajectory Data: `Links Provided in the staring of readme`
        *   TTSWING CSV: `Links Provided in the staring of readme`

5.  **Model Weights (for inference/evaluation of pre-trained models):**
    *   Download pre-trained model files (e.g., `best_stroke_power_model.pkl`, LSTM model weights if saved separately) from `[Insert Google Drive Link to Model Weights Here]`.
    *   Upload to your Colab/Kaggle environment and update file paths in the notebooks.

## 8. Running the Code (Notebooks)

Open the notebooks in Google Colab, Kaggle, or a local Jupyter environment and run the cells sequentially.

### Training Models

*   **Ball Trajectory Prediction (`Ball_trajectory_prediction.ipynb`):**
    *   **Functionality:** Loads and preprocesses HDF5 data, defines the LSTM model, trains it, and saves the model weights (e.g., `ball_trajectory_lstm_model.h5`).
    *   **Output:** Training history (loss, MAE, RMSE), saved model file.
*   **Player Profiling (`player_profiling_clustering.ipynb`):**
    *   **Functionality:** Preprocesses TTSWING data, performs PCA, trains K-Means, DBSCAN, HDBSCAN.
    *   **Output:** Metrics and visualizations.
*   **Stroke Power Prediction (`Stroke_Power_Prediction.ipynb`):**
    *   **Functionality:** Preprocesses TTSWING data, trains regression models.
    *   **Output:** Evaluation metrics, saves `best_stroke_power_model.pkl`.
*   **Bad Stroke Identification (`Bad_Stroke_Finder.ipynb`):**
    *   **Functionality:** Preprocesses TTSWING motion data, trains anomaly detection models.
    *   **Output:** Anomaly labels, visualizations.
*   **Synthetic Swing Generation (`synthetic-swing-generation-with-vae.ipynb`):**
    *   **Functionality:** Prepares TTSWING motion data, trains a VAE.
    *   **Output:** Training loss, saved VAE model (if implemented).

### Evaluating Models/Outputs

*   **Ball Trajectory Prediction:** The notebook includes evaluation on a test set, reporting MSE, MAE, and RMSE. Visualizations of predicted vs. actual trajectories are generated.
*   **Player Profiling:** Evaluated using Silhouette Score (K-Means) and qualitative assessment of PCA visualizations.
*   **Stroke Power Prediction:** The training notebook evaluates models using RMSE and R2 scores.
*   **Bad Stroke Identification:** Evaluated by visualizing anomalies and using consensus metrics.
*   **Synthetic Swing Generation:** Evaluated by monitoring VAE training loss and qualitative inspection of generated samples.

### Making Inferences/Generating Outputs

*   **Ball Trajectory Prediction:** Load the trained LSTM model (`ball_trajectory_lstm_model.h5`). Provide new launch parameters (scaled) to predict the 3D trajectory sequence. The notebook demonstrates this.
*   **Stroke Power Prediction:** The `Stroke_Power_Prediction.ipynb` demonstrates loading `best_stroke_power_model.pkl`, preprocessing new data, and generating power predictions (saved to `power_predictions_actual_scale.csv`).
*   **Bad Stroke Identification:** Feed new preprocessed swing data to the trained anomaly detection models.
*   **Player Profiling:** Assign new preprocessed swing data to clusters using trained clustering models.
*   **Synthetic Swing Generation:** Sample from the VAE's latent space and decode to generate new swing feature sets.

## 9. Model Weights & Architectures

*   **Ball Trajectory LSTM Model:**
    *   Architecture: As described in `ball_trajectory_prediction_lstm.ipynb` (Dense -> Dropout -> RepeatVector -> LSTM -> Dropout -> LSTM -> Dropout -> TimeDistributed(Dense)). The provided summary:
        ```
        Model: "sequential_1"
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
        ┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
        │ dense_2 (Dense)                 │ (None, 128)            │           768 │
        ├─────────────────────────────────┼────────────────────────┼───────────────┤
        │ dropout_3 (Dropout)             │ (None, 128)            │             0 │
        ├─────────────────────────────────┼────────────────────────┼───────────────┤
        │ repeat_vector_1 (RepeatVector)  │ (None, 629, 128)       │             0 │
        ├─────────────────────────────────┼────────────────────────┼───────────────┤
        │ lstm_2 (LSTM)                   │ (None, 629, 128)       │       131,584 │
        ├─────────────────────────────────┼────────────────────────┼───────────────┤
        │ dropout_4 (Dropout)             │ (None, 629, 128)       │             0 │
        ├─────────────────────────────────┼────────────────────────┼───────────────┤
        │ lstm_3 (LSTM)                   │ (None, 629, 128)       │       131,584 │
        ├─────────────────────────────────┼────────────────────────┼───────────────┤
        │ dropout_5 (Dropout)             │ (None, 629, 128)       │             0 │
        ├─────────────────────────────────┼────────────────────────┼───────────────┤
        │ time_distributed_1              │ (None, 629, 3)         │           387 │
        │ (TimeDistributed)               │                        │               │
        └━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━┛
         Total params: 264,323 (1.01 MB)
         Trainable params: 264,323 (1.01 MB)
         Non-trainable params: 0 (0.00 B)
        ```
    *   Weights: `ball_trajectory_lstm_model.h5` (or similar, generated by the notebook).
*   **Stroke Power Prediction Model:**
    *   Best Model: MLP Regressor (details in `Stroke_Power_Prediction.ipynb`).
    *   Weights: `best_stroke_power_model.pkl`.
*   **VAE Model (Synthetic Swing Generation):**
    *   Architecture: As defined in `synthetic-swing-generation-with-vae.ipynb` (Encoder: FC-ReLU-FC-ReLU -> mu/logvar; Decoder: FC-ReLU-FC-ReLU-FC).
    *   Weights: Saved by PyTorch within the notebook session (can be explicitly saved if needed).

**Link to All Model Weights:** `[Insert Single Google Drive Folder Link for All Model Weights Here]`

## 10. Detailed Results, Discussion & Conclusion

A comprehensive analysis of the quantitative results (e.g., Test MSE/MAE/RMSE for trajectory: 0.0350/0.0395m/0.1871m), qualitative observations, visualizations, interpretation of findings, strengths, limitations, challenges encountered, and potential future enhancements for each analytical component is provided in the **main project report document**. This README serves as a guide to the codebase and its execution.

---
