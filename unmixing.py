# -*- coding: utf-8 -*-
"""Unmixing.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/18F65zQ4OaCpsYvJSNJ3DBddgLbh6LPBn

# Imports
"""

import tensorflow as tf
import numpy as np
import scipy.io
from scipy.optimize import linear_sum_assignment
from scipy import ndimage
import matplotlib.pyplot as plt
from google.colab import drive
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import median_filter

# Mount Google Drive
drive.mount('/content/drive')

# Load the .mat file
mat_data = scipy.io.loadmat('/content/drive/My Drive/Unmixing/Artificial_Core_Image.mat')

"""# Helper Functions"""

def generate_abundance_maps(predicted_abundances, S_GT, mapping):
    adjusted_predicted_abundances = np.zeros_like(predicted_abundances)

    # Adjust predicted abundances based on the mapping
    for gt_index, pred_index in mapping.items():
        adjusted_predicted_abundances[:, :, gt_index] = predicted_abundances[:, :, pred_index]

    # Visualize the adjusted predicted abundance maps alongside the GT abundance maps
    num_abundance_maps = S_GT.shape[-1]
    fig, axs = plt.subplots(2, num_abundance_maps, figsize=(15, 6))

    for i in range(num_abundance_maps):
        # Ground truth abundance maps
        axs[0, i].imshow(S_GT[:, :, i], cmap='viridis', vmin=0, vmax=1)
        axs[0, i].set_title(f'GT {i + 1}')

        # Predicted abundance maps
        axs[1, i].imshow(adjusted_predicted_abundances[:, :, i], cmap='viridis', vmin=0, vmax=1)
        axs[1, i].set_title(f'Predicted {i + 1}')

    fig.suptitle('Ground Truth vs Adjusted Predicted Abundance Maps', fontsize=16)
    plt.show()


def spectral_angle_distance(vector1, vector2):
    y_true_normalized = tf.math.l2_normalize(vector1, axis=-1)
    y_pred_normalized = tf.math.l2_normalize(vector2, axis=-1)
    #cosine_similarity = tf.reduce_sum(tf.multiply(y_true_normalized, y_pred_normalized), axis=-1)
    # Ensure the dot product is within [-1, 1]
    cosine_similarity = tf.clip_by_value(tf.reduce_sum(tf.multiply(y_true_normalized, y_pred_normalized), axis=-1), -1.0, 1.0)

    sad = tf.acos(cosine_similarity)
    return sad

def map_gt_to_preds(ground_truth_endmembers, predicted_endmembers):
    # Calculate the cost matrix (SAD values)
    cost_matrix = np.zeros((ground_truth_endmembers.shape[0], predicted_endmembers.shape[0]))

    for i, gt in enumerate(ground_truth_endmembers):
        for j, pred in enumerate(predicted_endmembers):
            cost_matrix[i, j] = spectral_angle_distance(tf.cast(gt, dtype=tf.float32), tf.cast(pred, dtype=tf.float32))

    # Solve the assignment problem
    gt_indices, pred_indices = linear_sum_assignment(cost_matrix)

    # Create the mapping
    mapping = dict(zip(gt_indices, pred_indices))

    return mapping

def apply_median_filter(spectra, size=4):
    # Apply a median filter to the spectra.
    # The 'size' defines the size of the window over which the median is computed.
    filtered_spectra = median_filter(spectra, size=size, mode='reflect')
    return filtered_spectra

# Register the custom loss function
tf.keras.losses.spectral_angle_distance = spectral_angle_distance

"""# Data Preparation"""

# Extract data
Y = mat_data['Y']  # 200x200x343
GT = mat_data['GT']  # 10x343
S_GT = mat_data['S_GT']  # 200x200x10

# Flatten the input image for training
X_train = Y.reshape(-1, Y.shape[-1])

# Normalize the input data
X_train_normalized = X_train / np.max(X_train.flatten())

# Define the autoencoder architecture
input_dim = Y.shape[2]  # Number of bands
R = GT.shape[0]  # Number of neurons in the encoding (compression) layer

"""# Model"""

num_iterations = 1  # Number of iterations to run (To perform Monte Carlo Simulations)

mean_sad_values = []

for iteration in range(num_iterations):
  print(f"Starting iteration {iteration + 1}/{num_iterations}")

# Encoder
  encoder = tf.keras.Sequential([
      tf.keras.layers.Input(shape=(input_dim,)),
      tf.keras.layers.Dense(units=9*R, activation=tf.keras.layers.LeakyReLU(alpha=0.01)),  # Hidden layer 1
      tf.keras.layers.Dense(units=6*R, activation=tf.keras.layers.LeakyReLU(alpha=0.01)),  # Hidden layer 2
      tf.keras.layers.Dense(units=3*R, activation=tf.keras.layers.LeakyReLU(alpha=0.01)),  # Hidden layer 3
      tf.keras.layers.Dense(units=R, activation=tf.keras.layers.LeakyReLU(alpha=0.01), kernel_regularizer=tf.keras.regularizers.l2(0.005)),  # Hidden layer 4
      tf.keras.layers.BatchNormalization(),  # Utility layer: Batch Normalization (Layer 6)
      tf.keras.layers.Activation(activation=tf.keras.layers.LeakyReLU(alpha=0.01)),  # Layer 7
      tf.keras.layers.Dense(units=R, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(0.015)),  # Utility layer: Enforces ASC (Layer 8) with L2 regularization
      tf.keras.layers.GaussianDropout(rate=0.12)  # Utility layer: Gaussian Dropout (Layer 9)
  ])

# Decoder
  decoder = tf.keras.Sequential([
      tf.keras.layers.Input(shape=(R,)),
      tf.keras.layers.Dense(units=input_dim, activation='linear')  # Decoding layer
  ])

# Combine the encoder and decoder to create the autoencoder
  autoencoder = tf.keras.Sequential([encoder, decoder])

#Learning Rate Tuning
  initial_learning_rate = 0.001
  lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
      initial_learning_rate,
      decay_steps=100000,
      decay_rate=0.96,
      staircase=True)

  # Compile the autoencoder
  optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
  autoencoder.compile(optimizer=optimizer, loss=spectral_angle_distance)

# Train the autoencoder
  autoencoder.fit(X_train_normalized, X_train_normalized, epochs=50, verbose=1)  # Autoencoder tries to reconstruct input

# Encode and decode the test data
  encoded_data = encoder.predict(X_train_normalized)
  decoded_data = decoder.predict(encoded_data)

# Reshape the decoded data back to the original image shape
  decoded_data_reshaped = decoded_data.reshape(Y.shape)

# Evaluate the autoencoder (mean squared error between original and decoded data)
  mse = np.mean(np.square(X_train_normalized.reshape(Y.shape) - decoded_data_reshaped))
  print(f"Mean Squared Error (MSE): {mse:.4f}")

  # Evaluate the autoencoder (spectral angle distance between each endmember and ground truth)
  endmem_spectra = decoder.layers[-1].get_weights()[0]
  sad_values = []

  num_endmembers = GT.shape[0]

# Determine the number of predicted endmembers
  num_predicted_endmembers = endmem_spectra.shape[0]

# Mapping predicted endmembers to GT
  mapping = map_gt_to_preds(GT, endmem_spectra)

# List to store the SAD values for each GT to predicted mapping pair
  sad_values_list = []

  # Determine the common y-axis limits based on min and max values in ground truth and predictions
  all_data = np.concatenate((GT, endmem_spectra))
  y_min, y_max = np.min(all_data), np.max(all_data)

  # Number of columns for subplots
  num_columns = 2
  num_rows = int(np.ceil(num_endmembers / num_columns))

  # Create subplots with two columns and increased vertical spacing
  fig, axs = plt.subplots(num_rows, num_columns, figsize=(10, num_rows * 4))

  # Flatten the array of axes, so we can iterate over it
  axs = axs.flatten()

  for i in range(num_endmembers):
      pred_index = mapping[i]
      sad_value = spectral_angle_distance(
          tf.cast(GT[i, :], dtype=tf.float32),
          tf.cast(endmem_spectra[pred_index, :], dtype=tf.float32)
      )
      sad_values_list.append(sad_value.numpy())  # Store the calculated SAD value

    # Plotting the GT and predicted endmembers with a consistent y-axis scale
      axs[i].plot(GT[i, :], label='Ground Truth', color='b', linestyle='--', linewidth=1.0)
      axs[i].plot(endmem_spectra[pred_index, :], label='Predicted', color='r', linewidth=1.0)
      axs[i].set_ylim([y_min, y_max])  # Set common y-axis limits
      axs[i].set_title(f'GT Endmember {i + 1} Matched to Predicted {pred_index + 1} (SAD: {sad_value:.4f})')
      axs[i].legend()
      axs[i].set_xlabel('Band')
      axs[i].set_ylabel('Reflectance')

  # Hide any unused subplots if num_endmembers isn't a multiple of num_columns
  for i in range(num_endmembers, num_rows * num_columns):
    axs[i].axis('off')

  # Compute the mean SAD from the list of SAD values
  mean_sad_value = np.mean(sad_values_list)

  # Set the overall mean SAD as the main title for the subplots
  fig.suptitle(f'Overall Mean SAD: {mean_sad_value:.4f}', fontsize=16)

  # Adjust the layout
  plt.tight_layout(rect=[0, 0.03, 1, 0.95])
  plt.subplots_adjust(hspace=0.5)
  plt.show()

# Reshape the encoded data to match the shape of S_GT
abundances = encoded_data.reshape((200, 200, -1))

generate_abundance_maps(abundances, S_GT, mapping)