import streamlit as st
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import tensorflow as tf
import numpy as np
import scipy.io
from scipy.optimize import linear_sum_assignment
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
from sklearn.cluster import KMeans
from scipy.io import loadmat
from scipy.io import savemat
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_samples, silhouette_score
from kneed import KneeLocator
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objs as go
from plotly import tools
from plotly.subplots import make_subplots
import plotly.offline as py
import io

# Unmixing Helper Functions
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
    return fig  # Return the figure object instead of showing it


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

# Register the custom loss function
tf.keras.losses.spectral_angle_distance = spectral_angle_distance


# Interface Functions

def setup_session_state():
    # Initialize session state variables with default values if they don't exist
    if 'uploaded_file' not in st.session_state:
        st.session_state['uploaded_file'] = None
    if 'unmixing_results' not in st.session_state:
        st.session_state['unmixing_results'] = None
    if 'denoising_results' not in st.session_state:
        st.session_state['denoising_results'] = None
    if 'estimation_results' not in st.session_state:
        st.session_state['estimation_results'] = None
    if 'classification_results' not in st.session_state:
        st.session_state['classification_results'] = None


def setup_sidebar():
    st.sidebar.title("Navigation")

    # Unified options list with "Home" and operations
    options = ["Home", "Denoising", "ID Estimation", "Unmixing", "Classification"]
    selected_option = st.sidebar.radio("Go to:", options)

    # Update the session state based on selection
    if selected_option and st.session_state['page'] != selected_option:
        st.session_state['page'] = selected_option

def interactive_band_visualization(file):
    mat_data = loadmat(file)
    Y = mat_data['Y']
    
    # Slider for selecting the band
    st.markdown('**Select a Spectral Band to view:**')
    # Your code for slider or other components follows
    band_index = st.slider('', 0, 100, 25)  # Example slider without label as label is in markdown
    st.write(f'You selected band index: {band_index}')
    
    selected_band = Y[:, :, band_index]
    selected_band_normalized = (selected_band - selected_band.min()) / (selected_band.max() - selected_band.min())

    st.image(selected_band_normalized, caption=f"Visualization of Band {band_index + 1}", use_column_width=True)


# Interface

# Add logo at the top of the sidebar or main page content
logo_url = "data:image/png;base64,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"  # URL to your logo image

# Using HTML to position the logo at the right-most corner
st.markdown(
    f"""
    <div style="position: fixed; top: 50px; right: 40px; z-index: 999;">
        <img src="{logo_url}" alt="Logo" style="height: 50px;">
    </div>
    """,
    unsafe_allow_html=True
    )


# Initialize the session state for page control if not already present
if 'page' not in st.session_state or not st.session_state['page']:
    st.session_state['page'] = 'Home'  # Default to 'Home'

def main():
    # Page display functions
    def home():
        st.title('Mineral Detection From Hyperspectral Images')

        gif_url = 'https://www.csr.utexas.edu/projects/rs/hrs/pics/avcube.gif'
        st.image(gif_url)

        if st.session_state['uploaded_file'] is not None:
            st.success("File already uploaded.")
            st.write("Uploaded File:", st.session_state['uploaded_file'].name)
            if st.button('Replace File'):
                # Clear the existing file from session state to allow uploading a new file
                st.session_state['uploaded_file'] = None
                # Use st.experimental_rerun() to refresh the state of the app
                st.experimental_rerun()
        
            interactive_band_visualization(st.session_state['uploaded_file'])
            st.markdown('**Choose further processing options from the sidebar.**')
            

        # Show uploader if no file is present or after pressing 'Replace File'
        if st.session_state['uploaded_file'] is None:
            uploaded_file = st.file_uploader("Choose a hyperspectral image file (.mat)", type=["mat"])
            if uploaded_file is not None:
                st.session_state['uploaded_file'] = uploaded_file
                st.success("File uploaded successfully.")
                # Direct call to display the file details and visualization
                st.write("Uploaded File:", uploaded_file.name)
                interactive_band_visualization(uploaded_file)
                st.markdown('**Choose further processing options from the sidebar.**')


    def denoising():
        st.header("Denoising")
        # Display existing results if already computed
        if st.session_state['denoising_results'] is not None:
            st.pyplot(st.session_state['denoising_results'])

        if st.session_state['uploaded_file'] is not None and st.session_state['denoising_results'] is None:

            st.write("Ready to perform denoising with the uploaded file.")
            # Show file details for debugging
            st.write("Uploaded File:", st.session_state['uploaded_file'].name)

            # Button to start processing
            if st.button('Perform Denoising'):
                
                with st.spinner('Performing denoising... Please wait'):
                    st.write(st.session_state['denoising_results'])

        else:
            st.error("Please upload a file in the Home page before starting this operation.")
                    

    def id_estimation():
        st.header("Intrinsic Dimensionality Estimation")
        # Display existing results if already computed
        if st.session_state['estimation_results'] is not None:
            st.pyplot(st.session_state['estimation_results'])

        if st.session_state['uploaded_file'] is not None and st.session_state['estimation_results'] is None:

            st.write("Ready to perform intrinsic dimensionality estimation with the uploaded file.")
            # Show file details for debugging
            st.write("Uploaded File:", st.session_state['uploaded_file'].name)

            # Button to start processing
            if st.button('Perform ID Estimation'):
                
                with st.spinner('Performing ID Estimation... Please wait'):

                    data = st.session_state['uploaded_file']
                    # Load .mat file
                    mat_data = loadmat(data)
                    new_data = mat_data['Y']  # Assuming Y is your data key
                    x = new_data.astype(np.float32)

                    # Prepare to store WCSS for each number of clusters
                    avg_WCSS = [0] * 15  # There are 15 possible cluster numbers (1-15)

                    for p in range(x.shape[2]):  # Adjusted to iterate over the third dimension of x    
                        slice_x = x[:, :, p]
                        df = pd.DataFrame(slice_x)
                        slice_values = df.values

                        for i in range(1, 16):
                            model = KMeans(n_clusters=i, init='k-means++')
                            model.fit(slice_values)
                            # Accumulate WCSS values
                            avg_WCSS[i-1] += model.inertia_ / x.shape[2]  # Average by dividing by the number of slices

                    # Ensure lengths match
                    n_clusters_range = list(range(1, 16))
                    if len(n_clusters_range) != len(avg_WCSS):
                        st.error("Mismatch in data length for knee detection.")
                        st.write(f"Length of n_clusters_range: {len(n_clusters_range)}, Length of avg_WCSS: {len(avg_WCSS)}")
                        return

                    kneedle = KneeLocator(n_clusters_range, avg_WCSS, curve='convex', direction='decreasing')
                    st.write(f"Elbow point at: {kneedle.elbow}")

                    fig, ax = plt.subplots()
                    ax.plot(n_clusters_range, avg_WCSS, marker='o')
                    ax.set_xlabel('Number of clusters')
                    ax.set_ylabel('WCSS')
                    ax.axvline(x=kneedle.elbow, color='r', linestyle='--')
                    st.pyplot(fig)

                    st.write(st.session_state['estimation_results'])

        else:
            st.error("Please upload a file in the Home page before starting this operation.")


    def unmixing():
        st.header("Unmixing")
    
        # Display existing results if already computed
        if st.session_state['unmixing_results'] is not None:
            st.pyplot(st.session_state['unmixing_results'])

        if st.session_state['uploaded_file'] is not None and st.session_state['unmixing_results'] is None:

            st.write("Ready to perform unmixing with the uploaded file.")
            # Show file details for debugging
            st.write("Uploaded File:", st.session_state['uploaded_file'].name)

            # Button to start processing
            if st.button('Perform Unmixing'):
                
                with st.spinner('Performing unmixing... Please wait'):

                    data = st.session_state['uploaded_file']
                    # Load .mat file
                    mat_data = loadmat(data)

                    # Extract data
                    #Y = mat_data['Y']  # 200x200x343
                    GT = mat_data['GT']  # 10x343
                    S_GT = mat_data['S_GT']  # 200x200x10
                    Y = np.tensordot(S_GT, GT, axes=([2], [0]))

                    # Flatten the input image for training
                    X_train = Y.reshape(-1, Y.shape[-1])

                    # Normalize the input data
                    X_train_normalized = X_train / np.max(X_train.flatten())

                    # Define the autoencoder architecture
                    input_dim = Y.shape[2]  # Number of bands
                    R = GT.shape[0]  # Number of neurons in the encoding (compression) layer  

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

                        #GT = apply_median_filter(GT)
                        #endmem_spectra = apply_median_filter(endmem_spectra)

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
                        st.success('Unmixing completed!')

                        # Use an expander to allow users to decide if they want to see the plots
                        with st.expander("View Comparison Plots"):
                            st.pyplot(fig)  # Show the figure within the expander
                        
                        abundances = encoded_data.reshape((200, 200, -1))
                        fig = generate_abundance_maps(abundances, S_GT, mapping)

                        # Use an expander to allow users to view the plots
                        with st.expander("View Abundance Maps"):
                            st.pyplot(fig)

                        st.write('Downloading Options')

                        # Create a BytesIO buffer to save the .mat file in memory
                        buffer = io.BytesIO()
                        savemat(buffer, {'endmem_spectra': endmem_spectra})
                        buffer.seek(0)

                        # Create a download button
                        st.download_button(
                            label="Download Endmember Signatures",
                            data=buffer,
                            file_name="endmember_signatures.mat",
                            mime="application/octet-stream"
                        )

                        # Assuming 'encoded_data' is reshaped to (200, 200, -1)
                        abundances = encoded_data.reshape((200, 200, -1))

                        # Save and create download button for abundances
                        abundances_buffer = io.BytesIO()
                        savemat(abundances_buffer, {'abundances': abundances})
                        abundances_buffer.seek(0)

                        st.download_button(
                            label="Download Endmember Abundances",
                            data=abundances_buffer,
                            file_name="endmember_abundances.mat",
                            mime="application/octet-stream"
                        )
                    st.write(st.session_state['unmixing_results'])
        else:
            st.error("Please upload a file in the Home page before starting this operation.")

    def classification():
        st.title("Classification")
        st.write("Place classification algorithms and visualizations here.")

        # More functions for other operations...

    # Map pages to functions
    pages = {
        "Home": home,
        "Denoising": denoising,
        "ID Estimation": id_estimation,
        "Unmixing": unmixing,
        "Classification": classification
    }

    # Execute the function based on the current page
    if st.session_state['page'] in pages:
        pages[st.session_state['page']]()

# Intializations
setup_session_state()

# Setup the sidebar
setup_sidebar()

# Load the main content
main()



