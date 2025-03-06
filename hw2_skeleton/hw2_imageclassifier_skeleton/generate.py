import matplotlib.pyplot as plt
import numpy as np

# Data for Flatness Plot (Shuffled=True)
alpha_shuffled_true = np.array([0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.4])
validation_error_shuffled_true = np.array([9.00, 8.97, 8.95, 8.91, 8.89, 8.89, 8.87, 8.87, 8.875, 8.87]) * 10 # Multiply by 10 to match y-axis scale

# --- Plotting Function with Styling ---
def create_flatness_plot(alpha_v, v_err, title, filename, use_log_scale=False):
    """
    Generates a flatness plot with specified styling and saves it to a file.

    Args:
        alpha_v (np.array): Array of alpha values (x-axis).
        v_err (np.array): Array of validation error values (y-axis).
        title (str): Title of the plot.
        filename (str): Filename to save the plot as.
        use_log_scale (bool, optional): Whether to use a logarithmic scale for the y-axis.
                                        Defaults to False.
    """
    plt.figure(figsize=(8, 6)) # Create a new figure for each plot

    # Set logarithmic scale if specified
    if use_log_scale:
        plt.yscale('log')

    plt.plot(alpha_v, v_err, linestyle='-', marker='o', color='purple')  # Use purple to match example
    plt.title(title)
    plt.xlabel('Alpha')
    plt.ylabel('Validation Error (%)')
    plt.grid(False)  # Remove grid to match example
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close() # Close the figure to free memory


# --- Generate and Save the "Shuffled=True" Plot with Styling ---
create_flatness_plot(
    alpha_v=alpha_shuffled_true,
    v_err=validation_error_shuffled_true,
    title='Flatness Plot (Shuffled=True)',
    filename='interpolate_true_styled.png', # Filename for shuffled=true plot
    use_log_scale=False # No log scale for Shuffled=True, as per original image
)

print("Plot 'interpolate_true_styled.png' saved with requested styling.")