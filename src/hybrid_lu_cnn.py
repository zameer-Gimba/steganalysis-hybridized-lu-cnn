import tensorflow as tf

def hybridize_lu_activation_function_in_cnn(model, iterations=5):
    """
    Dynamically hybridizes LU-based activation functions (ELU, ReLU, LeakyReLU)
    inside a CNN model based on activation output behavior.

    This function is designed for steganalysis tasks, where subtle feature
    variations require adaptive non-linear transformations.

    Args:
        model (tf.keras.Model): Input CNN model
        iterations (int): Number of hybridization passes

    Returns:
        tf.keras.Model: Modified CNN model
    """

    for _ in range(iterations):
        modified_layers = []

        for layer in model.layers:
            # Only process activation layers
            if isinstance(layer, tf.keras.layers.Activation):
                activation_output = layer(layer.input)

                # Decide activation based on output distribution
                if tf.reduce_sum(activation_output) < 0:
                    new_activation = tf.keras.layers.ELU()
                elif tf.reduce_sum(activation_output) > 0:
                    new_activation = tf.keras.layers.ReLU()
                else:
                    new_activation = tf.keras.layers.LeakyReLU()

                modified_layers.append(new_activation)
            else:
                modified_layers.append(layer)

        model = tf.keras.Sequential(modified_layers)

    return model


def build_sample_cnn():
    """
    Builds a simple CNN model for demonstration and testing.
    """

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    return model

# Activation Function Visualization
def plot_activation_functions():
    """
    Plots ELU, ReLU, and LeakyReLU activation functions
    and saves the graph to the results directory.
    """
    x = np.linspace(-5, 5, 500)

    relu = tf.keras.activations.relu(x).numpy()
    elu = tf.keras.activations.elu(x).numpy()
    leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.1)(x).numpy()

    plt.figure(figsize=(10, 6))
    plt.plot(x, relu, label="ReLU")
    plt.plot(x, elu, label="ELU")
    plt.plot(x, leaky_relu, label="Leaky ReLU")

    plt.title("Comparison of ReLU, ELU, and Leaky ReLU Activation Functions")
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.legend()
    plt.grid(True)
    plt.show()

# Main Execution
if __name__ == "__main__":
    # Create original CNN
    original_model = create_sample_cnn()

    # Apply hybrid activation strategy
    hybrid_model = hybridize_lu_activation(original_model, iterations=1)

    # Display model summary
    hybrid_model.summary()

    # Plot activation functions
    plot_activation_functions()
