"""
Visualization utilities for neural network architectures.
"""

import matplotlib.pyplot as plt
from math import cos, sin, atan
from .nn import MLP


class Neuron:
    """Represents a single neuron in the visualization."""

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def draw(self, neuron_radius, ax, color="lightblue"):
        """Draw the neuron as a circle."""
        circle = plt.Circle(
            (self.x, self.y), radius=neuron_radius, fill=True, color=color
        )
        ax.add_patch(circle)


class Layer:
    """Represents a layer of neurons in the visualization."""

    def __init__(self, network, number_of_neurons, number_of_neurons_in_widest_layer):
        self.vertical_distance_between_layers = 6
        self.horizontal_distance_between_neurons = 2
        self.neuron_radius = 0.5
        self.number_of_neurons_in_widest_layer = number_of_neurons_in_widest_layer
        self.previous_layer = self.__get_previous_layer(network)
        self.y = self.__calculate_layer_y_position()
        self.neurons = self.__initialize_neurons(number_of_neurons)

    def __initialize_neurons(self, number_of_neurons):
        """Initialize neurons in the layer."""
        neurons = []
        x = self.__calculate_left_margin_so_layer_is_centered(number_of_neurons)
        for i in range(number_of_neurons):
            neuron = Neuron(x, self.y)
            neurons.append(neuron)
            x += self.horizontal_distance_between_neurons
        return neurons

    def __calculate_left_margin_so_layer_is_centered(self, number_of_neurons):
        """Calculate the left margin to center the layer."""
        return (
            self.horizontal_distance_between_neurons
            * (self.number_of_neurons_in_widest_layer - number_of_neurons)
            / 2
        )

    def __calculate_layer_y_position(self):
        """Calculate the y position of the layer."""
        if self.previous_layer:
            return self.previous_layer.y + self.vertical_distance_between_layers
        else:
            return 0

    def __get_previous_layer(self, network):
        """Get the previous layer in the network."""
        if len(network.layers) > 0:
            return network.layers[-1]
        else:
            return None

    def __line_between_two_neurons(self, neuron1, neuron2, ax):
        """Draw a line between two neurons."""
        angle = atan((neuron2.x - neuron1.x) / float(neuron2.y - neuron1.y))
        x_adjustment = self.neuron_radius * sin(angle)
        y_adjustment = self.neuron_radius * cos(angle)
        line = plt.Line2D(
            (neuron1.x - x_adjustment, neuron2.x + x_adjustment),
            (neuron1.y - y_adjustment, neuron2.y + y_adjustment),
            color="gray",
            linewidth=0.5,
        )
        ax.add_line(line)

    def draw(self, ax, layer_type=0):
        """Draw the layer and connections to previous layer."""
        # Draw neurons
        neuron_color = "lightblue"
        if layer_type == 0:
            neuron_color = "lightgreen"  # Input layer
        elif layer_type == -1:
            neuron_color = "salmon"  # Output layer

        for neuron in self.neurons:
            neuron.draw(self.neuron_radius, ax, color=neuron_color)
            if self.previous_layer:
                for previous_layer_neuron in self.previous_layer.neurons:
                    self.__line_between_two_neurons(neuron, previous_layer_neuron, ax)

        # Add layer labels
        x_text = (
            self.number_of_neurons_in_widest_layer
            * self.horizontal_distance_between_neurons
            + 1
        )
        if layer_type == 0:
            plt.text(x_text, self.y, "Input Layer", fontsize=12)
        elif layer_type == -1:
            plt.text(x_text, self.y, "Output Layer", fontsize=12)
        else:
            plt.text(x_text, self.y, f"Hidden Layer {layer_type}", fontsize=12)


class NeuralNetworkDrawing:
    """Class for drawing a neural network architecture."""

    def __init__(self, number_of_neurons_in_widest_layer):
        self.number_of_neurons_in_widest_layer = number_of_neurons_in_widest_layer
        self.layers = []
        self.layer_type = 0

    def add_layer(self, number_of_neurons):
        """Add a layer to the network visualization."""
        layer = Layer(self, number_of_neurons, self.number_of_neurons_in_widest_layer)
        self.layers.append(layer)

    def draw(self, title="Neural Network Architecture", figsize=(12, 8)):
        """Draw the complete neural network."""
        fig, ax = plt.subplots(figsize=figsize)

        for i in range(len(self.layers)):
            layer = self.layers[i]
            if i == len(self.layers) - 1:
                i = -1
            layer.draw(ax, i)

        plt.axis("scaled")
        plt.axis("off")
        plt.title(title, fontsize=15)
        return fig


def visualize_network(
    network_structure,
    title="Neural Network Architecture",
    figsize=(12, 8),
    save_path=None,
):
    """
    Visualize a neural network given its structure.

    Args:
        network_structure: List of integers representing the number of neurons in each layer
        title: Title for the plot
        figsize: Figure size as a tuple (width, height)
        save_path: Path to save the figure (if None, the figure is not saved)

    Returns:
        The matplotlib figure
    """
    widest_layer = max(network_structure)
    network = NeuralNetworkDrawing(widest_layer)

    for layer_size in network_structure:
        network.add_layer(layer_size)

    fig = network.draw(title=title, figsize=figsize)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)

    return fig


def visualize_mlp(mlp, title="MLP Architecture", figsize=(12, 8), save_path=None):
    """
    Visualize an MLP model from the tinynn library.

    Args:
        mlp: An instance of the MLP class
        title: Title for the plot
        figsize: Figure size as a tuple (width, height)
        save_path: Path to save the figure (if None, the figure is not saved)

    Returns:
        The matplotlib figure
    """
    if not isinstance(mlp, MLP):
        raise TypeError("Expected an MLP instance")

    # Extract network structure
    network_structure = [
        len(layer.neurons[0].w) for layer in mlp.layers
    ]  # Input sizes for each layer
    network_structure.append(len(mlp.layers[-1].neurons))  # Output size

    return visualize_network(network_structure, title, figsize, save_path)
