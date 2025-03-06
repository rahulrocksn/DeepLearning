#
import torch
import numpy as onp
from typing import List, cast
import torch.nn as nn
import torch.nn.functional as F


class Model(torch.nn.Module):
    R"""
    Model.
    """
    def initialize(self, rng: torch.Generator) -> None:
        R"""
        Initialize parameters.
        """
        ...


class MLP(Model):
    R"""
    MLP.
    """
    def __init__(self, /, *, size: int, shapes: List[int]) -> None:
        R"""
        Initialize the class.
        """
        #
        Model.__init__(self)

        #
        buf = []
        shapes = [size * size] + shapes
        for (num_ins, num_outs) in zip(shapes[:-1], shapes[1:]):
            #
            buf.append(torch.nn.Linear(num_ins, num_outs))
        self.linears = torch.nn.ModuleList(buf)

    def initialize(self, rng: torch.Generator) -> None:
        R"""
        Initialize parameters.
        """
        #
        for linear in self.linears:
            #
            (num_outs, num_ins) = linear.weight.data.size()
            a = onp.sqrt(6 / (num_ins + num_outs))
            linear.weight.data.uniform_(-a, a, generator=rng)
            linear.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        R"""
        Forward.
        """
        #
        x = torch.flatten(x, start_dim=1)
        for (l, linear) in enumerate(self.linears):
            #
            x = linear.forward(x)
            if l < len(self.linears) - 1:
                #
                x = torch.nn.functional.relu(x)
        return x


#
PADDING = 3


class CNN(torch.nn.Module):
    R"""
    CNN.
    """
    def __init__(
        self,
        /,
        *,
        size: int, channels: List[int], shapes: List[int],
        kernel_size_conv: int, stride_size_conv: int, kernel_size_pool: int,
        stride_size_pool: int,
    ) -> None:
        R"""
        Initialize the class.
        """
        #
         # Initialize using the provided Model constructor.
        Model.__init__(self)
        
        # As required, fix the convolutional padding to 3.
        padding_conv = 3
        
        # --- Create convolutional layers ---
        buf_conv = []
        for i in range(len(channels) - 1):
            in_channels = channels[i]
            out_channels = channels[i + 1]
            conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size_conv,
                stride=stride_size_conv,
                padding=padding_conv
            )
            buf_conv.append(conv)
        self.convs = nn.ModuleList(buf_conv)
        
        # --- Create shared max pooling layer ---
        self.pool = nn.MaxPool2d(kernel_size=kernel_size_pool, stride=stride_size_pool)
        
        # --- Compute spatial size after conv+pool layers ---
        current_size = size  # e.g., 28 for MNIST
        for _ in range(len(channels) - 1):
            # After convolution:
            current_size = (current_size + 2 * padding_conv - kernel_size_conv) // stride_size_conv + 1
            # After pooling:
            current_size = (current_size - kernel_size_pool) // stride_size_pool + 1
        flattened_dim = channels[-1] * (current_size ** 2)
        
        # --- Create fully connected (linear) layers ---
        buf_linear = []
        if shapes:
            # First FC layer: from flattened_dim to shapes[0]
            buf_linear.append(nn.Linear(flattened_dim, shapes[0]))
            # Subsequent FC layers:
            for i in range(len(shapes) - 1):
                buf_linear.append(nn.Linear(shapes[i], shapes[i + 1]))
        else:
            buf_linear.append(nn.Linear(flattened_dim, 2))
        self.linears = nn.ModuleList(buf_linear)

        # Create a list of Conv2D layers and shared max-pooling layer.
        # Input and output channles are given in `channels`.
        # ```
        # buf_conv = []
        # ...
        # self.convs = torch.nn.ModuleList(buf_conv)
        # self.pool = ...
        # ```
        # YOU SHOULD FILL IN THIS FUNCTION
        ...

        # Create a list of Linear layers.
        # Number of layer neurons are given in `shapes` except for input.
        # ```
        # buf = []
        # ...
        # self.linears = torch.nn.ModuleList(buf)
        # ```
        # YOU SHOULD FILL IN THIS FUNCTION
        ...
        
        

    def initialize(self, rng: torch.Generator) -> None:
        R"""
        Initialize parameters.
        """
        #
        for conv in self.convs:
            #
            (ch_outs, ch_ins, h, w) = conv.weight.data.size()
            num_ins = ch_ins * h * w
            num_outs = ch_outs * h * w
            a = onp.sqrt(6 / (num_ins + num_outs))
            conv.weight.data.uniform_(-a, a, generator=rng)
            conv.bias.data.zero_()
        for linear in self.linears:
            #
            (num_outs, num_ins) = linear.weight.data.size()
            a = onp.sqrt(6 / (num_ins + num_outs))
            linear.weight.data.uniform_(-a, a, generator=rng)
            linear.bias.data.zero_()

    def forward(self, x: torch.Tensor, /) -> torch.Tensor:
        R"""
        Forward.
        """
        for conv in self.convs:
            x = F.relu(conv(x))
            x = self.pool(x)
        # Flatten the convolutional output.
        x = x.view(x.size(0), -1)
        # Pass through each fully connected layer.
        for i, linear in enumerate(self.linears):
            if i < len(self.linears) - 1:
                x = F.relu(linear(x))
            else:
                x = linear(x)
        return x



import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as onp
from typing import List

class CGCNN(Model):
    r"""
    CGCNN.
    """
    def __init__(
        self,
        /,
        *,
        size: int, 
        channels: List[int], 
        shapes: List[int],
        kernel_size_conv: int, 
        stride_size_conv: int, 
        kernel_size_pool: int,
        stride_size_pool: int,
    ) -> None:
        r"""
        Initialize the class.
        """
        Model.__init__(self)

        # Set proper_size to kernel_size_conv (or the appropriate value as per your design)
        proper_size = kernel_size_conv

        # Load precomputed eigenvectors.
        with open("rf-{:d}.npy".format(proper_size), "rb") as file:
            onp.load(file)  # if your file has an extra object, otherwise remove this line
            eigenvectors = onp.load(file)
        self.register_buffer(
            "basis",
            torch.from_numpy(eigenvectors).to(torch.get_default_dtype()),
        )

        # Determine the basis dimension.
        basis_dim = self.basis.shape[0]  # e.g., if the basis shape is (basis_dim, 25)
        
        # Create G-invariant convolution layers.
        buf_coeffs = []
        buf_biases = []
        for i in range(len(channels) - 1):
            in_channels = channels[i]
            out_channels = channels[i+1]
            # Each filter is generated from coefficients of shape (out_channels, in_channels, basis_dim).
            coeff_shape = (out_channels, in_channels, basis_dim)
            coeffs = torch.nn.Parameter(torch.randn(coeff_shape))
            buf_coeffs.append(coeffs)
            bias = torch.nn.Parameter(torch.zeros(out_channels))
            buf_biases.append(bias)
        self.weights = torch.nn.ParameterList(buf_coeffs)
        self.biases = torch.nn.ParameterList(buf_biases)
        
        # (4) Compute spatial size after conv+pool layers.
        # Here we compute it as before, but note that now we will later average pool spatially.
        padding_conv = 3  # as per assignment; verify if this matches your baseline CNN
        current_size = size  # e.g., 28 for MNIST images
        num_conv_layers = len(channels) - 1
        for _ in range(num_conv_layers):
            # After convolution:
            current_size = (current_size + 2 * padding_conv - kernel_size_conv) // stride_size_conv + 1
            # After pooling:
            current_size = (current_size - kernel_size_pool) // stride_size_pool + 1

        # Instead of flattening all spatial dimensions, we now use global average pooling.
        # So the dimension going into the MLP is just the number of channels in the final conv layer.
        flattened_dim = channels[-1]
        
        # Create fully connected (linear) layers.
        buf_linear = []
        if shapes:
            buf_linear.append(nn.Linear(flattened_dim, shapes[0]))
            for i in range(len(shapes) - 1):
                buf_linear.append(nn.Linear(shapes[i], shapes[i+1]))
        else:
            buf_linear.append(nn.Linear(flattened_dim, 2))
        self.linears = nn.ModuleList(buf_linear)
        
        # Create shared max pooling layer (used within the conv layers)
        self.pool = nn.MaxPool2d(kernel_size=kernel_size_pool, stride=stride_size_pool)

    def initialize(self, rng: torch.Generator) -> None:
        r"""
        Initialize parameters.
        """
        for (weight, bias) in zip(self.weights, self.biases):
            (_, ch_ins, basis_dim) = weight.data.size()
            a = 1 / onp.sqrt(ch_ins * basis_dim)
            weight.data.uniform_(-a, a, generator=rng)
            bias.data.zero_()
        for linear in self.linears:
            (num_outs, num_ins) = linear.weight.data.size()
            a = onp.sqrt(6 / (num_ins + num_outs))
            linear.weight.data.uniform_(-a, a, generator=rng)
            linear.bias.data.zero_()

    def forward(self, x: torch.Tensor, /) -> torch.Tensor:
        r"""
        Forward pass.
        """
        # For each G-invariant convolution layer:
        for i in range(len(self.weights)):
            coeffs = self.weights[i]  # shape: (out_channels, in_channels, basis_dim)
            bias = self.biases[i]     # shape: (out_channels,)
            # Reconstruct full filters via Einstein summation.
            # self.basis has shape (basis_dim, proper_size*proper_size)
            weight_flat = torch.einsum("oib, bk -> oik", coeffs, self.basis)
            proper_size = int(self.basis.shape[1] ** 0.5)
            weight = weight_flat.view(coeffs.shape[0], coeffs.shape[1], proper_size, proper_size)
            # Perform convolution with stride=1 and padding=3 (as in the standard CNN).
            x = F.conv2d(x, weight, bias, stride=1, padding=3)
            x = F.relu(x)
            x = self.pool(x)
        
        # Instead of flattening spatially, apply global average pooling over height and width dimensions.
        x = x.mean(dim=(2, 3))  # now x has shape [batch_size, channels]
        
        # Fully connected layers (MLP).
        for i, linear in enumerate(self.linears):
            if i < len(self.linears) - 1:
                x = F.relu(linear(x))
            else:
                x = linear(x)
        return x


