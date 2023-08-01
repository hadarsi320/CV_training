from __future__ import print_function

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F

NOISE_DIM = 96


def hello_gan():
    print("Hello from gan.py!")


def sample_noise(batch_size, noise_dim, dtype=torch.float, device="cpu"):
    """
    Generate a PyTorch Tensor of uniform random noise.

    Input:
    - batch_size: Integer giving the batch size of noise to generate.
    - noise_dim: Integer giving the dimension of noise to generate.

    Output:
    - A PyTorch Tensor of shape (batch_size, noise_dim) containing uniform
      random noise in the range (-1, 1).
    """
    noise = None
    ##############################################################################
    # TODO: Implement sample_noise.                                              #
    ##############################################################################
    # Replace "pass" statement with your code
    noise = torch.rand((batch_size, noise_dim), device=device, dtype=dtype)
    noise = noise * 2 - 1
    ##############################################################################
    #                              END OF YOUR CODE                              #
    ##############################################################################

    return noise


def discriminator():
    """
    Build and return a PyTorch nn.Sequential model implementing the architecture
    in the notebook.
    """
    model = None
    ############################################################################
    # TODO: Implement discriminator.                                           #
    ############################################################################
    # Replace "pass" statement with your code
    dims = [256, 256, 1]
    last_dim = 784
    layers = []
    for dim in dims:
        layers.append(nn.Linear(last_dim, dim))
        if dim != 1:
            layers.append(nn.LeakyReLU(0.01))
        last_dim = dim
    model = nn.Sequential(*layers)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    return model


def generator(noise_dim=NOISE_DIM):
    """
    Build and return a PyTorch nn.Sequential model implementing the architecture
    in the notebook.
    """
    model = None
    ############################################################################
    # TODO: Implement generator.                                               #
    ############################################################################
    # Replace "pass" statement with your code
    dims = [1024, 1024, 784]
    last_dim = noise_dim
    layers = []
    for dim in dims:
        layers.append(nn.Linear(last_dim, dim))
        if dim != 784:
            layers.append(nn.ReLU())
        else:
            layers.append(nn.Tanh())
        last_dim = dim
    model = nn.Sequential(*layers)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return model


def discriminator_loss(logits_real, logits_fake):
    """
    Computes the discriminator loss described above.

    Inputs:
    - logits_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Returns:
    - loss: PyTorch Tensor containing (scalar) the loss for the discriminator.
    """
    loss = None
    ##############################################################################
    # TODO: Implement discriminator_loss.                                        #
    ##############################################################################
    # Replace "pass" statement with your code
    labels_real = torch.ones_like(logits_real)
    labels_fake = torch.zeros_like(logits_fake)
    loss = (F.binary_cross_entropy_with_logits(logits_real, labels_real) + 
            F.binary_cross_entropy_with_logits(logits_fake, labels_fake))
    ##############################################################################
    #                              END OF YOUR CODE                              #
    ##############################################################################
    return loss


def generator_loss(logits_fake):
    """
    Computes the generator loss described above.

    Inputs:
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Returns:
    - loss: PyTorch Tensor containing the (scalar) loss for the generator.
    """
    loss = None
    ##############################################################################
    # TODO: Implement generator_loss.                                            #
    ##############################################################################
    # Replace "pass" statement with your code
    labels = torch.ones_like(logits_fake)
    loss = F.binary_cross_entropy_with_logits(logits_fake, labels)
    ##############################################################################
    #                              END OF YOUR CODE                              #
    ##############################################################################
    return loss


def get_optimizer(model):
    """
    Construct and return an Adam optimizer for the model with learning rate 1e-3,
    beta1=0.5, and beta2=0.999.

    Input:
    - model: A PyTorch model that we want to optimize.

    Returns:
    - An Adam optimizer for the model with the desired hyperparameters.
    """
    optimizer = None
    ##############################################################################
    # TODO: Implement optimizer.                                                 #
    ##############################################################################
    # Replace "pass" statement with your code
    optimizer = optim.Adam(model.parameters(), 1e-3, betas=(0.5, 0.999))
    ##############################################################################
    #                              END OF YOUR CODE                              #
    ##############################################################################
    return optimizer


def ls_discriminator_loss(scores_real, scores_fake):
    """
    Compute the Least-Squares GAN loss for the discriminator.

    Inputs:
    - scores_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """
    loss = None
    ##############################################################################
    # TODO: Implement ls_discriminator_loss.                                     #
    ##############################################################################
    # Replace "pass" statement with your code
    loss = 1/2 * ((scores_real - 1).square().mean() + (scores_fake).square().mean())
    ##############################################################################
    #                              END OF YOUR CODE                              #
    ##############################################################################
    return loss


def ls_generator_loss(scores_fake):
    """
    Computes the Least-Squares GAN loss for the generator.

    Inputs:
    - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """
    loss = None
    ##############################################################################
    # TODO: Implement ls_generator_loss.                                         #
    ##############################################################################
    # Replace "pass" statement with your code
    loss = 1/2 * (scores_fake - 1).square().mean()
    ##############################################################################
    #                              END OF YOUR CODE                              #
    ##############################################################################
    return loss


def build_dc_classifier():
    """
    Build and return a PyTorch nn.Sequential model for the DCGAN discriminator
    implementing the architecture in the notebook.
    """
    model = None
    ############################################################################
    # TODO: Implement build_dc_classifier.                                     #
    ############################################################################
    # Replace "pass" statement with your code
    # Reshape into image tensor (Use nn.Unflatten!)
    # Conv2D: 32 Filters, 5x5, Stride 1
    # Leaky ReLU(alpha=0.01)
    # Max Pool 2x2, Stride 2
    # Conv2D: 64 Filters, 5x5, Stride 1
    # Leaky ReLU(alpha=0.01)
    # Max Pool 2x2, Stride 2
    # Flatten
    # Fully Connected with output size 4 x 4 x 64
    # Leaky ReLU(alpha=0.01)
    # Fully Connected with output size 1

    layers = [nn.Unflatten(1, (1, 28, 28))]
    last_channels = 1
    for channels in [32, 64]:
        layers.extend([
            nn.Conv2d(last_channels, channels, 5),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2, 2)
        ])
        last_channels = channels
    layers.extend([
        nn.Flatten(1),
        nn.Linear(1024, 1024),
        nn.LeakyReLU(0.01),
        nn.Linear(1024, 1)
    ])
    model = nn.Sequential(*layers)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return model


def build_dc_generator(noise_dim=NOISE_DIM):
    """
    Build and return a PyTorch nn.Sequential model implementing the DCGAN
    generator using the architecture described in the notebook.
    """
    model = None
    ############################################################################
    # TODO: Implement build_dc_generator.                                      #
    ############################################################################
    # Replace "pass" statement with your code
    

    # Fully connected with output size 1024
    # ReLU
    # BatchNorm
    # Fully connected with output size 7 x 7 x 128
    # ReLU
    # BatchNorm
    # Reshape into Image Tensor of shape 7 x 7 x 128
    # Conv2D^T (Transpose): 64 filters of 4x4, stride 2, 'same' padding (use padding=1)
    # ReLU
    # BatchNorm
    # Conv2D^T (Transpose): 1 filter of 4x4, stride 2, 'same' padding (use padding=1)
    # TanH
    # Should have a 28 x 28 x 1 image, reshape back into 784 vector

    layers = []
    last_dims = noise_dim
    for dim in [1024, 7 * 7 * 128]:
        layers.extend([
            nn.Linear(last_dims, dim),
            nn.ReLU(),
            nn.BatchNorm1d(dim)
        ])
        last_dims = dim
    layers.extend([
        nn.Unflatten(1, (128, 7, 7)),
        nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1),
        nn.Tanh(),
        nn.Flatten()
    ])
    model = nn.Sequential(*layers)

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return model
