# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 02:10:04 2022

@author: Mir Imtiaz Mostafiz

implements Hamiltonian Monte Carlo algorithm for Bayesian Averaging only for last layer
"""

import torch 
import numpy as np

import NeuralNetwork as mynn

class PerturbedHamiltonianMonteCarloSampler:
    
    def __init__(self, nnModel, rng, device):
        
        self.base_model = nnModel
        
        self.model = mynn.NeuralNetwork(nnModel.shape, nnModel.learning_rate).to(device)
        
        *_, self.last_layer_weight, self.last_layer_bias = self.model.parameters()
        
        #self.shapes = [p.shape for p in self.model.parameters()]
        self.shapes = [self.last_layer_weight.shape, self.last_layer_bias.shape]
        self.rng = rng
        pass

    def __repr__(self):
        
        ret = str(self.model)
        ret += "shapes" + str(self.shapes)
        return ret
    
    def get_sampled_velocities(self, stddv):
        """
        Sample random velocities from zero-mean Gaussian for last layer weight and biases.


        Parameters
        ----------
        stddv : float32
            standard deviation for all parameters sampling.

        Returns
        -------
        velocities : list of tensors with the same shape as each shape in self.shape sampled velocities for all parameters.


        """
        """
        TO DO COMPLETE IMPLEMENTATION
        """
        velocities = []

        velocities.append(torch.randn(self.last_layer_weight.shape).to(self.last_layer_weight.device) * stddv)
        velocities.append(torch.randn(self.last_layer_bias.shape).to(self.last_layer_bias.device) * stddv)

        return velocities


    def leapfrog(self, velocities, delta, *ARGS):
        """
        In-place leapfrog iteration.
        It should update `self.model`'s last layer wieghts and biases as position $x$ in
        HMC.
        It should update `velocities` as momentum $p$ in HMC.


        Parameters
        ----------
        velocities : list of length(2), float32
            sampled velocities for last layer wieghts and biases.
        delta : float32
            delta in HMC algorithm.
        *ARGS : (X, y, y_1hot) as described in utils.py and NeuralNetwork model learning

        Returns
        -------
        velocities : list of length(2), float32
            leapfrog updated velocities for last layer wieghts and biases.

        """
        """
        TO DO COMPLETE IMPLEMENTATION
        """
        last_layer_velocity, last_layer_bias_velocity = velocities
        grad_w, grad_b = torch.autograd.grad(self.model.energy(*ARGS), [self.last_layer_weight, self.last_layer_bias])
        grad_w = grad_w.to(next(self.model.parameters()).device)
        grad_b = grad_b.to(next(self.model.parameters()).device)
        last_layer_velocity -= 0.5 * delta * grad_w
        last_layer_bias_velocity -= 0.5 * delta * grad_b

        self.last_layer_weight.data += delta * last_layer_velocity
        self.last_layer_bias.data += delta * last_layer_bias_velocity

        grad_w, grad_b = torch.autograd.grad(self.model.energy(*ARGS), [self.last_layer_weight, self.last_layer_bias])
        grad_w = grad_w.to(next(self.model.parameters()).device)
        grad_b = grad_b.to(next(self.model.parameters()).device)

        last_layer_velocity -= 0.5 * delta * grad_w
        last_layer_bias_velocity -= 0.5 * delta * grad_b

        return [last_layer_velocity, last_layer_bias_velocity]




    def accept_or_reject(self, potential_energy_previous, potential_energy_current,
                         kinetic_energy_previous, kinetic_energy_current):
        """
        Given the potential and kinetic energies  of the last sample and new sample,
        check if we should accept new sample.
        If True, we will accept new sample.
        If False, we will reject new sample, and repeat the last sample.


        Parameters
        ----------
        potential_energy_previous : float32
            potential energy of last sample.
        potential_energy_current : float32
            potential energy of new sample.
        kinetic_energy_previous : float32
            kinetic energy of last sample.
        kinetic_energy_current : float32
            kinetic energy of new sample.

        Returns
        -------
        boolean
            True if to accept, False if to reject.

        """
        """
        TO DO COMPLETE IMPLEMENTATION
        """

        accepted_prob = torch.minimum(torch.tensor(1.0), torch.exp(torch.tensor(-(potential_energy_current - potential_energy_previous+kinetic_energy_current - kinetic_energy_previous))))
        return torch.rand(1)  < accepted_prob

    
    
    def sample(self, n, std_dev, delta, num_leapfrogs, /, *ARGS):
        """
        Sample from given parameters using Hamiltonian Monte Carlo.
        

        Parameters
        ----------
        n : int
            number of samples to generate.
        std_dev : float32
            standard deviation for sampling velocities.
        delta : float32
            delta in sampling velocities as in ALgorithm.
        num_leapfrogs : int
            number of leapfrog steps to do.
        *ARGS : (X, y, y_1hot) as described in utils.py and NeuralNetwork model learning
            
        Returns
        -------
        samples : list of length (1 + n), comprising of list of samples (model parameters) of length (self.model.shapes)
            initial and generated samples of model parameters.

        """
        # Initialize buffer.
        samples = []
        potentials = []

        # print(ARGS[0].shape)
        # Get initial sample from base model parameters
        inits = [param.data for param in self.base_model.parameters()]

        for (param, init) in zip(self.model.parameters(), inits):
            #
            param.data.copy_(init)

        with torch.no_grad():
            #
            nlf = self.model.energy(*ARGS).item()

        samples.append(
            [
                torch.clone(param.data.cpu())
                for param in self.model.parameters()
            ]
        )
        potentials.append(nlf)

        num_accepts = 0
        for i in range(1, n + 1):
            """
            this is running algorithm 1 for n times
            ke is $K(\Phi)$
            nlf is $U(\Phi)
            """
            # Sample a random velocity for last two layers.
            # Get corresponding potential and kenetic energies.
            velocities = []

            for param in self.model.parameters():
                velocities.append(torch.zeros(param.shape).to(param.device))

            # only use the last layers weights and biases
            velocities[-2:] = self.get_sampled_velocities(std_dev)
            potential_energy_previous = potentials[-1]
            kinetic_energy_previous = sum(0.5 * torch.sum(velocity ** 2).item() for velocity in velocities)

            # Update by multiple leapfrog steps to get a new sample.
            for _ in range(num_leapfrogs):
                #
                new_velocities_last = self.leapfrog(velocities[-2:], delta, *ARGS)
                new_velocities = velocities[:-2] + new_velocities_last

            with torch.no_grad():
                #
                potential_energy_current = self.model.energy(*ARGS).item()

            kinetic_energy_current = sum(0.5 * torch.sum(new_velocity ** 2).item() for new_velocity in new_velocities)

            # Metropolis-Hasting rejection sampling.
            accept_new = self.accept_or_reject(potential_energy_previous, potential_energy_current,
                                               kinetic_energy_previous, kinetic_energy_current)
            if accept_new:
                # Accept new samples.
                samples.append(
                    [
                        torch.clone(param.data.cpu())
                        for param in self.model.parameters()
                    ],
                )
                potentials.append(potential_energy_current)
                print(
                    "{:>3d} {:>6s} {:>8s}"
                    .format(i, "Accept", "{:.6f}".format(potential_energy_current)),
                )
            else:
                # Reject new samples.
                # Need to recover model parameters back to the last sample.
                samples.append(samples[-1])
                for (param, init) in zip(self.model.parameters(), samples[-1]):
                    #
                    param.data.copy_(init)
                potentials.append(potential_energy_previous)
                print(
                    "{:>3d} {:>6s} {:>8s}"
                    .format(i, "Reject", "{:.6f}".format(potential_energy_previous)),
                )
            num_accepts = num_accepts + int(accept_new)
        print("{:s} {:s}".format("-" * 3, "-" * 6))
        print("- Accept%: {:.1f}%".format(float(num_accepts) * 100 / float(n)))
        return samples
    
    