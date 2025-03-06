import torch 
import NeuralNetwork as mynn
import numpy as np

DEBUG_ACCEPT = False

class HamiltonianMonteCarloSampler:
    
    def __init__(self, nnModel, rng, device):
        self.base_model = nnModel
        self.model = mynn.NeuralNetwork(nnModel.shape, nnModel.learning_rate).to(device)
        self.shapes = [p.shape for p in self.model.parameters()]
        self.rng = rng
        self.device = device  # store the device
        
    def __repr__(self):
        ret = str(self.model)
        ret += " shapes: " + str(self.shapes)
        return ret
        
    def get_sampled_velocities(self, stddv):
        """
        Sample random velocities from zero-mean Gaussian for all parameters.
        """
        velocities = []
        for shape in self.shapes:
            velocity = torch.normal(mean=0.0, std=stddv, size=shape).to(self.device)
            velocities.append(velocity)
        return velocities
    
    def leapfrog(self, velocities, delta, /, *ARGS):
        """
        In-place leapfrog iteration.
        """
        # Half step for velocities
        grads = torch.autograd.grad(self.model.energy(*ARGS), self.model.parameters())

        for i, velocity in enumerate(velocities):
            velocities[i] = velocity - delta / 2 * grads[i]

        # Full step for positions (model parameters)
        for i, param in enumerate(self.model.parameters()):
            param.data = param.data + delta * velocities[i]

        # Half step for velocities again
        grads = torch.autograd.grad(self.model.energy(*ARGS), self.model.parameters())
        for i, velocity in enumerate(velocities):
            velocities[i] = velocity - delta / 2 * grads[i]

        return velocities
        

    def accept_or_reject(self, potential_energy_previous, potential_energy_current, 
                         kinetic_energy_previous, kinetic_energy_current):
        """
        Given the potential and kinetic energies  of the last sample and new sample, 
        check if we should accept new sample.
        """
        # Compute the total energy for both the current and previous samples.
        previous_energy = torch.tensor(potential_energy_previous + kinetic_energy_previous)
        current_energy = torch.tensor(potential_energy_current + kinetic_energy_current)
    
    # Compute the acceptance probability.
        acceptance_prob = min(1.0, torch.exp(previous_energy - current_energy).item())
    
    # Accept the new sample with probability acceptance_prob.
        return torch.rand(1).item() < acceptance_prob 
 
    
    def sample(self, n, std_dev, delta, num_leapfrogs, /, *ARGS):
        """
        Sample from the model's parameter space using Hamiltonian Monte Carlo.
        (This method is provided and uses the above functions.)
        """
        # Initialize buffers.
        samples = []
        potentials = []
        
        # Copy initial parameters from the base model.
        inits = [param.data for param in self.base_model.parameters()]
        for (param, init) in zip(self.model.parameters(), inits):
            param.data.copy_(init)
        
        with torch.no_grad():
            nlf = self.model.energy(*ARGS).item()
        samples.append([torch.clone(param.data.cpu()) for param in self.model.parameters()])
        potentials.append(nlf)
        
        num_accepts = 0
        for i in range(1, n + 1):
            # Sample a random velocity.
            velocities = self.get_sampled_velocities(std_dev)
            potential_energy_previous = potentials[-1]
            kinetic_energy_previous = sum(0.5 * torch.sum(v ** 2).item() for v in velocities)
            
            # Update by multiple leapfrog steps.
            for _ in range(num_leapfrogs):
                velocities = self.leapfrog(velocities, delta, *ARGS)
            
            with torch.no_grad():
                potential_energy_current = self.model.energy(*ARGS).item()
            kinetic_energy_current = sum(0.5 * torch.sum(v ** 2).item() for v in velocities)
            
            # Metropolis-Hastings acceptance.
            accept_new = self.accept_or_reject(potential_energy_previous, potential_energy_current,
                                               kinetic_energy_previous, kinetic_energy_current)
            if accept_new:
                samples.append([torch.clone(param.data.cpu()) for param in self.model.parameters()])
                potentials.append(potential_energy_current)
                print("{:>3d} {:>6s} {:>8s}".format(i, "Accept", "{:.6f}".format(potential_energy_current)))
            else:
                # Revert to previous parameters.
                samples.append(samples[-1])
                for (param, init) in zip(self.model.parameters(), samples[-1]):
                    param.data.copy_(init)
                potentials.append(potential_energy_previous)
                print("{:>3d} {:>6s} {:>8s}".format(i, "Reject", "{:.6f}".format(potential_energy_previous)))
            num_accepts += int(accept_new)
        print("{:s} {:s}".format("-" * 3, "-" * 6))
        print("- Accept%: {:.1f}%".format(float(num_accepts) * 100 / float(n)))
        return samples
