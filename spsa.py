import torch
import copy




class SPSA:

    def __init__(self, model, loss_fn, lr=0.01, delta=0.01):
        self.model = model
        self.loss_fn = loss_fn
        self.lr = lr
        self.delta = delta
        self.noise_scale = 0.01

        self.parameters = list(self.model.parameters())
        print("Parameters:", self.parameters[0].shape)
        print("Parameters true", self.model.parameters())
        self.dimensions = self.get_dimensions()
        print(self.dimensions)
        
        self._init_randomizer()


    def _init_randomizer(self):
        self.randomizer = []
        for param in self.parameters:
            self.randomizer.append(torch.distributions.bernoulli.Bernoulli(param.data.new_ones(param.shape) * 0.5))

    def _randomize_parameters(self):
        return [r.sample()-0.5 for r in self.randomizer]


    def get_dimensions(self):
        return sum(p.numel() for p in self.parameters if p.requires_grad)

    def forward(self, x):
        return self.model(x)

    def _perturb_parameters(self, direction, magnitude):
        for d, param in zip(direction, self.model.parameters()):
            param.data += d * magnitude


    def step(self, x, y):
        x = x.clone().detach().requires_grad_(True)
        y = y.clone().detach()

        # Random perturbation
        perturbation = self._randomize_parameters()

        
        self._perturb_parameters(perturbation, self.delta)
        
        left = self.loss_fn(self.forward(x), y)

        self._perturb_parameters(perturbation, -2 * self.delta)
        right = self.loss_fn(self.forward(x), y)
        self._perturb_parameters(perturbation, self.delta)

        
        # Compute gradient approximation
        gradient = [(left - right) / (2 * self.delta*p) + p*self.noise_scale for p in perturbation]
        # Update parameters
    
        self._perturb_parameters(gradient, -self.lr)




class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc = torch.nn.Linear(10, 20)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(20, 1)

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    model = Model()
    loss_fn = torch.nn.MSELoss()
    spsa_optimizer = SPSA(model, loss_fn, lr=0.01, delta=0.01)


    x = torch.randn(5, 10)
    y = torch.randn(5, 1)

    for _ in range(100):
        spsa_optimizer.step(x, y)
        #print("Updated parameters:", [p.data for p in model.parameters()])
        print("Loss:", loss_fn(model(x), y).item())



