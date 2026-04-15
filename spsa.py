import torch
import copy
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


#################################
## CONSTANTS
#################################


#################################
## Class definitions
#################################


class SPSA:
    def __init__(self, model, loss_fn, lr, delta, noise_factor=0):
        self.model = model
        self.loss_fn = loss_fn
        self.lr = lr
        self.delta = delta
        self.noise_factor = noise_factor

        self.params = [p for p in model.parameters() if p.requires_grad]

    def _flatten(self):
        return torch.cat([p.data.view(-1) for p in self.params])

    def _unflatten(self, flat):
        idx = 0
        for p in self.params:
            n = p.numel()
            p.data.copy_(flat[idx:idx+n].view_as(p))
            idx += n

    def step(self, x, y):
        if isinstance(x, dict):
            x = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in x.items()}
        else:
            x = x.to(device)
        y = y.to(device)

        theta = self._flatten()
        dim = theta.numel()

        # SPSA direction
        delta_vec = torch.randint(0, 2, (dim,), device=device).float()
        delta_vec = 2 * delta_vec - 1

        theta_plus = theta + self.delta * delta_vec
        theta_minus = theta - self.delta * delta_vec

        # f(theta+)
        self._unflatten(theta_plus)
        loss_plus = self.loss_fn(self.model(x), y)

        # f(theta-)
        self._unflatten(theta_minus)
        loss_minus = self.loss_fn(self.model(x), y)

        # restore original
        self._unflatten(theta)

        # gradient estimate
        grad_est = (loss_plus - loss_minus) / (2 * self.delta) * delta_vec + self.noise_factor*delta_vec

        # update
        new_theta = theta - self.lr * grad_est
        self._unflatten(new_theta)

#################################
## Functions
#################################




#################################
## Test function
#################################

if __name__ == "__main__":
    print("-----------------------------------")
    print("FITTING TO RANDOM NOISE")
    print("-----------------------------------")

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
    model = Model()
    loss_fn = torch.nn.MSELoss()
    spsa_optimizer = SPSA(model, loss_fn, lr=0.01, delta=0.01, noise_factor=0.1)


    x = torch.randn(5, 10)
    y = torch.randn(5, 1)

    first = loss_fn(model(x), y).item()
    for _ in range(100):
        spsa_optimizer.step(x, y)
    last = loss_fn(model(x), y).item()
    print("First: ", first, " Last", last, " diff: ", first-last)
