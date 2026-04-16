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

        # Deduplicate: tied weights (e.g. BERT input/output embeddings) appear
        # more than once in model.parameters().  Including duplicates inflates
        # `dim`, makes the gradient estimate inconsistent for those parameters,
        # and—critically—multiplies the total number of perturbed dimensions,
        # which drives the SPSA signal-to-noise ratio to near zero for large
        # models.  Keep only the first occurrence of each unique tensor.
        seen_ids: set = set()
        self.params = []
        for p in model.parameters():
            if p.requires_grad and id(p) not in seen_ids:
                seen_ids.add(id(p))
                self.params.append(p)

    def _flatten(self):
        return torch.cat([p.data.view(-1) for p in self.params])

    def _unflatten(self, flat):
        idx = 0
        for p in self.params:
            n = p.numel()
            p.data.copy_(flat[idx:idx+n].view_as(p))
            idx += n

    def step(self, batch):
        

        theta = self._flatten()
        dim = theta.numel()

        # SPSA direction (Rademacher ±1). Use integer randint then cast to the
        # parameter dtype; passing a float dtype directly to randint is not
        # portable across PyTorch versions.
        delta_vec = (torch.randint(0, 2, (dim,), device=theta.device) * 2 - 1).to(dtype=theta.dtype)

        theta_plus = theta + self.delta * delta_vec
        theta_minus = theta - self.delta * delta_vec

        # f(theta+)
        self._unflatten(theta_plus)
        loss_plus = self.model(**batch).loss

        # f(theta-)
        self._unflatten(theta_minus)
        loss_minus = self.model(**batch).loss

        # restore original
        self._unflatten(theta)

        # gradient estimate
        grad_est = (loss_plus - loss_minus) / (2 * self.delta) * delta_vec + self.noise_factor * delta_vec

        # update
        new_theta = theta - self.lr * grad_est
        self._unflatten(new_theta)
        return self.model(**batch).loss

    def step_with_closure(self, loss_closure):
        """Run one SPSA update using a closure that returns a scalar loss tensor.

        Returns:
            torch.Tensor: The mean of f(theta + delta) and f(theta - delta), used
            as a low-cost per-step loss estimate for logging.
        """
        theta = self._flatten()
        dim = theta.numel()

        # SPSA direction (Rademacher ±1). Use integer randint then cast to the
        # parameter dtype; passing a float dtype directly to randint is not
        # portable across PyTorch versions.
        delta_vec = (torch.randint(0, 2, (dim,), device=theta.device) * 2 - 1).to(dtype=theta.dtype)

        theta_plus = theta + self.delta * delta_vec
        theta_minus = theta - self.delta * delta_vec

        # f(theta+)
        self._unflatten(theta_plus)
        loss_plus = loss_closure()

        # f(theta-)
        self._unflatten(theta_minus)
        loss_minus = loss_closure()

        # restore original
        self._unflatten(theta)

        # gradient estimate
        # Optional isotropic noise can be added to the update direction to
        # encourage exploration and reduce deterministic update bias.
        grad_est = (loss_plus - loss_minus) / (2 * self.delta) * delta_vec + self.noise_factor * delta_vec

        # update
        new_theta = theta - self.lr * grad_est
        self._unflatten(new_theta)
        return loss_closure()

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
