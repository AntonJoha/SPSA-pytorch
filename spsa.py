import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


#################################
## CONSTANTS
#################################


#################################
## Class definitions
#################################

class SPSA:
    def __init__(self, model, lr, delta, noise_factor, loss_fn):
        self.model = model
        self.lr = lr
        self.delta = delta
        self.noise_factor = noise_factor
        self.loss_fn = loss_fn
        self._make_param_list()

    def _flatten(self, weights):
        a = torch.cat([w.flatten() for w in weights])
        return a
    
    def get_weights(self):
        return self._flatten([param.data for param in self.params])


    def _make_param_list(self):
        """
        Make a list of the model parameters that require gradients.
        """
        param_list = []
        for p in self.model.parameters():
            if p.requires_grad:
                param_list.append(p.data)
        self.params = param_list

    def _set_weights(self, new_weights):
        """
        Set the model parameters to the new weights.
        """
        idx = 0
        for param in self.params:
            numel = param.numel()
            param.copy_(new_weights[idx:idx+numel].view_as(param))
            idx += numel
    def get_model(self):
        return self.model

    def step(self, x, y, attention_mask=None):

        weights = self.get_weights()

        self.model.eval()

        with torch.no_grad():
            def loss_closure():
                return self.loss_fn(self.model(x, attention_mask=attention_mask), y)

            delta_vec = (torch.randint(0, 2, weights.shape, device=weights.device) * 2 - 1).to(dtype=weights.dtype)

            weights_plus = weights + self.delta * delta_vec
            weights_minus = weights - self.delta * delta_vec
            
            self._set_weights(weights_plus)
            loss_plus = loss_closure()
            self._set_weights(weights_minus)
            loss_minus = loss_closure()

            diff = loss_plus - loss_minus
            grad_est = diff / (2 * self.delta) * delta_vec
        new_weights = weights - self.lr * grad_est
        self._set_weights(new_weights)
        return loss_closure()




#################################
## Private functions
#################################


#################################
## Public functions
#################################


#################################
## Test code
#################################

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(10, 20)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(20, 1)

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    loss_fn = torch.nn.MSELoss()
    model = Model().to(device)
    spsa_optimizer = SPSA(model, lr=0.01, delta=0.01, noise_factor=0.1, loss_fn=loss_fn)

    x = torch.randn(50, 10).to(device)
    y = torch.randn(50, 1).to(device)
    
    first = loss_fn(model(x), y).item()
    for _ in range(1000):
        spsa_optimizer.step(x, y).item()
    last = loss_fn(model(x), y).item()
    print("First: ", first, " Last", last, " diff: ", first-last)
    assert last < first, "Loss did not decrease"

