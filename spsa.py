import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


#################################
## CONSTANTS
#################################


#################################
## Class definitions
#################################

class SPSA:
    def __init__(self, model, lr, delta, loss_fn, noise_factor=0.0, num_estimates=1):
        self.model = model
        self.lr = lr
        self.delta = delta
        self.noise_factor = noise_factor
        self.num_estimates = num_estimates
        self.loss_fn = loss_fn
        self._make_param_list()
        self.stability_constant = 10
        self.alpha = 0.602
        self.gamma = 0.101
        self.optimizer = None

    def init_sgd(self, lr):
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        
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

    def _optim_step(self,x, y, epoch, attention_mask):

        self.optimizer.zero_grad()

        loss = self.loss_fn(self.model(x,attention_mask=attention_mask), y)
        loss.backward()
        self.optimizer.step()
        return self.loss_fn(self.model(x,attention_mask=attention_mask), y)

    def step(self, x, y, epoch, attention_mask=None):
        if self.optimizer is not None:
            return self._optim_step(x,y,epoch,attention_mask)
        weights = self.get_weights()

        self.model.eval()
        ak = self.lr/ (epoch + self.stability_constant) ** self.alpha
        ck = self.delta / (epoch**self.gamma)
        with torch.no_grad():
            def loss_closure():
                return self.loss_fn(self.model(x, attention_mask=attention_mask), y)

            grad_est = torch.zeros_like(weights)
            try:
                for _ in range(self.num_estimates):
                    delta_vec = (torch.randint(0, 2, weights.shape, device=weights.device) * 2 - 1).to(dtype=weights.dtype)

                    self._set_weights(weights + ck * delta_vec)
                    loss_plus = loss_closure()
                    self._set_weights(weights - ck * delta_vec)
                    loss_minus = loss_closure()

                    grad_est += (loss_plus - loss_minus) / (2 * ck) * delta_vec

                grad_est /= self.num_estimates
            finally:
                # Always restore original weights before applying the gradient update.
                self._set_weights(weights)
        new_weights = weights - ak * grad_est
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
    spsa_optimizer = SPSA(model, lr=0.05, delta=0.01, loss_fn=loss_fn)

    x = torch.randn(50, 10).to(device)
    y = torch.randn(50, 1).to(device)
    
    first = loss_fn(model(x), y).item()
    for i in range(2000):
        spsa_optimizer.step(x, y, i +1).item()
        #loss = loss_fn(model(x), y).item()
        #print(f"Epoch {i+1}, Loss: {loss}")

    last = loss_fn(model(x), y).item()
    print("First: ", first, " Last", last, " diff: ", first-last)
    assert last < first, "Loss did not decrease"

