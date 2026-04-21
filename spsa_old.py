import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


#################################
## CONSTANTS
#################################


#################################
## Class definitions
#################################


class SPSA:
    def __init__(
        self,
        model,
        loss_fn=None,
        lr=1e-3,
        delta=1e-3,
        noise_factor=0,
        num_estimates=8,
        alpha=None,
        gamma=None,
        A=0,
    ):
        """Simultaneous Perturbation Stochastic Approximation optimizer.

        Parameters
        ----------
        model : torch.nn.Module
            The model whose ``requires_grad`` parameters will be optimized.
        loss_fn : callable or None, optional
            Unused by the built-in ``step`` / ``step_with_closure`` methods but
            kept for backward compatibility.  Defaults to ``None``.
        lr : float
            Base learning rate (``a`` in Spall 1992).  When ``alpha`` is set
            this is the initial scale; it decays as ``lr / (A + t)^alpha``.
        delta : float, optional
            Base perturbation magnitude (``c`` in Spall 1992).  Default is
            ``1e-3``, which is appropriate for pre-trained BERT-scale weights.
            When ``gamma`` is set it decays as ``delta / t^gamma``.
        noise_factor : float, optional
            Kept for backward compatibility; no longer added to the gradient
            estimate.  Pass ``0`` (default) to disable.
        num_estimates : int, optional
            Number of independent Rademacher directions to average per step.
            Higher values reduce gradient variance at the cost of extra forward
            passes.  Default is ``1``; values of 4–8 are recommended for large
            models such as BERT.
        alpha : float or None, optional
            Decay exponent for the learning rate schedule
            ``lr(t) = lr / (A + t)^alpha``.  The theoretically optimal value
            (Spall 1992) is ``0.602``.  ``None`` (default) disables decay and
            keeps a fixed learning rate.
        gamma : float or None, optional
            Decay exponent for the perturbation schedule
            ``delta(t) = delta / t^gamma``.  The theoretically optimal value
            (Spall 1992) is ``0.101``.  ``None`` (default) disables decay and
            keeps a fixed perturbation size.
        A : float, optional
            Stability constant for the learning-rate schedule (see Spall 1992).
            Typical choice is roughly 10 % of the total number of iterations.
            Default is ``0``.
        """
        self.model = model
        self.loss_fn = loss_fn
        self.lr = lr
        self.delta = delta
        self.noise_factor = noise_factor
        self.num_estimates = num_estimates
        self.alpha = alpha
        self.gamma = gamma
        self.A = A
        self.t = 0  # step counter used for decay schedules

        # Deduplicate: tied weights (e.g. BERT input/output embeddings) appear
        # more than once in model.parameters().  Including duplicates inflates
        # `dim`, makes the gradient estimate inconsistent for those parameters,
        # and—critically—multiplies the total number of perturbed dimensions,
        # which drives the SPSA signal-to-noise ratio to near zero for large
        # models.  Keep only the first occurrence of each unique tensor.
        seen_ids: set[int] = set()
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

    def _current_lr(self):
        """Return the (possibly decayed) learning rate for the current step."""
        if self.alpha is not None:
            return self.lr / (self.A + self.t + 1) ** self.alpha
        return self.lr

    def _current_delta(self):
        """Return the (possibly decayed) perturbation magnitude for the current step."""
        if self.gamma is not None:
            return self.delta / (self.t + 1) ** self.gamma
        return self.delta

    def step(self, batch):
        # Switch to eval to suppress stochastic layers (dropout, etc.) during
        # the two perturbation evaluations.  Without this, BERT's dropout
        # produces different random masks for f(θ+δ) and f(θ−δ), injecting
        # noise that dominates the gradient signal.
        training = self.model.training
        self.model.eval()

        theta = self._flatten()
        dim = theta.numel()
        lr = self._current_lr()
        delta = self._current_delta()
        self.t += 1

        # Average `num_estimates` independent Rademacher perturbations to
        # reduce gradient variance at the cost of extra forward passes.
        grad_est = torch.zeros_like(theta)
        for _ in range(self.num_estimates):
            # SPSA direction (Rademacher ±1). Use integer randint then cast to
            # the parameter dtype; passing a float dtype directly to randint is
            # not portable across PyTorch versions.
            delta_vec = (torch.randint(0, 2, (dim,), device=theta.device) * 2 - 1).to(dtype=theta.dtype)

            # f(theta+)
            self._unflatten(theta + delta * delta_vec)
            loss_plus = self.model(**batch).loss

            # f(theta-)
            self._unflatten(theta - delta * delta_vec)
            loss_minus = self.model(**batch).loss

            grad_est += (loss_plus - loss_minus) / (2 * delta) * delta_vec

        grad_est /= self.num_estimates
        grad_est = grad_est / (grad_est.norm() + 1e-8)

        # Restore original weights before applying the update.
        self._unflatten(theta)

        # Restore training mode before the final update and loss evaluation so
        # that downstream code (e.g. batch-norm running stats) behaves as
        # expected.
        if training:
            self.model.train()

        # Update parameters.
        new_theta = theta - lr * grad_est
        self._unflatten(new_theta)
        # Return training-mode loss for logging.  This is computed with dropout
        # active (if the model was in train mode before the step), which is
        # consistent with how training loss is typically reported.
        return self.model(**batch).loss

    def step_with_closure(self, loss_closure):
        """Run one SPSA update using a closure that returns a scalar loss tensor.

        Returns:
            torch.Tensor: The mean of f(theta + delta) and f(theta - delta), used
            as a low-cost per-step loss estimate for logging.
        """
        # Switch to eval to suppress stochastic layers (dropout, etc.) during
        # the two perturbation evaluations.  Without this, BERT's dropout
        # produces different random masks for f(θ+δ) and f(θ−δ), injecting
        # noise that dominates the gradient signal.
        training = self.model.training
        self.model.eval()

        theta = self._flatten()
        dim = theta.numel()
        lr = self._current_lr()
        delta = self._current_delta()
        self.t += 1

        # Average `num_estimates` independent Rademacher perturbations to
        # reduce gradient variance at the cost of extra forward passes.
        grad_est = torch.zeros_like(theta)
        for _ in range(self.num_estimates):
            # SPSA direction (Rademacher ±1). Use integer randint then cast to
            # the parameter dtype; passing a float dtype directly to randint is
            # not portable across PyTorch versions.
            delta_vec = (torch.randint(0, 2, (dim,), device=theta.device) * 2 - 1).to(dtype=theta.dtype)

            # f(theta+)
            self._unflatten(theta + delta * delta_vec)
            loss_plus = loss_closure()

            # f(theta-)
            self._unflatten(theta - delta * delta_vec)
            loss_minus = loss_closure()

            grad_est += (loss_plus - loss_minus) / (2 * delta) * delta_vec

        grad_est /= self.num_estimates

        # Restore original weights before applying the update.
        self._unflatten(theta)

        # Restore training mode before the final update and loss evaluation.
        if training:
            self.model.train()

        # Update parameters.
        new_theta = theta - lr * grad_est
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
            super().__init__()
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
