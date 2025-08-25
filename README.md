# **Cooper**

## What is **Cooper**?

**Cooper** is a library for solving constrained optimization problems in [PyTorch](https://github.com/pytorch/pytorch).

**Cooper** implements several Lagrangian-based (first-order) update schemes that are applicable to a wide range of continuous constrained optimization problems. **Cooper** is mainly targeted for deep learning applications, where gradients are estimated based on mini-batches, but it is also suitable for general continuous constrained optimization tasks.

There exist other libraries for constrained optimization in PyTorch, like [CHOP](https://github.com/openopt/chop) and [GeoTorch](https://github.com/Lezcano/geotorch), but they rely on assumptions about the constraints (such as admitting efficient projection or proximal operators). These assumptions are often not met in modern machine learning problems. **Cooper** can be applied to a wider range of constrained optimization problems (including non-convex problems) thanks to its Lagrangian-based approach.

You can check out **Cooper**'s FAQ [here](#faq).

**Cooper**'s companion paper is available [here](https://arxiv.org/abs/2504.01212).

- [**Cooper**](#cooper)
  - [What is **Cooper**?](#what-is-cooper)
  - [Installation](#installation)
  - [Getting Started](#getting-started)
    - [Quick Start](#quick-start)
    - [Example](#example)
  - [Contributions](#contributions)
  - [Acknowledgements](#acknowledgements)
  - [License](#license)
  - [How to cite **Cooper**](#how-to-cite-cooper)


## Installation

To install the latest release of **Cooper**, use the following command:

```bash
pip install .
```

## Getting Started


### Quick Start

To use **Cooper**, you need to:

- Implement a `ConstrainedMinimizationProblem` (CMP) class and its associated `ConstrainedMinimizationProblem.compute_cmp_state` method. This method computes the value of the objective function and constraint violations, and packages them in a `CMPState` object.
- The initialization of the `CMP` must create a `Constraint` object for each constraint. It is necessary to specify a formulation type (e.g. `Lagrangian`). Finally, if the chosen formulation requires it, each constraint needs an associated `Multiplier` object corresponding to the Lagrange multiplier for that constraint.
- Create a `torch.optim.Optimizer` for the primal variables and a `torch.optim.Optimizer(maximize=True)` for the dual variables (i.e. the multipliers). Then, wrap these two optimizers in a `cooper.optim.CooperOptimizer` (such as `SimultaneousOptimizer` for executing simultaneous primal-dual updates).
- You are now ready to perform updates on the primal and dual parameters using the `CooperOptimizer.roll()` method. This method triggers the following calls:
  - `zero_grad()` on both optimizers,
  - `compute_cmp_state()` on the `CMP`,
  - compute the Lagrangian based on the latest `CMPState`,
  - `backward()` on the Lagrangian,
  - `step()` on both optimizers.
- To access the value of the loss, constraint violations, and Lagrangian terms, you can inspect the returned `RollOut` object from the call to `roll()`.

### Example

This is an abstract example on how to solve a constrained optimization problem with
**Cooper**. You can find runnable notebooks with concrete examples in our **Tutorials**.

```python
import cooper
import torch

# Set up GPU acceleration
DEVICE = ...

class MyCMP(cooper.ConstrainedMinimizationProblem):
    def __init__(self):
        super().__init__()
        multiplier = cooper.multipliers.DenseMultiplier(num_constraints=..., device=DEVICE)
        # By default, constraints are built using `formulation_type=cooper.formulations.Lagrangian`
        self.constraint = cooper.Constraint(
            multiplier=multiplier, constraint_type=cooper.ConstraintType.INEQUALITY
        )

    def compute_cmp_state(self, model, inputs, targets):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        loss = ...
        constraint_state = cooper.ConstraintState(violation=...)
        observed_constraints = {self.constraint: constraint_state}

        return cooper.CMPState(loss=loss, observed_constraints=observed_constraints)


train_loader = ...
model = (...).to(DEVICE)
cmp = MyCMP()

primal_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# Must set `maximize=True` since the Lagrange multipliers solve a _maximization_ problem
dual_optimizer = torch.optim.SGD(cmp.dual_parameters(), lr=1e-2, maximize=True)

cooper_optimizer = cooper.optim.SimultaneousOptimizer(
    cmp=cmp, primal_optimizers=primal_optimizer, dual_optimizers=dual_optimizer
)

for epoch_num in range(NUM_EPOCHS):
    for inputs, targets in train_loader:
        # `roll` is a convenience function that packages together the evaluation
        # of the loss, call for gradient computation, the primal and dual updates and zero_grad
        compute_cmp_state_kwargs = {"model": model, "inputs": inputs, "targets": targets}
        roll_out = cooper_optimizer.roll(compute_cmp_state_kwargs=compute_cmp_state_kwargs)
        # `roll_out` is a namedtuple containing the loss, last CMPState, and the primal
        # and dual Lagrangian stores, useful for inspection and logging
```

## Contributions

We appreciate all contributions. Please let us know if you encounter a bug by filing an issue.

If you plan to contribute new features, utility functions, or extensions, please first open an issue and discuss the feature with us. To learn more about making a contribution to **Cooper**, please see our Contribution page.

## Papers Using **Cooper**

**Cooper** has enabled several papers published at top machine learning conferences: [Gallego-Posada et al. (2022)](https://arxiv.org/abs/2208.04425); [Lachapelle and Lacoste-Julien (2022)](https://arxiv.org/abs/2207.07732); [Ramirez and Gallego-Posada (2022)](https://arxiv.org/abs/2207.04144); [Zhu et al. (2023)](https://arxiv.org/abs/2310.08106); [Hashemizadeh et al. (2024)](https://arxiv.org/abs/2310.20673); [Sohrabi et al. (2024)](https://arxiv.org/abs/2406.04558); [Lachapelle et al. (2024)](https://arxiv.org/abs/2401.04890); [Jang et al. (2024)](https://arxiv.org/abs/2312.10289); [Navarin et al. (2024)](https://ieeexplore.ieee.org/document/10650578); [Chung et al. (2024)](https://arxiv.org/abs/2404.01216).

## License

**Cooper** is distributed under an MIT license, as found in the
LICENSE file.
