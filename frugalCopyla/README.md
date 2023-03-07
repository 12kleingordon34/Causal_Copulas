# FrugalCopyla

Clear and efficient sampling from customisable frugal causal models.

## Setup
Clone the repo and run
```
pip install -e .
pip install -r requirements.txt
```
to install the package and its dependencies.


## Usage

The core of the package is the `frugalCopyla.CopulaModel` class, which takes in a dictionary specifying a frugal causal model and allows you to draw data from that distribution. The input provided to the class (i.e. the specification of the model) is flexible and straightforward, and does not require you to program a custom model/sampling routine in the `numpyro`/`jax` backend.

See example of use [here]('./examples/basic_demo.ipynb').

### Input

The input should be a dictionary whose keys label the variables in your model. For each of these, specify in a sub-dictionary:

* `dist`: The distribution the variable is drawn from. These must be selected from `numpyro.distributions`
* `formula`: For each parameter in the chosen distribution, specify its linear model **only using variables defined earlier in the dictionary**. The names of the correct parameters can be found by either [searching the `numpyro` documentation](https://num.pyro.ai/en/stable/distributions.html) or looking at the `arg_constraints` of the distribution by running (using the Normal as an example) : 
```
> numpyro.distributions.Normal.arg_constraints 

{'loc': Real(), 'scale': GreaterThan(lower_bound=0.0)}
```
* `params` (name will most likely change): Specifies the linear coefficients used to generate the primary variable through the linear model. A set of coefficients must be provided for each parameter. Note that the labelling of parameters (e.g. `'formula': {'rate': 'X ~ 1 + Z + A'}, 'params': {'rate': {'x_0': 0., 'x_1': 2., 'x_2': 1}}`) does not affect the linear model. Only the order of the specification matters. For example, `x_0` will be the coefficient of the first variable in the formula (always the intercept) and `x_2` will always be the last.
* `link` allows the user to provide a link function for each of the linear formulas. For example, the command ```'X': {'dist': dist.Exponential, 'formula': {'rate': 'X ~ 1 + Z + A'}, 'params': {'rate': {'x_0': 0., 'x_1': 2., 'x_2': 1}}, 'link': {'rate': jnp.exp}},``` 
will wrap the linear predictor in an exponential function such that the probabilistic model is $$X \sim \text{Exponential}(\lambda=\exp(2Z + A)).$$ **Note that the link function must have a `jax` base.** If no link function is require, leave it as `None`.
    * Additionally, note that the order of the floats within `'params'` will automatically be the the order they are multiplied to the ordered variables in `'rate'`. **The intercept term will always be the first term in the parsed formula.**
* `copula`: To specify a copula, first choose a `'class'` of copula from [frugalCopyla/copula_functions.py](../frugalCopyla/copula_functions.py). The copula functions will take in keyword arguments to calculate the log-likelihood of the copula factor. 
    * Under `vars`, provide a mapping of the variables linked by the copula and the function arguments using a dictionary. For example, the `multivar_gaussian_copula_lpdf(vars, rhos)` factor takes two variables: a dictionary of random variables (`vars`) and a dictionary of the copula correlation matrix elements (`rhos`). If we wish to simulate a copula between `Z` and `Y`, provide `vars` the dictionary `..., 'vars': {'u': 'Z', 'v': 'Y'}`.
    * Under `'formula'`, specify the form of the linear predictor for the parameters passed to the copula. The coefficients for the linear predictor are specified under `'params'`.
    * Similarly to the other inputs, a link function can be chosen to wrap the linear predictor specified in `'formula'` and `'params'`.

#### Viewing Parsed Model

To check whether the model has been parsed correctly, you can check the `parsed_model` property of a `CopulaModel` class:
```
...
>>> cop_mod = CopulaModel(input_dict)
>>> cop_mod.parsed_model

{'Z': {'dist': numpyro.distributions.continuous.Normal,
  'formula': {'loc': 'A ~ 1', 'scale': 'A ~ 1'},
  'coeffs': {'loc': [0.0], 'scale': [1.0]},
  'link': {},
  'linear_predictor': {'loc': '0.0', 'scale': '1.0'}},
 'X': {'dist': numpyro.distributions.continuous.Normal,
  'formula': {'loc': "X ~ 1 + record_dict['Z']", 'scale': 'X ~ 1'},
  'coeffs': {'loc': [0.0, 0.5], 'scale': [1.0]},
  'link': {},
  'linear_predictor': {'loc': "0.0 + 0.5 * record_dict['Z']", 'scale': '1.0'}},
 'Y': {'dist': numpyro.distributions.continuous.Normal,
  'formula': {'loc': "Y ~ 1 + record_dict['X']", 'scale': 'Y ~ 1'},
  'coeffs': {'loc': [0.0, 0.5], 'scale': [0.5]},
  'link': {'loc': None},
  'linear_predictor': {'loc': "0.0 + 0.5 * record_dict['X']", 'scale': '0.5'}}}
```
Note that the `linear_predictor` field shows the form of the linear predictor for a distribution's parameter (before being passed to the inverse link function).

### Code Example

For example, consider the following input:
```
import numpyro.distributions as dist
import jax
import jax.numpy as jnp

import frugalCopyla

input_dict = {
    'Z': {'dist': dist.Normal, 'formula': {'loc': 'Z ~ 1', 'scale': 'Z ~ 1'}, 'coeffs': {'loc': [0.], 'scale': [1.]}, 'link': None},
    'X': {'dist': dist.Exponential, 'formula': {'rate': 'X ~ Z'}, 'coeffs': {'rate': [1., 1.]}, 'link': {'rate': jnp.exp}},
    'Y': {'dist': dist.Normal, 'formula': {'loc': 'Y ~ X', 'scale': 'Y ~ 1'}, 'coeffs': {'loc': [-0.5, 1.], 'scale': [1.]}, 'link': None},
    'copula': {
        'class': frugalCopyla.copula_functions.multivar_gaussian_copula_lpdf,
        'vars': {'u': 'Z', 'v': 'Y'},
        'formula': {'rho': 'c ~ Z'},
        'coeffs': {'rho': [1., 0.]},
        'link': {'rho': jax.nn.sigmoid}
    }
}

model = frugalCopyla.model.CopulaModel(input_dict)
data = model.simulate_data(num_warmup=1000, num_samples=1000, joint_status='continuous')
```
which allows one to simulate from the following causal model: $$Z \sim \mathcal{N}(0, 1) \\ X \sim \text{Exponential}(\exp(Z + 1) \\ Y | \text{do}(X) \sim \mathcal{N}(X - 0.5, 1)$$ with a bivariate Gaussian copula between $Z$ and $Y$ parameterised by a fixed covariance term $\rho_{ZY} = logit(1)$
 
