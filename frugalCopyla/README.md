# FrugalCopyla

Clear and efficient sampling from customisable frugal causal models.

## Setup
Clone the repo and run
```
python setup.py install
pip install -r requirements
```
to install the package and its dependencies.


## Usage

The core of the package is the `frugalCopyla.Copula_Model` class, which takes in a dictionary specifying a frugal causal model and allows you to draw data from that distribution. The input provided to the class (i.e. the specification of the model) is flexible and straightforward, and does not require you to program a custom model/sampling routine in the `numpyro`/`jax` backend.

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
* `link` allows the user to provide a link function for each of the linear formulas. For example, the command 
```'X': {'dist': dist.Exponential, 'formula': {'rate': 'X ~ 1 + Z + A'}, 'params': {'rate': {'x_0': 0., 'x_1': 2., 'x_2': 1}}, 'link': {'rate': jnp.exp}},``` will wrap the linear predictor in an exponential function such that the probabilistic model is $$X \sim \text{Exponential}(\lambda=\exp(2Z + A)).$$ **Note that the link function must have a `jax` base.** If no link function is require, leave it as `None`.
* `copula`: To specify a copula, first choose a `'class'` of copula from [frugalCopyla/copula_functions.py](../frugalCopyla/frugalCopyla/copula_functions.py). The copula functions will take in keyword arguments to calculate the log-likelihood of the copula factor. 
    * Under `vars`, provide a mapping of the variables linked by the copula and the function arguments using a dictionary. For example, the `bivariate_gaussian_copula_lpdf(u, v, rho)` factor takes two variables, and one `rho` parameter. If we wish to simulate a copula between `Z` and `Y`, provide `vars` the dictionary `..., 'vars': {'u': 'Z', 'v': 'Y'}`.
    * Under `'formula'`, specify the form of the linear predictor for the parameters passed to the copula. The coefficients for the linear predictor are specified under `'params'`.
    * Similarly to the other inputs, a link function can be chosen to wrap the linear predictor specified in `'formula'` and `'params'`.

For example, consider the following input:
```
input_dict = {
    'Z': {'dist': dist.Normal, 'formula': {'loc': 'Z ~ 1', 'scale': 'Z ~ 1'}, 'params': {'loc': {'z_0': 0.}, 'scale': {'z_0': 1}}, 'link': None},
    'X': {'dist': dist.Exponential, 'formula': {'rate': 'X ~ Z'}, 'params': {'rate': {'x_0': 1., 'x_1': 1.}}, 'link': {'rate': jnp.exp}},
    'Y': {'dist': dist.Normal, 'formula': {'loc': 'Y ~ X', 'scale': 'Y ~ 1'}, 'params': {'loc': {'y_0': -0.5, 'y_1': 1.}, 'scale': {'phi': 1.}}, 'link': None},
    'copula': {'class': 'bivariate_gaussian_copula', 'vars': {'u': 'Z', 'v': 'Y'}, 'formula': {'rho': 'c ~ Z'}, 'params': {'rho': {'a': 1., 'b': 0.}}, 'link': {'rho': jax.nn.sigmoid}}
}
```
which allows one to simulate from the following causal model: $$Z \sim \mathcal{N}(0, 1) \\ X \sim \text{Exponential}(\exp(Z + 1) \\ Y | \text{do}(X) \sim \mathcal{N}(X - 0.5, 1)$$ with a bivariate Gaussian copula between $Z$ and $Y$ parameterised by a fixed covariance term $\rho_{ZY} = logit(1)$
 
