import json

from jax import random
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pandas as pd

from frugalCopyla import copula_functions
from frugalCopyla.model import Copula_Model


def generate_cop_data(input_template: dict, sample_size: int, joint_status: str, seed: int) -> dict:
	mod = Copula_Model(input_template)
	sim_data = mod.simulate_data(num_warmup=5000, num_samples=sample_size, joint_status=joint_status, seed=seed)
	return pd.DataFrame(sim_data)


def main():
	rhos = [0.2, 0.4, 0.5, 0.7]
	for rho in rhos:
		multivar_gaussian_cop = {
			'Z': {'dist': dist.Normal, 'formula': {'loc': 'Z ~ 1', 'scale': 'Z ~ 1'}, 'coeffs': {'loc': [0.], 'scale': [1.]}, 'link': {}},
			#'X': {'dist': dist.Normal, 'formula': {'loc': 'X ~ Z', 'scale': 'X ~ Z'}, 'coeffs': {'loc': [0., rho], 'scale': [1., -jnp.square(rho)]}, 'link': {'scale': jnp.sqrt}},
			#'Y': {'dist': dist.Normal, 'formula': {'loc': 'Y ~ X', 'scale': 'Y ~ 1'}, 'coeffs': {'loc': [-0.5, rho], 'scale': [1.]}, 'link': {}},
			'X': {'dist': dist.Normal, 'formula': {'loc': 'X ~ Z', 'scale': 'X ~ Z'}, 'coeffs': {'loc': [0., 1], 'scale': [1., 0.]}, 'link': {'scale': jnp.sqrt}},
			'Y': {'dist': dist.Normal, 'formula': {'loc': 'Y ~ X', 'scale': 'Y ~ 1'}, 'coeffs': {'loc': [-0.5, 1.], 'scale': [1.]}, 'link': {}},
			'copula': {
				'class': copula_functions.multivar_gaussian_copula_lpdf,
				'vars': ['Z', 'Y'],
				'formula': {'rho_ZY': 'cop ~ Z'},
				'coeffs': {'rho_ZY': [1., 0.]},
				'link': {'rho_ZY': jax.scipy.special.expit}
			}
		}

		bernoulli_gaussian_cop = {
			'Z': {'dist': dist.BernoulliProbs, 'formula': {'probs': 'Z ~ 1'}, 'coeffs': {'probs': [0.9]}, 'link': {}},
			'X': {'dist': dist.BernoulliProbs, 'formula': {'probs': 'X ~ Z'}, 'coeffs': {'probs': [0., rho]}, 'link': {'probs': jax.nn.sigmoid}},
			'Y': {'dist': dist.Normal, 'formula': {'loc': 'Y ~ X', 'scale': 'Y ~ 1'}, 'coeffs': {'loc': [-1., rho], 'scale': [1.]}, 'link': {}},
		}

		didelez_cop = {
			'A': {'dist': dist.BernoulliProbs, 'formula': {'probs': 'A ~ 1'}, 'coeffs': {'probs': [0.]}, 'link': {'probs': jax.nn.sigmoid}},
			'L': {'dist': dist.Exponential, 'formula': {'rate': 'L ~ A'}, 'coeffs': {'rate': [-0.3, 0.2]}, 'link': {'rate': jnp.exp}},
			'B': {'dist': dist.BernoulliProbs, 'formula': {'probs': 'B ~ A + L + A:L'}, 'coeffs': {'probs': [-0.3, 0.4, 0.3, 0.]}, 'link': {'probs': jax.nn.sigmoid}},
			'Y': {'dist': dist.Normal, 'formula': {'loc': 'Y ~ A + B + A:B', 'scale': 'Y ~ 1'}, 'coeffs': {'loc': [-0.5, 0.2, 0.3, 0.], 'scale': [1.]}, 'link': {}},
			'copula': {
				'class': copula_functions.multivar_gaussian_copula_lpdf,
				'vars': ['L', 'Y'],
				'formula': {'rho_LY': 'cop ~ A'},
				'coeffs': {'rho_LY': [1., 0.5]},
				'link': {'rho_LY': jax.nn.sigmoid}
			}
		}

		gaussian_data = generate_cop_data(multivar_gaussian_cop, 1_000_000, 'continuous', 0)
		#bernoulli_data = generate_cop_data(bernoulli_gaussian_cop, 1_000_000, 'mixed', 0)

		gaussian_data.to_csv(f'./validation_datasets/multivar_gaussian/gaussian_cop_data_rho_{rho}.csv', index=False)
		#bernoulli_data.to_csv(f'./validation_datasets/gaussian_bernoulli/bernoulli_cop_data_rho_{rho}.csv', index=False)
	# Generate Didelez Data
	didelez_data = generate_cop_data(didelez_cop, 1_000_000, 'mixed', 0)
	didelez_data.to_csv(f'./validation_datasets/didelez_simulation.csv', index=False)


if __name__ == '__main__':
	main()