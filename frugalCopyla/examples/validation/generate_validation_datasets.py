import json

from jax import	random
import jax
import jax.numpy as	jnp
import jax.scipy.stats as jax_stats

import numpyro
import numpyro.distributions as	dist
import numpy as	np
import pandas as pd

from frugalCopyla import copula_functions
from frugalCopyla.model	import CopulaModel

numpyro.set_host_device_count(4)

def	generate_cop_data(input_template: dict,	sample_size: int, joint_status:	str, seed :	int	= None)	-> dict:
	mod	= CopulaModel(input_template)
	if seed:
		sampled_model =	mod.simulate_data(num_warmup=5000, num_samples=sample_size,	joint_status=joint_status, seed=seed)
	else:
		sampled_model =	mod.simulate_data(num_warmup=5000, num_samples=sample_size,	joint_status=joint_status)
	print(sampled_model['model'].print_summary())
	return pd.DataFrame(sampled_model['data'])


def	model_b(beta, rho):
	rho_zz_beta = beta/jnp.sqrt(1+jnp.square(beta))
	rho_z1z2 = 0.
	rho_z1y	= 0.5
	rho_z2y	= 0.25
	#print(jnp.sqrt(2*rho_zz_beta*rho_z1y*rho_z2y))
	Z1 = numpyro.sample('Z1', dist.Normal(0., 1.))
	Z2 = numpyro.sample('Z2', dist.Normal(beta * Z1, 1.))
	#Z2 = numpyro.sample('Z2', dist.Normal(0., jnp.sqrt(1. + jnp.square(beta))))
	Y = numpyro.sample('Y', dist.Normal(0., 1))
	q_Z1 = numpyro.deterministic('qZ1', dist.Normal(0., 1.).cdf(Z1))
	#q_Z2 = numpyro.deterministic('qZ2', dist.Normal(0., jnp.sqrt(1. + jnp.square(beta))).cdf(Z2))
	q_Z2 = numpyro.deterministic('qZ2', dist.Normal(beta * Z1,	1.).cdf(Z2))
	#q_Y = numpyro.deterministic('qY', dist.Normal(X - 0.5, 1.).cdf(Y))
	q_Y	= numpyro.deterministic('qY', dist.Normal(0., 1).cdf(Y))
	#q_A	= numpyro.deterministic('qA', dist.Normal(0., jnp.sqrt(1 + 2*rho_zz_beta*rho_z1y*rho_z2y)).cdf(Y))
	#q_A	= numpyro.deterministic('qA', dist.Normal(0., jnp.sqrt(0.75)).cdf(Y))
	std_normal_Z1 =	numpyro.deterministic('std_normal_z1', dist.Normal(0., 1.).icdf(q_Z1))
	std_normal_Z2 =	numpyro.deterministic('std_normal_z2', dist.Normal(0., 1.).icdf(q_Z2))
	std_normal_Y = numpyro.deterministic('std_normal_y', dist.Normal(0., 1.).icdf(q_Y))
	#q_Z1 = numpyro.sample('q_Z1', dist.Uniform(low=0, high=1))
	#q_Z2 = numpyro.sample('q_Z2', dist.Uniform(low=0, high=1))
	#q_Y = numpyro.sample('q_Y', dist.Uniform(low=0, high=1))
	#std_normal_Z1 =	numpyro.deterministic('std_normal_z1', dist.Normal(0., 1.).icdf(q_Z1))
	#std_normal_Z2 =	numpyro.deterministic('std_normal_z2', dist.Normal(0., 1.).icdf(q_Z2))
	#std_normal_Y = numpyro.deterministic('std_normal_y', dist.Normal(0., 1.).icdf(q_Y))
	#Z1 = numpyro.deterministic('Z1', dist.Normal(0., 1.).icdf(q_Z1))
	#Z2 = numpyro.deterministic('Z2', dist.Normal(beta * Z1, 1).icdf(q_Z2))
	#Y = numpyro.deterministic('Y', dist.Normal(0., 1.).icdf(q_Y))


	numpyro.factor(
		'cop_log_prob_full',
		copula_functions.multivar_gaussian_copula_lpdf(
			{'z1': std_normal_Z1, 'z2': std_normal_Z2, 'z3': std_normal_Y},
			{'rhoz1z2': rho_z1z2, 'rho_z1y': rho_z1y, 'rho_z2y': rho_z2y}
		)
	)


def	sample_model_b(model, beta,	rho, num_warmup, num_samples, seed=None):
	kernel = numpyro.infer.NUTS(model)
	mcmc_model = numpyro.infer.MCMC(
		kernel,
		num_warmup=num_warmup,
		num_samples=num_samples, 
		progress_bar=False
	)
	if not seed:
		seed = np.random.randint(0, 1e7)
	mcmc_model.run(
		beta=beta,
		rho=rho,
		rng_key=jax.random.PRNGKey(seed)
	)
	print(mcmc_model.print_summary())
	return pd.DataFrame(mcmc_model.get_samples())#[['Z1', 'Z2', 'X',	'Y']]


def	main():
	rhos = [0.2, 0.4]# 0.6, 0.8]
	for	rho	in rhos:
		multivar_gaussian_cop =	{
			'Z': {'dist': dist.Normal, 'formula': {'loc': 'Z ~ 1', 'scale':	'Z ~ 1'}, 'coeffs':	{'loc':	[0.], 'scale': [1.]}, 'link': {}},
			'X': {'dist': dist.Normal, 'formula': {'loc': 'X ~ Z', 'scale':	'X ~ 1'}, 'coeffs':	{'loc':	[0., rho], 'scale':	[1.	- jnp.square(rho)]}, 'link': {'scale': jnp.sqrt}},
			'Y': {'dist': dist.Normal, 'formula': {'loc': 'Y ~ X', 'scale':	'Y ~ 1'}, 'coeffs':	{'loc':	[-0.5, rho], 'scale': [1.]}, 'link': {}},
			'copula': {
				'class': copula_functions.multivar_gaussian_copula_lpdf,
				'vars':	['Z', 'Y'],
				'formula': {'rho_ZY': 'cop ~ 1'},
				'coeffs': {'rho_ZY': [0.5]},
				'link':	{'rho_ZY': None}
			}
		}
		beta = 1
		multiconfounder_gaussian_cop_model_a = {
			'Z1': {'dist': dist.Normal,	'formula': {'loc': 'Z1 ~ 1', 'scale': 'Z1 ~	1'}, 'coeffs': {'loc': [0.], 'scale': [1.]}, 'link': {}},
			'Z2': {'dist': dist.Normal,	'formula': {'loc': 'Z2 ~ 1', 'scale': 'Z1 ~	1'}, 'coeffs': {'loc': [0.], 'scale': [1. +	jnp.square(beta)]},	'link':	{'scale': jnp.sqrt}},
			'X': {'dist': dist.Normal, 'formula': {'loc': 'X ~ Z1 +	Z2', 'scale': 'X ~ 1'},	'coeffs': {'loc': [0., rho,	0.], 'scale': [1.]}, 'link': {'scale': jnp.sqrt}},
			'Y': {'dist': dist.Normal, 'formula': {'loc': 'Y ~ X', 'scale':	'Y ~ 1'}, 'coeffs':	{'loc':	[-0.5, 1], 'scale':	[1.]}, 'link': {}},
			'copula': {
				'class': copula_functions.multivar_gaussian_copula_lpdf,
				'vars':	['Z1', 'Z2', 'Y'],
				'formula': {'rho_Z1_Z2': 'cop ~	1',	'rho_Z1_Y':	'cop ~ 1','rho_Z2_Y': 'cop ~ 1'},
				#'coeffs': {'rho_Z1_Z2': [0.], 'rho_Z1_Y': [0.2],'rho_Z2_Y': [0.95]},
				'coeffs': {'rho_Z1_Z2':	[beta /	jnp.sqrt(1 + jnp.square(beta))], 'rho_Z1_Y': [0.2],'rho_Z2_Y': [0.]},
				'link':	{'rho_Z1_Z2': None,	'rho_Z1_Y':	None,'rho_Z2_Y': None},
			}
		}
		multiconfounder_gaussian_cop_model_b = {
			'Z1': {'dist': dist.Normal,	'formula': {'loc': 'Z1 ~ 1', 'scale': 'Z1 ~	1'}, 'coeffs': {'loc': [0.], 'scale': [1.]}, 'link': {'scale': jnp.sqrt}},
			'Z2': {'dist': dist.Normal,	'formula': {'loc': 'Z2 ~ Z1', 'scale': 'Z2 ~ 1'}, 'coeffs':	{'loc':	[0., beta],	'scale': [1.]},	'link':	{'scale': jnp.sqrt}},
			'X': {'dist': dist.Normal, 'formula': {'loc': 'X ~ Z1 +	Z2', 'scale': 'X ~ 1'},	'coeffs': {'loc': [0., rho,	0.], 'scale': [1.]}, 'link': {'scale': jnp.sqrt}},
			'Y': {'dist': dist.Normal, 'formula': {'loc': 'Y ~ X', 'scale':	'Y ~ 1'}, 'coeffs':	{'loc':	[-0.5, 1], 'scale':	[1.]}, 'link': {}},
			'copula': {
				'class': copula_functions.multivar_gaussian_copula_lpdf,
				'vars':	['Z1', 'Z2', 'Y'],
				'formula': {'rho_Z1_Z2': 'cop ~	1',	'rho_Z1_Y':	'cop ~ 1','rho_Z2_Y': 'cop ~ 1'},
				'coeffs': {'rho_Z1_Z2':	[0.0], 'rho_Z1_Y': [0.5],'rho_Z2_Y': [0.5]},
				'link':	{'rho_Z1_Z2': None,	'rho_Z1_Y':	None,'rho_Z2_Y': None},
			}
		}
		multiconfounder_studentt_cop_model_b = {
			'Z1': {'dist': dist.Normal,	'formula': {'loc': 'Z1 ~ 1', 'scale': 'Z1 ~	1'}, 'coeffs': {'loc': [0.], 'scale': [1.]}, 'link': {'scale': jnp.sqrt}},
			'Z2': {'dist': dist.Normal,	'formula': {'loc': 'Z2 ~ Z1', 'scale': 'Z2 ~ 1'}, 'coeffs':	{'loc':	[0., beta],	'scale': [1.]},	'link':	{'scale': jnp.sqrt}},
			#'Z2': {'dist': dist.Normal,	'formula': {'loc': 'Z2 ~ 1', 'scale': 'Z2 ~ 1'}, 'coeffs':	{'loc':	[0.],	'scale': [1.]},	'link':	{'scale': jnp.sqrt}},
			#'X': {'dist': dist.Normal, 'formula': {'loc': 'X ~ Z1 +	Z2', 'scale': 'X ~ 1'},	'coeffs': {'loc': [0., rho,	0.], 'scale': [1.]}, 'link': {'scale': jnp.sqrt}},
			#'Y': {'dist': dist.Normal, 'formula': {'loc': 'Y ~ X', 'scale':	'Y ~ 1'}, 'coeffs':	{'loc':	[-0.5, 1], 'scale':	[1.]}, 'link': {}},
			'Y': {'dist': dist.Normal, 'formula': {'loc': 'Y ~ 1', 'scale':	'Y ~ 1'}, 'coeffs':	{'loc':	[0.0], 'scale':	[1.]}, 'link': {}},
			'copula': {
				'class': copula_functions.multivar_studentt_copula_lpdf,
				'vars':	['Z1', 'Z2', 'Y'],
				'formula': {'rho_Z1_Z2': 'cop ~	1',	'rho_Z1_Y':	'cop ~ 1','rho_Z2_Y': 'cop ~ 1'},
				'coeffs': {'rho_Z1_Z2':	[0.0], 'rho_Z1_Y': [0.5],'rho_Z2_Y': [0.2]},
				'link':	{'rho_Z1_Z2': None,	'rho_Z1_Y':	None,'rho_Z2_Y': None},
				'misc': {'df': 2.}
			}
		}

		bernoulli_gaussian_cop = {
			'Z': {'dist': dist.BernoulliProbs, 'formula': {'probs':	'Z ~ 1'}, 'coeffs':	{'probs': [0.9]}, 'link': {}},
			'X': {'dist': dist.BernoulliProbs, 'formula': {'probs':	'X ~ Z'}, 'coeffs':	{'probs': [0., rho]}, 'link': {'probs':	jax.nn.sigmoid}},
			'Y': {'dist': dist.Normal, 'formula': {'loc': 'Y ~ X', 'scale':	'Y ~ 1'}, 'coeffs':	{'loc':	[-1., rho],	'scale': [1.]},	'link':	{}},
		}

		didelez_cop	= {
			'A': {'dist': dist.BernoulliProbs, 'formula': {'probs':	'A ~ 1'}, 'coeffs':	{'probs': [0.]}, 'link': {'probs': jax.nn.sigmoid}},
			'L': {'dist': dist.Exponential,	'formula': {'rate':	'L ~ A'}, 'coeffs':	{'rate': [-0.3,	0.2]}, 'link': {'rate':	jnp.exp}},
			'B': {'dist': dist.BernoulliProbs, 'formula': {'probs':	'B ~ A + L + A:L'},	'coeffs': {'probs':	[-0.3, 0.4,	0.3, 0.]}, 'link': {'probs': jax.nn.sigmoid}},
			'Y': {'dist': dist.Normal, 'formula': {'loc': 'Y ~ A + B + A:B', 'scale': 'Y ~ 1'},	'coeffs': {'loc': [-0.5, 0.2, 0.3, 0.],	'scale': [1.]},	'link':	{}},
			'copula': {
				'class': copula_functions.multivar_gaussian_copula_lpdf,
				'vars':	['L', 'Y'],
				'formula': {'rho_LY': 'cop ~ A'},
				'coeffs': {'rho_LY': [1., 0.5]},
				'link':	{'rho_LY': jax.nn.sigmoid}
			}
		}

		#gaussian_data = generate_cop_data(multivar_gaussian_cop, num_samples, 'continuous', 0)

		#gaussian_data.to_csv(f'./validation_datasets/multivar_gaussian/gaussian_cop_data_rho_{rho}.csv', index=False)
		num_samples	= 100_000
		#multiconfounder_data_model_a = generate_cop_data(multiconfounder_gaussian_cop_model_a, num_samples,	'continuous')
		#multiconfounder_data_model_a.to_csv(f'./validation_datasets/multivar_gaussian/multiconfounder_gaussian_data_rho_{rho}_model_a.csv',	index=False)
		#multiconfounder_data_model_b = generate_cop_data(multiconfounder_gaussian_cop_model_b, num_samples,	'continuous')
		#multiconfounder_data_model_b.to_csv(f'./validation_datasets/multivar_gaussian/multiconfounder_gaussian_data_rho_{rho}_model_b.csv',	index=False)
		studentt_multiconfounder_data_model_b = generate_cop_data(multiconfounder_studentt_cop_model_b, num_samples,	'continuous')
		studentt_multiconfounder_data_model_b.to_csv(f'./validation_datasets/multivar_studentt/multiconfounder_studentt_data_rho_{rho}_model_b.csv',	index=False)
		#multiconfounder_data_model_b = sample_model_b(model_b, beta=beta, rho=rho, num_warmup=5000, num_samples=num_samples)
		#multiconfounder_data_model_b.to_csv(f'./validation_datasets/multivar_gaussian/multiconfounder_gaussian_data_rho_{rho}_model_b.csv',	index=False)
		#multiconfounder_data_model_broken =	sample_model_b(model_b, beta=1.5, rho=rho, num_warmup=5000, num_samples=num_samples)
		#multiconfounder_data_model_broken.to_csv(f'./validation_datasets/multivar_gaussian/multiconfounder_gaussian_data_rho_{rho}_model_broken.csv', index=False)
	# Generate Didelez Data
	#didelez_data =	generate_cop_data(didelez_cop, num_samples,	'mixed', 0)
	#didelez_data.to_csv(f'./validation_datasets/didelez_simulation.csv', index=False)


if __name__	== '__main__':
	main()