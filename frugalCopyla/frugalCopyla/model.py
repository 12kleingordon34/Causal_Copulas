import copy
import inspect
import random
import re
from typing import Literal

import jax
import jax.numpy as jnp
import numpyro
from numpyro.distributions import *
import patsy

from frugalCopyla.copula_functions import _reshape_matrix

##### PERHAPS CHECK WHETHER A LINK FUNCTION IS NECESSARY FOR EACH PARAMETER
###### PERHAPS CHECK WHETHER A LINK FUNCTION IS NECESSARY FOR EACH PARAMETER
###### PERHAPS CHECK WHETHER A LINK FUNCTION IS NECESSARY FOR EACH PARAMETER
###### PERHAPS CHECK WHETHER A LINK FUNCTION IS NECESSARY FOR EACH PARAMETER

###### REMEMBER THAT THE INTERCEPT IS ALWAYS PARSED FIRST WITH PATSY
###### REMEMBER THAT THE INTERCEPT IS ALWAYS PARSED FIRST WITH PATSY
###### REMEMBER THAT THE INTERCEPT IS ALWAYS PARSED FIRST WITH PATSY
###### REMEMBER THAT THE INTERCEPT IS ALWAYS PARSED FIRST WITH PATSY
_DISTRIBUTION_TYPES = ['discrete', 'continuous', 'mixed']


class CopulaModel:
	def __init__(self, input_dict: dict) -> None:
		self.parsed_model = self._process_input_model(input_dict)
		assert self._is_model_ordered_and_acyclic(self.parsed_model)


	def _process_input_model(self, input_dict: dict) -> dict:
		"""
		Key steps:
		* Check distributions are valid
		* Check coeffs and formula keys are consistent for distribution parameter type
		* Check formula type
		* Check link function is valid
		"""
		parsed_model = dict()

		model_dict = input_dict
		copula_settings = model_dict.pop('copula', None)
		for var, spec in model_dict.items():
			parsed_model[var] = {'dist': model_dict[var]['dist'], 'formula': {}, 'coeffs': [], 'link': {}}

			# Currently use numpyro.distributions in input, but can change
			assert self._is_dist_from_numpyro(model_dict[var]['dist'])
			distribution_params = parsed_model[var]['dist'].arg_constraints.keys()
			
			if model_dict[var].get('link', None):
				for param in model_dict[var]['link'].keys():
					assert self._is_link_from_jax(model_dict[var]['link'][param])
					parsed_model[var]['link'][param] = model_dict[var]['link'][param]

			assert model_dict[var]['formula'].keys() == distribution_params
			assert len(model_dict[var]['coeffs']) == len(distribution_params)

			# Check correct number of parameters have been provided
			### TO DO: Fix confusion between `param` and `'coeffs'` entries
			parsed_model[var]['linear_predictor'] = dict()
			for param in distribution_params:
				parsed_model[var]['formula'][param] = self._regex_variable_adjustment(model_dict[var]['formula'][param])
				assert len(model_dict[var]['coeffs'][param]) == len(
					patsy.ModelDesc.from_formula(
						parsed_model[var]['formula'][param]
					).rhs_termlist
				)
				parsed_model[var]['coeffs'] = model_dict[var]['coeffs'].copy()
				parsed_model[var]['linear_predictor'][param] = self._align_transformation(
					parsed_model[var]['formula'][param],
					parsed_model[var]['coeffs'][param]
				)

		if not copula_settings:
			print('INFO: No copula specified.')
			return parsed_model

		parsed_model['copula'] = {
			'class': None,
			'vars': [],
			'corr_linear_predictor': {},
			'link': {},
			'misc': {}
		}
		parsed_model['copula']['class'] = copula_settings['class']
		parsed_model['copula']['vars'] = copula_settings['vars'].copy()
		for param in copula_settings['formula'].keys():
			copula_settings['formula'][param] = self._regex_variable_adjustment(copula_settings['formula'][param])
			parsed_model['copula']['corr_linear_predictor'][param] = self._align_transformation(
				copula_settings['formula'][param],
				copula_settings['coeffs'][param]
			)
			assert self._is_link_from_jax(copula_settings['link'][param])
			parsed_model['copula']['link'][param] = copula_settings['link'][param]
		misc_params = copula_settings.get('misc', None)
		if misc_params:
			parsed_model['copula']['misc'] = misc_params
		
		return parsed_model
		
	def simulate_data(
		self,
		num_warmup: int,
		num_samples: int,
		joint_status: _DISTRIBUTION_TYPES,
		num_chains: int = 4,
		seed: int = random.randint(0, 10e6)
	) -> dict:
		# assert joint_status.lower() in ['discrete', 'continuous', 'mixed']
		if joint_status.lower() == 'discrete':
			pass
		elif joint_status == 'continuous':
			kernel = numpyro.infer.NUTS(self._simulated_data_model)
		elif joint_status == 'mixed':
			kernel = numpyro.infer.DiscreteHMCGibbs(
				numpyro.infer.NUTS(self._simulated_data_model),
				modified=True
			)
		numpyro.set_host_device_count(num_chains)		
		mcmc_model = numpyro.infer.MCMC(
			kernel,
			num_chains=num_chains,
			num_warmup=num_warmup,
			num_samples=num_samples,
			chain_method='parallel',
			progress_bar=False
		)
		
		mcmc_model.run(
			rng_key=jax.random.PRNGKey(seed)
		)
		## TO DO: Figure out how to return the data in a useful format.
		return {
			'model': mcmc_model,
			'data': mcmc_model.get_samples()
		}

	def inference(self) -> dict:
		"""
		Performs inference on the parameters given a specified model
		and input data.
		
		TBC.
		"""
		pass

	def _simulated_data_model(self) -> None:
		prob_model = copy.deepcopy(self.parsed_model)
		copula_params = prob_model.pop('copula', None)
		record_dict = {}
		for test_idx, test_row in prob_model.items():
			lin_models_str = test_row['linear_predictor']

			lin_models_evaluated = {}
			for k, v in lin_models_str.items():
				if test_row['link'].get(k, None):
					lin_models_evaluated[k] = test_row['link'][k](eval(v))
				else:
					lin_models_evaluated[k] = eval(v)
			#### ITS NOT DOING ANYTHING WITH THE SCALE PARAMETER --- NEED TO FIX
			if test_row['link']:
				record_dict[test_idx] = numpyro.sample(
					test_idx,
					test_row['dist'](**lin_models_evaluated)
				)
			else:
				record_dict[test_idx] = numpyro.sample(
					test_idx,
					test_row['dist'](**lin_models_evaluated)
				)
		
		 	# Generate quantiles
			if copula_params and test_idx in copula_params['vars']:
				record_dict[f"q_{test_idx}"] = numpyro.deterministic(
					f"q_{test_idx}",
					prob_model[test_idx]['dist'](**lin_models_evaluated).cdf(record_dict[test_idx])
				)

		if copula_params:
			for idx, formula in copula_params['corr_linear_predictor'].items():
				if copula_params['link'][idx]:
					record_dict[idx] = numpyro.deterministic(
						idx,
						###### PERHAPS CHECK WHETHER A LINK FUNCTION IS NECESSARY FOR EACH PARAMETER
						copula_params['link'][idx](eval(copula_params['corr_linear_predictor'][idx]))
					)
				else:
					record_dict[idx] = numpyro.deterministic(
						idx,
						###### PERHAPS CHECK WHETHER A LINK FUNCTION IS NECESSARY FOR EACH PARAMETER
						eval(copula_params['corr_linear_predictor'][idx])
					)

			

			#### HOW TO MAP THE VARS TO THE DICT
			copula_var_dict = {}
			copula_rho_dict = {}
			for k in copula_params['vars']:
				copula_var_dict[k] = record_dict[f"q_{k}"]
			for k, v in copula_params['corr_linear_predictor'].items():
				copula_rho_dict[k] = record_dict[k]

			record_dict['cop_log_prob'] = numpyro.factor(
				'cop_log_prob',
				copula_params['class'](copula_var_dict, copula_rho_dict, **copula_params['misc'])
			)

	def _is_dist_from_numpyro(self, prob_distribution: numpyro.distributions) -> bool:
		"""
		Check that the user provided probability distribution
		is sourced from numpyro
		"""
		dist_module_source = inspect.getmodule(prob_distribution)
		valid_source_modules = [numpyro.distributions.discrete, numpyro.distributions.continuous]

		if dist_module_source in valid_source_modules:
			return True
		else:
			return False

	def _is_link_from_jax(self, link_function) -> bool:
		"""
		Map the link function in the input model dict
		to its corresponding link function in Numpyro.

		The package requires users to pick their link functions
		from jax.numpy. The full list of options can be found from

		https://jax.readthedocs.io/en/latest/jax.numpy.html
		"""
		if not link_function:
			return True
		link_func_module_source_name = inspect.getmodule(link_function).__name__
		valid_source_modules = [
			'jax._src.nn.functions',
			'jax._src.numpy.ufuncs',
			'jax',
			'jax.nn',
			'jax.numpy',
			'jax.scipy',
			'jax._src.scipy.special',
			'jax._src.scipy'
		]
		if link_func_module_source_name in valid_source_modules:
			return True
		else:
			return False

	def _is_model_ordered_and_acyclic(self, parsed_model: dict) -> bool:
		"""
		Check whether the input model is acyclic, and whether
		the order variables are entered are """
		print('WARNING: CHECK PASSES AUTOMATICALLY. FULL SOLUTION ADDED TBC')
		return True

	def _align_transformation(self, formula: str, coeffs: list[float]) -> str:
		"""
		Convert input formula and parameters into numpyro-readable format
		for data simulation
		"""
		formula_factors = patsy.ModelDesc.from_formula(formula)
		rhs_terms = formula_factors.rhs_termlist

		assert len(rhs_terms) == len(coeffs)

		if len(coeffs) == 1:
			return str(coeffs[0])
		if len(coeffs) >= 2:
			list_of_terms = ([f"{coeffs[0]}"] +
				[f'{i} * {j.name()}' for i, j in zip(coeffs[1:], rhs_terms[1:])])
			return  " + ".join(list_of_terms).replace(':', '*')

	def _regex_variable_adjustment(self, formula: str) -> str:
		"""
		Ensure that variables in input formula can be parsed by numpyro.

		TODO: NEED TO ENSURE THAT THE REGEX IS RUN ON THE OUTPUT OF PATSY, AND UPDATE THINGS IN A FOR LOOP (I.E. IT DOESN'T WORK IF THERE ARE MULTIPLE ADDITIVE COMPONENTS IN THE REGEX STRING)
		"""
		parsed_terms = patsy.ModelDesc.from_formula(formula)
		lhs_terms = parsed_terms.lhs_termlist
		lhs_term_str = lhs_terms[0].name()

		rhs_terms = parsed_terms.rhs_termlist
		enhanced_rhs_term_list = []
		for term in rhs_terms:
			if len(term.factors) <= 1:
				enhanced_rhs_term_list.append(
					self.__regex_variable_adjustment(term.name())
				)
			else:
				product_term_list = []
				for factor in term.factors:
					product_term_list.append(
						self.__regex_variable_adjustment(factor.name())
					)
				enhanced_rhs_term_list.append(
					' * '.join(product_term_list)
				)
		enhanced_rhs_term_str = ' + '.join(enhanced_rhs_term_list)
		
		enhanced_formula = ' ~ '.join([
			lhs_term_str, enhanced_rhs_term_str
		]).replace(':', '*')
		return enhanced_formula

	def __regex_variable_adjustment(self, term_name: str) -> str:
		"""
		"""
		# For each linear argument, extract core arguments from within 
		# functions (if functions exist)
		functional_regex = r"[\.,\w,\(]+\(([a-zA-Z, , :_,-,*,//]+)[\)]+$"

		if term_name == 'Intercept':
			return '1'

		regexed_term = re.findall(functional_regex, term_name)
		if not regexed_term:
			return f"record_dict['{term_name}']"

		key_formula = regexed_term[0]
		orig_vars = re.findall(r"[\w,\_]", key_formula)
		new_vars = []
		for var in orig_vars:
			new_vars.append(f"record_dict['{var.strip()}']")
		
		new_formula = key_formula
		for orig_var, new_var in zip(orig_vars, new_vars):
			new_formula = new_formula.replace(orig_var, new_var)
		new_formula = term_name.replace(key_formula, new_formula)
		return new_formula