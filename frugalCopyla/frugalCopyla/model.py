import random
import re
from typing import Literal

import jax
import jax.numpy as jnp
import numpyro
import patsy

from frugalCopyla import copula_functions

###### PERHAPS CHECK WHETHER A LINK FUNCTION IS NECESSARY FOR EACH PARAMETER
###### PERHAPS CHECK WHETHER A LINK FUNCTION IS NECESSARY FOR EACH PARAMETER
###### PERHAPS CHECK WHETHER A LINK FUNCTION IS NECESSARY FOR EACH PARAMETER
###### PERHAPS CHECK WHETHER A LINK FUNCTION IS NECESSARY FOR EACH PARAMETER

COPULA_TAGS = {
	'bivariate_gaussian_copula': copula_functions.bivariate_gaussian_copula_lpdf
}
_DISTRIBUTION_TYPES = ['discrete', 'continuous', 'mixed']
# {
# 	'copula': {
# 		'class': 'bivariate_gaussian_copula',
# 		'variables': ['Z', 'Y'],
# 		'correlation_formula': {'rho': 'cop ~ Z'},
# 		'correlation_params': {'rho': [0.2, 0.5]},
# 		'link': 'jax.nn.sigmoid'
# 	}
# }
class Copula_Model:
	def __init__(self, model_dict: dict) -> None:
		self.functional_regex = r"[\.,\w,\(]+\(([a-zA-Z, , _,-,*,//]+)[\)]+$"

		self.parsed_model = self._process_input_model(model_dict)
		assert self._is_model_ordered_and_acyclic(self.parsed_model)

		pass

	def _process_input_model(self, model_dict: dict) -> dict:
		"""
		Key steps:
		* Check distributions are valid
		* Check params and formula keys are consistent for distribution parameter type
		* Check formula type
		* Check link function is valid
		"""
		parsed_model = dict()

		copula_settings = model_dict.pop('copula', None)
		for var, spec in model_dict.items():
			parsed_model[var] = {'dist': None, 'formula': {}, 'params': {}, 'link': None}

			# Currently use numpyro.distributions in input, but can change
			parsed_model[var]['dist'] = self._map_probability_distributions(model_dict[var]['dist'])

			distribution_params = parsed_model[var]['dist'].arg_constraints.keys()
			assert model_dict[var]['formula'].keys() == distribution_params
			assert model_dict[var]['params'].keys() == distribution_params

			# Check correct number of parameters have been provided
			### TO DO: Fix confusion between `param` and `'params'` entries
			parsed_model[var]['full_formula'] = dict()
			for param in distribution_params:
				parsed_model[var]['formula'][param] = self._regex_variable_adjustment(model_dict[var]['formula'][param])
				assert len(model_dict[var]['params'][param].keys()) == len(
					patsy.ModelDesc.from_formula(
						parsed_model[var]['formula'][param]
					).rhs_termlist
				)
				parsed_model[var]['params'] = model_dict[var]['params'].copy()
				parsed_model[var]['full_formula'][param] = self._align_transformation(
					parsed_model[var]['formula'][param],
					parsed_model[var]['params'][param]
				)

			parsed_model[var]['link'] = self._map_link_functions(model_dict[var]['link'])

		if not copula_settings:
			print('INFO: No copula specified.')
			return parsed_model

		parsed_model['copula'] = {
			'class': None,
			'vars': [],
			'corr_full_formula': {},
			'link': None
		}
		parsed_model['copula']['class'] = COPULA_TAGS[copula_settings['class']]
		parsed_model['copula']['vars'] = copula_settings['vars'].copy()
		for param in copula_settings['formula'].keys():
			parsed_model['copula']['corr_full_formula'] = self._align_transformation(
				copula_settings['formula'][param],
				copula_settings['params'][param]
			)
		parsed_model['copula']['link'] = copula_settings['link']

	
		return parsed_model
		
	def simulate_data(
		self,
		num_warmup: int,
		num_samples: int,
		joint_status: _DISTRIBUTION_TYPES,
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
					
		mcmc_model = numpyro.infer.MCMC(
			kernel,
			num_warmup=num_warmup,
			num_samples=num_samples,
			progress_bar=False
		)
		
		mcmc_model.run(
			rng_key=jax.random.PRNGKey(seed)
		)
		## TO DO: Figure out how to return the data in a useful format.
		return mcmc_model.get_samples()
	

	def _simulated_data_model(self) -> None:
		prob_model = self.parsed_model.copy()
		copula_model = prob_model.pop('copula', None)
		record_dict = {}
		for test_idx, test_row in prob_model.items():
			lin_models_str = test_row['full_formula']

			lin_models_evaluated = {}
			for k, v in lin_models_str.items():
				lin_models_evaluated[k] = eval(v)

			#### ITS NOT DOING ANYTHING WITH THE SCALE PARAMETER --- NEED TO FIX
			if test_row['link']:
				record_dict[test_idx] = numpyro.sample(
					test_idx,
					test_row['dist'](test_row['link'](**lin_models_evaluated))
				)
			else:
				record_dict[test_idx] = numpyro.sample(
					test_idx,
					test_row['dist'](**lin_models_evaluated)
				)
		
		 	# Generate quantiles
			if copula_model and test_idx in copula_model['vars'].keys():
				record_dict[f"q_{test_idx}"] = numpyro.deterministic(
					f"q_{test_idx}",
					prob_model[test_idx](**lin_models_evaluated).cdf(record_dict[test_idx])
				)
				record_dict[f"std_normal_{test_idx}"] = numpyro.deterministic(
					"std_normal_{test_idx}",
					jax.distributions(0, 1).icdf(record_dict[f"q_{test_idx}"])
				)
			
		if copula_model:
			for idx, formula in copula_model['corr_full_formula'].items():
				record_dict[idx] = numpyro.deterministic(
					idx,
					###### PERHAPS CHECK WHETHER A LINK FUNCTION IS NECESSARY FOR EACH PARAMETER
					copula_model['link'](eval(copula_model[idx]['full_formula']))
				)
			

			#### HOW TO MAP THE VARS TO THE DICT
			copula_var_dict = {}
			for k, v in copula_model['vars']:
				copula_var_dict[k] = record_dict[f"std_normal_{v}"]
			for k, v in copula_model['full_formula'].items():
				copula_var_dict[k] = record_dict[k]

			record_dict['cop_log_prob'] = numpyro.factor(
				'cop_log_prob',
				copula_model['class'](**copula_var_dict)
			)


	def _map_probability_distributions(self, prob_distribution: str) -> numpyro.distributions:
		"""
		Map the distribution names in the input model dict
		to their corresponding distribution in Numpyro.
		"""
		return eval(prob_distribution)

	def _map_link_functions(self, link_function: str) -> None:
		"""
		Map the link function in the input model dict
		to its corresponding link function in Numpyro.
		"""
		if not link_function:
			return None
		else:
			return eval(link_function)

	def _is_model_ordered_and_acyclic(self, parsed_model: dict) -> bool:
		"""
		Check whether the input model is acyclic, and whether
		the order variables are entered are """
		print('WARNING: CHECK PASSES AUTOMATICALLY. FULL SOLUTION ADDED TBC')
		return True

	def _align_transformation(self, formula: str, params: list[float]) -> str:
		"""
		Convert input formula and parameters into numpyro-readable format
		for data simulation
		"""
		formula_factors = patsy.ModelDesc.from_formula(formula)
		rhs_terms = formula_factors.rhs_termlist
		params_coeff = list(params.values())

		assert len(rhs_terms) == len(params)

		if len(params_coeff) == 1:
			return str(params_coeff[0])
		if len(params_coeff) >= 2:
			list_of_terms = ([f"{params_coeff[0]}"] +
				[f'{i} * {j.name()}' for i, j in zip(params_coeff[1:], rhs_terms[1:])])
			return  " + ".join(list_of_terms)

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
			enhanced_rhs_term_list.append(
				self.__regex_variable_adjustment(term)
			)
		enhanced_rhs_term_str = ' + '.join(enhanced_rhs_term_list)
		
		enhanced_formula = ' ~ '.join([
			lhs_term_str, enhanced_rhs_term_str
		])
		return enhanced_formula

	def __regex_variable_adjustment(self, formula_term: patsy.Term) -> str:
		"""
		"""
		term_name = formula_term.name()
		if term_name == 'Intercept':
			return '1'

		regexed_term = re.findall(self.functional_regex, term_name)
		if not regexed_term:
			return f"record_dict['{term_name}']"

		key_formula = regexed_term[0]
		orig_vars = re.findall(r"[\w,\_]", key_formula)
		new_vars = []
		for var in orig_vars:
			new_vars.append(f"record_dict['{var}']")
		
		new_formula = key_formula
		for orig_var, new_var in zip(orig_vars, new_vars):
			new_formula = new_formula.replace(orig_var, new_var)
		new_formula = term_name.replace(key_formula, new_formula)
		return new_formula