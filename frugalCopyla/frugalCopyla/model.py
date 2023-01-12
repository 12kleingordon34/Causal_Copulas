import random
import re
from typing import Literal

import jax
import jax.numpy as jnp
import numpyro
import patsy

_DISTRIBUTION_TYPES = ['discrete', 'continuous', 'mixed']

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
		for var, spec in model_dict.items():
			parsed_model[var] = {'dist': None, 'formula': {}, 'params': {}, 'link': None}

			# Currently use numpyro.distributions in input, but can change
			parsed_model[var]['dist'] = self._map_probability_distributions(model_dict[var]['dist'])

			distribution_params = parsed_model[var]['dist'].arg_constraints.keys()
			assert model_dict[var]['formula'].keys() == distribution_params
			assert model_dict[var]['params'].keys() == distribution_params

			# Check correct number of parameters have been provided
			### TO DO: Fix confusion between `param` and `'params'` entries
			for param in distribution_params:
				parsed_model[var]['formula'][param] = self._regex_variable_adjustment(model_dict[var]['formula'][param])
				assert len(model_dict[var]['params'][param].keys()) == len(
					patsy.ModelDesc.from_formula(
						parsed_model[var]['formula'][param]
					).rhs_termlist
				)
				parsed_model[var]['params'] = model_dict[var]['params'].copy()

			parsed_model[var]['link'] = self._map_link_functions(model_dict[var]['link'])
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
			key=jax.random.PRNGKey(seed)
		)
		## TO DO: Figure out how to return the data in a useful format.
		pass
	

	def _simulated_data_model(self) -> None:
		record_dict = {}
		for test_idx, test_row in self.parsed_model.items():
			test_formula = self._align_transformation(test_row['formula']['loc'], test_row['params']['loc']) 
			#### ITS NOT DOING ANYTHING WITH THE SCALE PARAMETER --- NEED TO FIX
			if test_row['link']:
				record_dict[test_idx] = numpyro.sample(
					test_idx,
					test_row['dist'](test_row['link'](eval(test_formula)))
				)
			else:
				record_dict[test_idx] = numpyro.sample(
					test_idx,
					test_row['dist']((eval(test_formula)))
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

	def _align_transformation(self, formula, params) -> str:
		"""
		Convert input formula and parameters into numpyro-readable format
		for data simulation
		"""
		pass

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
			return term_name

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