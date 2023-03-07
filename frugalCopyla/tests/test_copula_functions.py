import jax.numpy as jnp
import pytest

import frugalCopyla.copula_functions as cops


def test_binary_non_chol_reshape_matrix():
	n_vars = 2
	rhos = [0.5]
	outcome_corr = cops._reshape_matrix(n_vars, rhos, is_cholesky=False)
	target_outcome = jnp.array([
		[1, 0.5],
		[0.5, 1]
	]) 
	assert (outcome_corr == target_outcome).all()


def test_three_class_non_chol_reshape_matrix():
	n_vars = 3
	rhos = [0.1, 0.2, 0.3]
	outcome_corr = cops._reshape_matrix(n_vars, rhos, is_cholesky=False)
	target_outcome = jnp.array([
		[1, 0.1, 0.2],
		[0.1, 1, 0.3],
		[0.2, 0.3, 1]
	]) 
	assert (outcome_corr == target_outcome).all()


def test_four_class_non_chol_reshape_matrix():
	n_vars = 4
	rhos = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
	outcome_corr = cops._reshape_matrix(n_vars, rhos, is_cholesky=False)
	target_outcome = jnp.array([
		[1, 0.1, 0.2, 0.3],
		[0.1, 1, 0.4, 0.5],
		[0.2, 0.4, 1, 0.6],
		[0.3, 0.5, 0.6, 1]
	]) 
	assert (outcome_corr == target_outcome).all()


def test_binary_chol_reshape_matrix():
	pass


def test_three_class_chol_reshape_matrix():
	pass
