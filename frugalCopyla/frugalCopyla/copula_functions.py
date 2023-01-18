import jax.scipy.stats as jax_stats
import jax.numpy as jnp


def reshape_matrix(n_vars: int, rhos: list[float]) -> jnp.array:
    assert int((n_vars ** 2 - n_vars) / 2) == len(rhos)
    position_idx = 0
    for n in reversed(range(1, n_vars+1)):
        rhos.insert(position_idx, 1)
        position_idx += n
    corr = jnp.zeros((n_vars, n_vars))
    triu = jnp.triu_indices(n_vars)
    tril = jnp.tril_indices(n_vars, -1)
    corr = corr.at[triu].set(rhos)
    corr = corr.at[tril].set(corr.T[tril])
    return corr


def bivariate_gaussian_copula_lpdf(u, v, rho) -> float:
    u_2 = jnp.square(u)
    v_2 = jnp.square(v)
    rho_2 = jnp.square(rho)
    return (
        -0.5 * (1 - rho_2) - (
            rho_2 * (u_2 + v_2) - 2 * rho * u * v
        ) / (2 * (1 - rho_2))
    )


def multivar_gaussian_copula_lpdf(vars: dict, rhos: dict) -> float:
    rvs = jnp.array(list(vars.values()))
    mean = jnp.zeros(len(rvs_list))
    cov = reshape_matrix(len(rvs_list), list(rhos.values()))
    return jax_stats.multivariate_normal.logpdf(rvs, mean, cov)	


_COPULA_TAGS = {
	'bivariate_gaussian_copula': bivariate_gaussian_copula_lpdf,
    'multivariate_gaussian_copula': multivar_gaussian_copula_lpdf
}
