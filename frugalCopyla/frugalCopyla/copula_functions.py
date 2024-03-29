import jax.numpy as jnp
import jax.scipy.stats as jax_stats


def __reshape_matrix(n_vars: int, rhos: list[float], is_cholesky=False) -> jnp.array:
    assert int((n_vars ** 2 - n_vars) / 2) == len(rhos)
    position_idx = 0
    for n in reversed(range(1, n_vars+1)):
        rhos.insert(position_idx, 1)
        position_idx += n
    corr = jnp.zeros((n_vars, n_vars))
    triu = jnp.triu_indices(n_vars)
    tril = jnp.tril_indices(n_vars, -1)
    if is_cholesky:
        corr = corr.at[triu].set(rhos).T
        for i in range(len(corr)):
            corr = corr.at[i, i].set(jnp.sqrt(2 - corr[i, ].dot(corr[i, ])))
    else:
        corr = corr.at[triu].set(rhos).T.at[triu].set(rhos)
    return corr


def multivar_gaussian_copula_lpdf(vars: dict, rhos: dict) -> float:
    rvs = jnp.array(list(vars.values()))
    mean = jnp.zeros(len(rvs))
    cov = __reshape_matrix(len(rvs), list(rhos.values()), is_cholesky=False)
    llhood = jax_stats.multivariate_normal.logpdf(rvs, mean, cov)	
    return llhood


#def bivariate_gaussian_copula_lpdf(u, v, rho) -> float:
def bivariate_gaussian_copula_lpdf(vars: dict, rhos: dict) -> float:
    u, v = vars.values()
    rho = list(rhos.values())[0]
    u_2 = jnp.square(u)
    v_2 = jnp.square(v)
    rho_2 = jnp.square(rho)
    llhood = (
        -0.5 * (1 - rho_2) - (
            rho_2 * (u_2 + v_2) - 2 * rho * u * v
        ) / (2 * (1 - rho_2))
    )
    return llhood
    


def chol_multivar_gaussian_copula_lpdf(vars: dict, rhos: dict) -> float:
    rvs = jnp.array(list(vars.values()))
    mean = jnp.zeros(len(rvs))
    chol_cov = __reshape_matrix(len(rvs), list(rhos.values()), is_cholesky=True)
    cov = jnp.matmul(chol_cov, chol_cov.T)
    llhood = jax_stats.multivariate_normal.logpdf(rvs, mean, cov)	
    if not jnp.isnan(llhood).any():
        return llhood
    else:
        return -jnp.inf
    