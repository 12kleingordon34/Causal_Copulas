import jax.numpy as jnp


def bivariate_gaussian_copula_lpdf(u, v, rho):
    u_2 = jnp.square(u)
    v_2 = jnp.square(v)
    rho_2 = jnp.square(rho)
    return (
        -0.5 * (1 - rho_2) - (
            rho_2 * (u_2 + v_2) - 2 * rho * u * v
        ) / (2 * (1 - rho_2))
    )
	
_COPULA_TAGS = {
	'bivariate_gaussian_copula': bivariate_gaussian_copula_lpdf
}
