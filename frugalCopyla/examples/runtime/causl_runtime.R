library(tidyverse)
library(causl)

data_sizes <- c(1e2, 1e3, 1e4, 1e5, 1e6)

list_of_formulae <- list(
  'didelez' = list(
    L ~ A0, 
    list(A0 ~ 1, A1 ~ A0*L),
    Y ~ A0*A1, 
    ~ A0
  ),
  'trivariate_gaussian' = list(
    Z ~ 1, X ~ Z, Y ~ X, ~ 1
  )
)

generate_corr_gaussian_pars <- function(rho) {
  pars <- list(
    Z = list(beta=0, phi=1),
    X = list(beta=c(0, rho), phi=rho),
    Y = list(beta=c(-0.5, 0.5), phi=1),
    cop=list(beta=c(0.5))
  )
}

list_of_pars <- list(
  'didelez' = list(
    A0 = list(beta = 0),
    L = list(beta = c(0.3,-0.2), phi=1),
    A1 = list(beta = c(-0.3,0.4,0.3,0)), 
    Y = list(beta = c(-0.5,0.2,0.3,0), phi=1),
    cop = list(beta = c(1,0.5))
  )
)

list_of_families <- list(
  'didelez' = list(3, c(5, 5), 1, 1),
  'trivariate_gaussian' = list(1, 1, 1, 1)
)

sim_data_functions <- list(
  'rfrugalParam'=rfrugalParam,
  'causlSamp'=causalSamp
)
rhos <- c(0.2, 0.4, 0.6, 0.8)
data_sizes <- c(1e2, 1e3, 1e4, 1e5)#, 1e6)
model_types <- c('didelez', 'trivariate_gaussian')
n_runs <- 5

total_num_runs <- length(data_sizes) * n_runs * (1 + length(rhos)) * length(sim_data_functions)
results <- tibble()
i <- 1
for (sampler_name in names(sim_data_functions)) {
  for (run_iter in 1:n_runs) {
    for (N in data_sizes) {
      for (model_type in model_types) {
        if (model_type == 'trivariate_gaussian') {
          for (rho in rhos) {
            print(paste0('Run ', i, ' / ', total_num_runs))
            pars = generate_corr_gaussian_pars(rho)
            start_time <- Sys.time()
            dat_max <- sim_data_functions[[sampler_name]](
              N, 
              formulas = list_of_formulae[[model_type]], 
              pars=pars, 
              family = list_of_families[[model_type]]
            )
            runtime <- Sys.time() - start_time
            dat_max <- NA
            results <- bind_rows(
              results,
              tibble(sampler=sampler_name, model_type=model_type, N=N, runtime=runtime, rho=rho)
            )    
            i <- i + 1
          }
        } else {
          print(paste0('Run ', i, ' / ', total_num_runs))
          start_time <- Sys.time()
          dat_max <- sim_data_functions[[sampler_name]](
            N, 
            formulas = list_of_formulae[[model_type]], 
            pars=list_of_pars[[model_type]], 
            family = list_of_families[[model_type]]
          )
          runtime <- Sys.time() - start_time
          
          results <- bind_rows(
            results,
            tibble(sampler=sampler_name, model_type=model_type, N=N, runtime=runtime, rho=NA)
          )    
          i <- i + 1
        }
      }
    }
  }
}
print(results)
write_csv(results, 'causl_runtime_results.csv')
