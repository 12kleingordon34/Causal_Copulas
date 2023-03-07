library(tidyverse)
library(causl)

rhos <- c(0.2, 0.4, 0.6, 0.8)
data_sizes <- c(1e2, 1e3, 1e4, 1e5, 1e6, 2e6, 1e7)
model_types <- c('didelez', 'trivariate_gaussian')
sampler_names <- c('rfrugalParam')#, 'causlSamp')
n_runs <- seq(1, 5, 1)

metadata <- expand.grid(
  N=data_sizes, rho=rhos, model_type=model_types, run_idx=n_runs, sampler_name=sampler_names
) %>%
  mutate(rho=if_else(model_type == 'trivariate_gaussian', rho, 0)) %>%
  distinct(N, rho, model_type, run_idx, sampler_name)

write_csv(metadata, 'runtime_metadata.csv')
