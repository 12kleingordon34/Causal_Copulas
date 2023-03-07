suppressMessages(library(argparse))
suppressMessages(library(tidyverse))
suppressMessages(library(causl))

suppressMessages(gc())

parser <- ArgumentParser()
parser$add_argument("-l", "--line", default=TRUE,
                    help="Line to read from metadata")

args <- parser$parse_args()
i <- args$line

metadata <- suppressMessages(read_csv('runtime_metadata.csv'))
run_settings <- metadata[i,]

model_type <- run_settings$model_type
sampler_name <- run_settings$sampler_name
N <- run_settings$N
rho <- run_settings$rho
run_idx <- run_settings$rho

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


pars = generate_corr_gaussian_pars(rho)
start_time <- Sys.time()
if (model_type == 'trivariate_gaussian') {
  dat_max <- sim_data_functions[[sampler_name]](
    N,
    formulas = list_of_formulae[[model_type]],
    pars=pars,
    family = list_of_families[[model_type]]
  )
} else {
  dat_max <- sim_data_functions[[sampler_name]](
    N,
    formulas = list_of_formulae[[model_type]],
    pars=list_of_pars[[model_type]],
    family = list_of_families[[model_type]]
  )
}
runtime <- difftime(Sys.time(), start_time, units='s')
results <- tibble(
  sampler_name=sampler_name, model_type=model_type, N=N, runtime=runtime, run_idx=i, rho=rho
)

print(paste0("Time: ", Sys.time(), ' -- Run ', i))
if (!dir.exists('./data/')){
  dir.create('./data/')
}
write_csv(results, paste0('./data/results_row_', i, '.csv'))
suppressMessages(gc())