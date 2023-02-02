# Testing runtime length on `causl`
Make sure all relevant packages are preinstalled.

## Single Thread

Run the script:

```
Rscript causl_runtime.R
```

The results of the simulations are returned in `causl_runtime_results.csv`

## Multithread
Generate a (`csv`) file of the metadata by running
```
Rscript causl_generate_metadata.R
```

See how many different iterations `<M>` are to be simulated by running `wc -l runtime_metadata.csv`. Don't forget to subtract $1$ due to the column headers.

Depending on the number of cores you wish to use (referred to henceforth as `<D>`) run the following command:

```
seq 1 <M> | xargs -n 1 -P <D> Rscript single_causl_runtime.R -l
```

which will run each simulation run on a separate processor, outputing the results of each run as a separate csv within the subdirectory `/data/`.

A script to unify these separate files into one single results file will be made shortly....