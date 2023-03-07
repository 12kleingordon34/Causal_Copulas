#!/bin/bash

#seq 1 50 | xargs -n 1 -P 6 Rscript single_causl_runtime.R -l
#echo "Pausing for 1 minute"
#sleep 60
#seq 51 70 | xargs -n 1 -P 4 Rscript single_causl_runtime.R -l
#echo "Pausing for 1 minute"
#sleep 60
#seq 71 80 | xargs -n 1 -P 4 Rscript single_causl_runtime.R -l
#echo "Pausing for 1 minute"
#sleep 60
#seq 81 90 | xargs -n 1 -P 2 Rscript single_causl_runtime.R -l
#echo "Pausing for 1 minute"
#sleep 60
#seq 91 100 | xargs -n 1 -P 2 Rscript single_causl_runtime.R -l
#echo "Pausing for 1 minute"
#sleep 60
#seq 101 110| xargs -n 1 -P 2 Rscript single_causl_runtime.R -l
#echo "Pausing for 1 minute"
#sleep 60
#seq 111 130 | xargs -n 1 -P 2 Rscript single_causl_runtime.R -l
#echo "Pausing for 1 minute"
#sleep 60
#seq 131 150 | xargs -n 1 -P 2 Rscript single_causl_runtime.R -l



#Rscript causl_generate_metadata.R

seq 1 155 | xargs -n 1 -P 4 Rscript single_causl_runtime.R -l
#for i in `seq 1 155`
#do 
#	Rscript single_causl_runtime.R -l $i
#done
