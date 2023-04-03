#!/bin/bash

for model in vneivneivneipow

do
    for spectrum in /n03data/ellien/CasA/data/opt_selected/opt_*pi
    do
        echo "Launch ${model} on region ${spectrum}"
        qsub qsub_acis_bxa.sh -v spec=${spectrum},mod=${model}
        sleep 2
    done
done
