#!/bin/bash

for model in vneivneivneipow

do
    for spectrum in /n03data/ellien/CasA/data/opt_selected/opt_spec_selected_1.pi
    do
        echo "Launch ${model} on region ${spectrum}"
        n=$(basename "$spectrum")
        qsub qsub_acis_bxa.sh -v spec=${n},mod=${model}
        sleep 2
    done
done
