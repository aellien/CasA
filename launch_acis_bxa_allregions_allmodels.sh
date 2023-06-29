#!/bin/bash

for model in vneivneivneipow

do
    for spectrum in /n03data/ellien/CasA/data/Box_3x3_within_20x20/opt_spec_3x3_within_20x20_*.pi
    do
        echo "Launch ${model} on region ${spectrum}"
        n=$(basename "$spectrum")
        qsub qsub_acis_bxa.sh -v spec=${n},mod=${model}
    done
done
