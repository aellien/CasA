#!/bin/bash

for model in vneivneivneipow

do
    for spectrum in /n03data/ellien/CasA/data//n03data/ellien/CasA/data/Test_spec4x4_within25x25/*pi
    do
        echo "Launch ${model} on region ${spectrum}"
        n=$(basename "$spectrum")
        qsub qsub_acis_bxa.sh -v spec=${n},mod=${model}
    done
done
