#!/bin/bash

for model in vneivneipownei

do
    for spectrum in /n03data/ellien/CasA/data/synth_spec/Synth*
        do
        echo "Launch ${model} on region ${spectrum}"
        n=$(basename "$spectrum")
        qsub qsub_acis_bxa.sh -v spec=${n},mod=${model}
    done
done
