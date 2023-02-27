#!/bin/bash
#PBS -o /home/ellien/CasA/logs/log_acis_bxa_test_vnei.log
#PBS -j oe
#PBS -N bxa
#PBS -l nodes=1:ppn=8,walltime=47:59:00
#PSB -S /bin/bash

conda init bash
source /home/ellien/.bashrc
conda env list

module () {
  eval $(/usr/bin/modulecmd bash $*)
}

module purge
module load gcc/10.2.0
module load intelpython/3-2021.1.1

export HEADAS="/softs/heasoft/6.28-python3/x86_64-pc-linux-gnu-libc2.17/"
. $HEADAS/headas-init.sh

conda activate 2021.1.1
conda env list

#python -c 'import xspec'
#python -c 'import bxa.xspec'
#python -c 'from mpi4py import MPI'
#which mpiexec

#echo "mpiexec -n 36 ipython /home/ellien/Tycho/scripts/acis_bxa_region_${nr}_${mod}.py"
mpiexec -n 8 ipython /home/ellien/CasA/CasA/bxa/acis_bxa_test_vnei.py
#mpiexec -n 48 ipython /home/ellien/Tycho/scripts/acis_bxa_ejecta_${nr}_${mod}.py
