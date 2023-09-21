#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Last modification: 05/2022
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# MODULES

import xspec as xs
import bxa.xspec as bxa
import numpy as np
import matplotlib.pyplot as plt
import os
import ray
import json
from datetime import datetime
import sys
import shutil
#from def_plot_qq_post import plot_posterior_predictions, plot_qq

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

startTime = datetime.now()

# paths, lists & variables
path_scripts = '/home/ellien/CasA/CasA'
path_priors  = '/n03data/ellien/CasA/data/selected'
path_spectra = '/n03data/ellien/CasA/data/Test_spec4x4_within25x25'

spectrum = sys.argv[1]
num_reg = spectrum.split('.')[0][-3:] # spec_4x4_within_25x25_5_4

path_bxa = '/n03data/ellien/CasA/analysis/out10/acis_bxa_spec4x4_within25x25_vneivneivneipow_%s/' %num_reg
if os.path.isdir( path_bxa ) == False:
    os.makedirs( path_bxa, exist_ok = True )
shutil.copyfile( os.path.abspath(__file__), os.path.join( path_bxa, 'input.script.py' ) )

# xspec model
xs.AllModels.clear()
model_name = 'Tbabs(gsmooth(vnei+vnei+vnei)+pow)'
#________________________________________________________________________
# par  comp
#   1    1   TBabs      nH         10^22    1.00000      +/-  0.0
#   2    2   gsmooth    Sig_6keV   keV      1.00000      +/-  0.0
#   3    2   gsmooth    Index               0.0          frozen
#   4    3   vnei       kT         keV      1.00000      +/-  0.0
#   5    3   vnei       H                   1.00000      frozen
#   6    3   vnei       He                  1.00000      frozen
#   7    3   vnei       C                   1.00000      frozen
#   8    3   vnei       N                   1.00000      frozen
#   9    3   vnei       O                   1.00000      frozen
#  10    3   vnei       Ne                  1.00000      frozen
#  11    3   vnei       Mg                  1.00000      frozen
#  12    3   vnei       Si                  1.00000      frozen
#  13    3   vnei       S                   1.00000      frozen
#  14    3   vnei       Ar                  1.00000      frozen
#  15    3   vnei       Ca                  1.00000      frozen
#  16    3   vnei       Fe                  1.00000      frozen
#  17    3   vnei       Ni                  1.00000      frozen
#  18    3   vnei       Tau        s/cm^3   1.00000E+11  +/-  0.0
#  19    3   vnei       Redshift            0.0          frozen
#  20    3   vnei       norm                1.00000      +/-  0.0
#  21    4   vnei       kT         keV      1.00000      +/-  0.0
#  22    4   vnei       H                   1.00000      frozen
#  23    4   vnei       He                  1.00000      frozen
#  24    4   vnei       C                   1.00000      frozen
#  25    4   vnei       N                   1.00000      frozen
#  26    4   vnei       O                   1.00000      frozen
#  27    4   vnei       Ne                  1.00000      frozen
#  28    4   vnei       Mg                  1.00000      frozen
#  29    4   vnei       Si                  1.00000      frozen
#  30    4   vnei       S                   1.00000      frozen
#  31    4   vnei       Ar                  1.00000      frozen
#  32    4   vnei       Ca                  1.00000      frozen
#  33    4   vnei       Fe                  1.00000      frozen
#  34    4   vnei       Ni                  1.00000      frozen
#  35    4   vnei       Tau        s/cm^3   1.00000E+11  +/-  0.0
#  36    4   vnei       Redshift            0.0          frozen
#  37    4   vnei       norm                1.00000      +/-  0.0
#  38    5   vnei       kT         keV      1.00000      +/-  0.0
#  39    5   vnei       H                   1.00000      frozen
#  40    5   vnei       He                  1.00000      frozen
#  41    5   vnei       C                   1.00000      frozen
#  42    5   vnei       N                   1.00000      frozen
#  43    5   vnei       O                   1.00000      frozen
#  44    5   vnei       Ne                  1.00000      frozen
#  45    5   vnei       Mg                  1.00000      frozen
#  46    5   vnei       Si                  1.00000      frozen
#  47    5   vnei       S                   1.00000      frozen
#  48    5   vnei       Ar                  1.00000      frozen
#  49    5   vnei       Ca                  1.00000      frozen
#  50    5   vnei       Fe                  1.00000      frozen
#  51    5   vnei       Ni                  1.00000      frozen
#  52    5   vnei       Tau        s/cm^3   1.00000E+11  +/-  0.0
#  53    5   vnei       Redshift            0.0          frozen
#  54    5   vnei       norm                1.00000      +/-  0.0
#  55    6   powerlaw   PhoIndex            1.00000      +/-  0.0
#  56    6   powerlaw   norm                1.00000      +/-  0.0
#________________________________________________________________________


list_input_par  = [[     0.7, 0.001,    0.1,    0.1,    4.0,   4.0 ],
                  [      0.1,  0.01,  1e-03,  1e-03,   0.15,  0.15 ],
                  [        1, -0.01,  1e-01,  1e-01,  1e+05, 1e+05 ],
                  [      2.0,  0.01,    0.1,    0.1,  5e+00, 5e+00 ],
                  [        1, -0.01,  1e-01,  1e-01,  1e+05, 1e+05 ],
                  [        1, -0.01,  1e-01,  1e-01,  1e+05, 1e+05 ],
                  [        1, -0.01,  1e-01,  1e-01,  1e+05, 1e+05 ],
                  [      100, -0.01,  1e-01,  1e-01,  1e+05, 1e+05 ],
                  [    1e+04, -0.01,  1e-01,  1e-01,  1e+05, 1e+05 ],
                  [    2e+03,  0.01,  1e-01,  1e-01,  1e+05, 1e+05 ],
                  [    2e+02,  0.01,  1e-01,  1e-01,  1e+05, 1e+05 ],
                  [        0, -0.01,      0,      0,  1e+03, 1e+03 ],
                  [        0, -0.01,      0,      0,  1e+03, 1e+03 ],
                  [        0, -0.01,      0,      0,  1e+03, 1e+03 ],
                  [        0, -0.01,      0,      0,  1e+03, 1e+03 ],
                  [        0, -0.01,      0,      0,  1e+03, 1e+03 ],
                  [        0, -0.01,      0,      0,  1e+03, 1e+03 ],
                  [    1e+10, 1e+10,  5e+09,  5e+09,  8e+11, 8e+11 ],
                  [        0, 0.005,  -0.01,  -0.01,   0.01,  0.01 ],
                  [    1e-07,  0.01,  1e-10,  1e-10,  1e-01, 1e-01 ],
                  [      2.0,  0.01,    0.1,    0.1,  5e+00, 5e+00 ],
                  [        1, -0.01,  1e-01,  1e-01,  1e+05, 1e+05 ],
                  [        1, -0.01,  1e-01,  1e-01,  1e+05, 1e+05 ],
                  [        1, -0.01,  1e-01,  1e-01,  1e+05, 1e+05 ],
                  [        0, -0.01,      0,      0,  1e+04, 1e+04 ],
                  [        0, -0.01,      0,      0,  1e+03, 1e+03 ],
                  [        0, -0.01,      0,      0,  1e+03, 1e+03 ],
                  [        0, -0.01,      0,      0,  1e+03, 1e+03 ],
                  [    1e+04, -0.01,  1e-01,  1e-01,  1e+05, 1e+05 ],
                  [    1e+04,  0.01,  1e-01,  1e-01,  1e+05, 1e+05 ],
                  [    1e+04,  0.01,  1e-01,  1e-01,  1e+05, 1e+05 ],
                  [    3e+04,  0.01,  1e-01,  1e-01,  1e+05, 1e+05 ],
                  [        0, -0.01,      0,      0,  1e+03, 1e+03 ],
                  [        0, -0.01,      0,      0,  1e+03, 1e+03 ],
                  [    1e+10, 1e+10,  5e+09,  5e+09,  8e+11, 8e+11 ],
                  [        0, 0.005,  -0.01,  -0.01,   0.01,  0.01 ],
                  [    1e-07,  0.01,  1e-10,  1e-10,  1e-01, 1e-01 ],
                  [      2.0,  0.01,    0.1,    0.1,  5e+00, 5e+00 ],
                  [        1, -0.01,  1e-01,  1e-01,  1e+05, 1e+05 ],
                  [        1, -0.01,  1e-01,  1e-01,  1e+05, 1e+05 ],
                  [        1, -0.01,  1e-01,  1e-01,  1e+05, 1e+05 ],
                  [        0, -0.01,      0,      0,  1e+04, 1e+04 ],
                  [        0, -0.01,      0,      0,  1e+03, 1e+03 ],
                  [        0, -0.01,      0,      0,  1e+03, 1e+03 ],
                  [        0, -0.01,      0,      0,  1e+03, 1e+03 ],
                  [        0, -0.01,      0,      0,  1e+05, 1e+05 ],
                  [        0, -0.01,      0,      0,  1e+03, 1e+03 ],
                  [        0, -0.01,      0,      0,  1e+03, 1e+03 ],
                  [        0, -0.01,      0,      0,  1e+03, 1e+03 ],
                  [    1e+04, -0.01,  1e-01,  1e-01,  1e+03, 1e+05 ],
                  [    1e+04, -0.01,  1e-01,  1e-01,  1e+03, 1e+05 ],
                  [    1e+10, 1e+10,  5e+09,  5e+09,  8e+11, 8e+11 ],
                  [        0, 0.005,  -0.01,  -0.01,   0.01,  0.01 ],
                  [    1e-07,  0.01,  1e-10,  1e-10,  1e-01, 1e-01 ],
                  [        3,  0.01,    1.5,    1.5,    3.5,   3.5 ],
                  [    1e-06,  0.01,  1e-10,  1e-10,  1e-01, 1e-01 ] ]

model = xs.Model(model_name )

for k in range( 1, model.nParameters + 1 ):
    model(k).values = list_input_par[ k - 1 ]

ncomp = len( model.componentNames )

# xspec xset parameters
xs.Xset.abund = "angr"
xs.Xset.cosmo = "70 0 0.73"
xs.Xset.xsect = "bcmc"

# xspec fitting parameters
xs.Fit.statMethod  = 'cstat'
xs.Fit.method      = "leven 1000 0.01"
#xs.Fit.query       = "no"

os.chdir( path_spectra ) # Necessary to bypass the abscence of absolute path in
                         # anscillary file names in primary file header.

xs.AllData -= "*" # Clear all previous spectra.

# xspec read spectrum
infilepath = os.path.join( path_spectra, spectrum )

spectrum = xs.Spectrum( infilepath )
os.chdir( path_scripts ) # go back to script directory.

spectrum.ignore( "**-0.5,8.-**" )

# bxa priors
transformations = []
for k in range( 1, model.nParameters + 1 ):
    if k in [ 2, 10, 11, 18, 20, 30, 31, 32, 35, 37, 52, 54, 56 ]:
        transformations.append( bxa.create_loguniform_prior_for( model, model(k) ) )
    elif k in [ 1, 4, 19, 21, 36, 38, 53, 55 ]:
        transformations.append( bxa.create_uniform_prior_for( model, model(k) ) )
    else:
        pass

# bxa solver
solver = bxa.BXASolver( transformations = transformations, outputfiles_basename = path_bxa )
results = solver.run( resume = True, log_dir = os.path.join( path_bxa, 'logs' ), speed = 10, frac_remain = 0.5, max_num_improvement_loops = 0 )

#plot_qq( solver, path_bxa )
#plot_posterior_predictions( solver, path_bxa )

print(datetime.now() - startTime)
