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
path_spectra = '/n03data/ellien/CasA/data/Box_3x3_within_20x20'

spectrum = sys.argv[1]
#num_reg = spectrum.split('.')[0][18:] # opt_selected
#num_reg = spectrum.split('.')[0][14:] # selected
num_reg = spectrum.split('.')[0][-3:] # opt_selected_3x3_within_20x20


path_bxa     = '/n03data/ellien/CasA/analysis/out8/acis_bxa_opt_spec_3x3_within_20x20_pownei_%03d/' %int(num_reg)
#path_bxa = '/n03data/ellien/CasA/analysis/out4/acis_bxa_opt_selected_vneivneivneipownei_%03d/' %int(num_reg)
if os.path.isdir( path_bxa ) == False:
    os.makedirs( path_bxa, exist_ok = True )
shutil.copyfile( os.path.abspath(__file__), os.path.join( path_bxa, 'input.script.py' ) )

# xspec model
xs.AllModels.clear()
model_name = 'Tbabs(pow+nei)'
#________________________________________________________________________
#par  comp
#  1    1   TBabs      nH         10^22    1.00000      +/-  0.0
#  2    2   powerlaw   PhoIndex            1.00000      +/-  0.0
#  3    2   powerlaw   norm                1.00000      +/-  0.0
#  4    3   nei        kT         keV      1.00000      +/-  0.0
#  5    3   nei        Abundanc            1.00000      frozen
#  6    3   nei        Tau        s/cm^3   1.00000E+11  +/-  0.0
#  7    3   nei        Redshift            0.0          frozen
#  8    3   nei        norm                1.00000      +/-  0.0
#________________________________________________________________________

list_input_par  = [ [     0.7, 0.001,    0.1,    0.1,    4.0,   4.0 ],
                  [         3,  0.01,    1.5,    1.5,    3.5,   3.5 ],
                  [     1e-06,  0.01,  1e-10,  1e-10,  1e-01, 1e-01 ],
                  [       2.0,  0.01,    0.1,    0.1,  5e+00, 5e+00 ],
                  [         1, -0.01,      0,      0,  1e+03, 1e+03 ],
                  [     1e+10, 1e+10,  1e+09,  1e+09,  8e+11, 8e+11 ],
                  [         0, -0.01, -0.999, -0.999,     10,    10 ],
                  [     1e-07,  0.01,  1e-10,  1e-10,  1e-01, 1e-01 ] ]


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
    if k in [ 3, 6, 8 ]:
        transformations.append( bxa.create_loguniform_prior_for( model, model(k) ) )
    elif k in [ 1, 2, 4 ]:
        transformations.append( bxa.create_uniform_prior_for( model, model(k) ) )
    else:
        pass

# bxa solver
solver = bxa.BXASolver( transformations = transformations, outputfiles_basename = path_bxa )
results = solver.run( resume = True, log_dir = os.path.join( path_bxa, 'logs' ), speed = 10, frac_remain = 0.5, max_num_improvement_loops = 0 )

#plot_qq( solver, path_bxa )
#plot_posterior_predictions( solver, path_bxa )

print(datetime.now() - startTime)
