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
import shutil
#from def_plot_qq_post import plot_posterior_predictions, plot_qq

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

startTime = datetime.now()

# paths, lists & variables

path_scripts = '/home/ellien/CasA/scripts'
path_spectra = '/home/ellien/CasA/test'
path_data    = '/home/ellien/CasA/test'

path_bxa     = '/home/ellien/CasA/test/acis_bxa_test_powvnei/'
if os.path.isdir( path_bxa ) == False:
    os.makedirs( path_bxa, exist_ok = True )
shutil.copyfile( os.path.abspath(__file__), os.path.join( path_bxa, 'input.script.py' ) )

# xspec model
xs.AllModels.clear()
model_name = 'Tbabs(pow+vnei)'
#________________________________________________________________________
# par  comp
#   1    1   TBabs      nH         10^22    1.00000      +/-  0.0
#   2    2   powerlaw   PhoIndex            1.00000      +/-  0.0
#   3    2   powerlaw   norm                1.00000      +/-  0.0
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
#________________________________________________________________________

list_input_par  = [ [     0.7, 0.001,    0.1,    0.1,    4.0,   4.0 ],
                [           3,  0.01,      1,      1,      4,     4 ],
                [       1e-02,  0.01,  1e-07,  1e-07,  1e-02, 1e-02 ],
                [         1.0,  0.01,    0.1,    0.1,  5e+00, 5e+00 ],
                [           1,  0.01,  1e-01,  1e-01,  1e+05, 1e+05 ],
                [           1,  0.01,  1e-01,  1e-01,  1e+05, 1e+05 ],
                [           1,  0.01,  1e-01,  1e-01,  1e+05, 1e+05 ],
                [           1,  0.01,  1e-01,  1e-01,  1e+04, 1e+04 ],
                [           1,  0.01,  1e-01,  1e-01,  1e+03, 1e+03 ],
                [           1,  0.01,  1e-01,  1e-01,  1e+03, 1e+03 ],
                [           1,  0.01,  1e-01,  1e-01,  1e+03, 1e+03 ],
                [           1,  0.01,  1e-01,  1e-01,  1e+03, 1e+03 ],
                [           1,  0.01,  1e-01,  1e-01,  1e+03, 1e+03 ],
                [           1,  0.01,  1e-01,  1e-01,  1e+03, 1e+03 ],
                [           1,  0.01,  1e-01,  1e-01,  1e+03, 1e+03 ],
                [           1,  0.01,  1e-01,  1e-01,  1e+03, 1e+03 ],
                [           1,  0.01,  1e-01,  1e-01,  1e+03, 1e+03 ],
                [       1e+09, 1e+10,  1e+08,  1e+08,  3e+12, 3e+12 ],
                [           0,  0.01, -0.999, -0.999,     10,    10 ],
                [       1e-07,  0.01,  1e-07,  1e-07,  1e-01, 1e-01 ] ]

model = xs.Model( model_name )

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
infilename = 'spec_test_bayes_2x2arcsec.pi'
infilepath = os.path.join( path_spectra, infilename )

spectrum = xs.Spectrum( infilepath )
os.chdir( path_scripts ) # go back to script directory.

spectrum.ignore( "**-0.5,8.-**" )

# bxa priors
transformations = []
for k in range( 1, model.nParameters + 1 ):
    if k in [  3, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20 ]:
        transformations.append( bxa.create_loguniform_prior_for( model, model(k) ) )
    elif k in [ 5, 6, 7, 8, 17, 19 ]:
        pass
    else:
        transformations.append( bxa.create_uniform_prior_for( model, model(k) ) )

# bxa solver
solver = bxa.BXASolver( transformations = transformations, outputfiles_basename = path_bxa )
results = solver.run( resume = True, log_dir = os.path.join( path_bxa, 'logs' ),  frac_remain = 0.5 )

#plot_qq( solver, path_bxa )
#plot_posterior_predictions( solver, path_bxa )

print(datetime.now() - startTime)
