#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Last modification: 07/2021
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# MODULES

import xspec as xs
import bxa.xspec as bxa
import numpy as np
import matplotlib.pyplot as plt
import os
import ray
from datetime import datetime
import shutil

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

startTime = datetime.now()

# paths, lists & variables
#path_scripts = '/home/ellien/CasA/scripts'
#path_spectra = '/home/ellien/CasA/test'
#path_data    = '/home/ellien/CasA/test'

path_scripts = '/home/ellien/CasA/CasA'
path_data    = '/n03data/ellien/CasA/tests'
path_spectra = '/n03data/ellien/CasA/tests'

path_bxa     = '/n03data/ellien/CasA/tests/out3/acis_bxa_test_freeab_8core_vnei/'
if os.path.isdir( path_bxa ) == False:
    os.makedirs( path_bxa, exist_ok = True )
shutil.copyfile( os.path.abspath(__file__), os.path.join( path_bxa, 'input.script.py' ) )

# xspec model
xs.AllModels.clear()
model_name = 'Tbabs(vnei)'
#_______________________________________________________________________________
# par  comp
#   1    1   TBabs      nH         10^22    1.00000      +/-  0.0
#   2    2   vnei       kT         keV      1.00000      +/-  0.0
#   3    2   vnei       H                   1.00000      frozen
#   4    2   vnei       He                  1.00000      frozen
#   5    2   vnei       C                   1.00000      frozen
#   6    2   vnei       N                   1.00000      frozen
#   7    2   vnei       O                   1.00000      frozen
#   8    2   vnei       Ne                  1.00000      frozen
#   9    2   vnei       Mg                  1.00000      frozen
#  10    2   vnei       Si                  1.00000      frozen
#  11    2   vnei       S                   1.00000      frozen
#  12    2   vnei       Ar                  1.00000      frozen
#  13    2   vnei       Ca                  1.00000      frozen
#  14    2   vnei       Fe                  1.00000      frozen
#  15    2   vnei       Ni                  1.00000      frozen
#  16    2   vnei       Tau        s/cm^3   1.00000E+11  +/-  0.0
#  17    2   vnei       Redshift            0.0          frozen
#  18    2   vnei       norm                1.00000      +/-  0.0
#________________________________________________________________________


list_input_par = [ [      0.7, 0.001,    0.1,    0.1,    4.0,   4.0 ],
                [         1.0,  0.01,    0.1,    0.1,  5e+00, 5e+00 ],
                [           1,  0.01,  1e-01,  1e-01,  1e+01, 1e+01 ],
                [           1,  0.01,  1e-01,  1e-01,  1e+01, 1e+01 ],
                [           1,  0.01,  1e-01,  1e-01,  1e+01, 1e+01 ],
                [           1,  0.01,  1e-01,  1e-01,  1e+01, 1e+01 ],
                [           1,  0.01,  1e-01,  1e-01,  1e+01, 1e+01 ],
                [           1,  0.01,  1e-01,  1e-01,  1e+01, 1e+01 ],
                [           1,  0.01,  1e-01,  1e-01,  1e+01, 1e+01 ],
                [           1,  0.01,  1e-01,  1e-01,  1e+01, 1e+01 ],
                [           1,  0.01,  1e-01,  1e-01,  1e+01, 1e+01 ],
                [           1,  0.01,  1e-01,  1e-01,  1e+01, 1e+01 ],
                [           1,  0.01,  1e-01,  1e-01,  1e+01, 1e+01 ],
                [           1,  0.01,  1e-01,  1e-01,  1e+01, 1e+01 ],
                [           1,  0.01,  1e-01,  1e-01,  1e+01, 1e+01 ],
                [       1e+10, 1e+10,  5e+09,  5e+09,  8e+11, 8e+11 ],
                [           0,  0.01, -0.999, -0.999,     10,    10 ],
                [       1e-02,  0.01,  1e-04,  1e-04,  1e-01, 1e-01 ] ]

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

spectrum.ignore( "**-0.5,7.-**" )
#SPECTRUM.ignore( "1-35,480-1024" )

# bxa priors
transformations = []
for k in range( 1, model.nParameters + 1 ):
    if k in [ 8, 9, 10, 11, 12, 13, 14, 16, 18 ]:
        transformations.append( bxa.create_loguniform_prior_for( model, model(k) ) )
    elif k in [ 3, 4, 5, 6, 7, 15, 17 ]:
        pass
    else:
        transformations.append( bxa.create_uniform_prior_for( model, model(k) ) )

# bxa solver
solver = bxa.BXASolver( transformations = transformations, outputfiles_basename = path_bxa )
results = solver.run( resume = 'overwrite', log_dir = os.path.join( path_bxa, 'logs' ), speed = 'auto', frac_remain = 0.5 )

print(datetime.now() - startTime)
