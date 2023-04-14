import xspec as xs
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import glob
import re
from bxa.xspec.solver import XSilence

if __name__ == '__main__':

    # paths, lists & variables
    path_bxa_list = glob.glob( '/home/ellien/CasA/analysis/out7/*' )
    path_spectra = '/home/ellien/CasA/data/Box_3x3_within_20x20'
    path_scripts = '/home/ellien/CasA/scripts'

    map = np.zeros((7,7))

    for path_bxa in path_bxa_list:

        print(path_bxa)

        dir = path_bxa.split('/')[-1]
        model_name = dir.split('_')[-3]
        num_reg = dir[-3:]  #dir.split('_')[-1]
        x = int(num_reg[0])
        y = int(num_reg[2])


        # Read results
        pathfile = os.path.join( path_bxa, 'info/results.json' )
        file = open( pathfile )
        results = json.load( file )

        paramnames = results['paramnames']
        paramvals = results['posterior']['mean']
        logz = results['logz']
        logl = results['maximum_likelihood']['logl']

        n = 10
        map[ x-1, y-1 ] = paramvals[n]


    plt.figure()
    plt.title(paramnames[n])
    plt.imshow(map, cmap = 'gray', origin = 'upper' )
    plt.colorbar()
    plt.show()
