#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Last modification: 06/2022
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

import xspec as xs
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import glob
import re
from bxa.xspec.solver import XSilence
from matplotlib.offsetbox import AnchoredText

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def find_list_input_par( path_list_input ):

    list_input_par = []
    with open(path_list_input, 'r') as f:
        flag = False
        for line in f:
            flag_split = line.split()

            try:

                if flag_split[0] == 'list_input_par':
                    flag = True

                if flag == True:

                    sublist_input_par = []
                    split = line.split(',')

                    for str in split:

                        split_str = str.split('[')
                        for substr in split_str:

                            split_substr = substr.split(']')
                            for str_input_par in split_substr:

                                try:

                                    if '*' in str_input_par:
                                        subsplit = str_input_par.split('*')
                                        float_input_par = 1.

                                        for sub in subsplit:
                                            float_input_par = float_input_par * float(sub)

                                    else:

                                        float_input_par = float(str_input_par)

                                    sublist_input_par.append(float_input_par)
                                except:
                                    continue

                    list_input_par.append(sublist_input_par)

                if flag_split[-1] == ']':
                    flag = False

            except:
                continue

    return list_input_par

def make_qdp( path_bxa, path_spectra, path_scripts, model_name, display = False ):

    # Read list_input_par in input script
    print(path_bxa)
    try:
        path_list_input = os.path.join( path_bxa, 'input.script.py')
        list_input_par = find_list_input_par( path_list_input )
    except:
        print('No input file ?')
        return None

    # Read results
    pathfile = os.path.join( path_bxa, 'info/results.json' )
    file = open( pathfile )
    results = json.load( file )

    paramnames = results['paramnames']
    #paramvals = results['posterior']['mean']
    paramvals = results['maximum_likelihood']['point']
    logz = results['logz']
    logl = results['maximum_likelihood']['logl']

    #with XSilence():

    os.chdir( path_spectra )
    xs.AllData -= "*"

    spectrum = xs.Spectrum( 'spec_test_bayes_2x2arcsec.pi' )
    spectrum.ignore( "**-0.5,8.-**" )

    # Model
    model = xs.Model( model_name )
    for k in range( 1, model.nParameters + 1 ):
        model(k).values = list_input_par[ k - 1 ]

    # Model parameter values
    for c in model.componentNames:
        for p in model.__dict__[c].parameterNames:

            #print(p, model.__dict__[c].__dict__[p])
            for i, ( bxapn, bxapv ) in enumerate(zip(paramnames, paramvals)):

                bxapv = float(bxapv)

                if bxapn[:3] == 'log':
                    bxapn = bxapn[4:-1]
                    bxapv = 10**bxapv

                if p == bxapn:
                    model.__dict__[c].__dict__[p].values = bxapv
                    paramnames.pop(i)
                    paramvals.pop(i)
                    break

    # Plot with PyXspec
    os.chdir( path_bxa )
    outfile = 'model_values.qdp'
    if os.path.exists(outfile):
        print('Removing previous %s.'%(outfile))
        os.remove(outfile)

    model.show()

    xs.Xset.abund = "angr"
    xs.Xset.cosmo = "70 0 0.73"
    xs.Xset.xsect = "bcmc"
    xs.Fit.statMethod  = 'cstat'

    xs.Plot.commands   = ()
    if display == True:
        xs.Plot.device     = "/xs"
    else:
        xs.Plot.device     = "/null"
    xs.Plot.xLog       = True
    xs.Plot.yLog       = True
    xs.Plot.xAxis      = 'keV'
    xs.Plot.add        = True
    xs.Plot.background = False
    xs.Plot.setRebin( minSig = 5, maxBins = 10, groupNum = -1, errType = 'quad' )
    xs.Plot.addCommand('wd %s' %( outfile ) )
    xs.Plot('ldata delchi')
    xs.Plot.commands = ()

    os.chdir( path_scripts )

    #cstat = xs.Fit.statistic
    dof = xs.Fit.dof
    cstat = -2 * logl

    return cstat, dof, logz

def read_qdp( path_bxa ):

    try:
        pathfile = os.path.join( path_bxa, 'model_values.qdp' )
        data = []
        delchi = []
        flag = False
        with open( pathfile, 'r' ) as f:
            for i in range(3):
                f.readline()
            for line in f:
                split = line.split()
                if flag == False:
                    if split[0] == 'NO':
                        flag = True
                        continue
                    data.append(split)
                else:
                    delchi.append( split[:3] )
        data = np.array(data).astype(np.float)
        delchi = np.array(delchi).astype(np.float)
    except:
        print('No file.')
        return None

    return data, delchi

def plot_bxa_fit_model( data, delchi, path_bxa, stats, model_name, display = False ):

    # Parse data
    eng = data[:, 0]
    eeng = data[:, 1]
    spec = data[:, 2]
    espec = data[:, 3]
    model = data[:, 4]
    delc = delchi[:, 2]

    model_name = model_name[6:-1]
    compnames = model_name.split('+')
    case_name = path_bxa.split('/')[-1]

    # plot
    fig = plt.figure( figsize = (6, 6) )
    gs = fig.add_gridspec(4, 4, wspace = 0., hspace = 0.)
    ax_spec = fig.add_subplot( gs[:3, :] )
    ax_delc = fig.add_subplot( gs[3, :] )

    # source spectrum
    ax_spec.errorbar(eng, spec, xerr = eeng, yerr = espec, fmt = 'o', \
                            markersize = 1, \
                            color = 'black', \
                            label = 'Source', \
                            alpha = 0.5 )

    # model spectrum
    ax_spec.plot(eng, model, color = 'black', linewidth = 2, label = 'Total model' )

    # model components
    for i in range( 5, data.shape[1] ):
        ax_spec.plot(eng, data[:,i], linewidth = 1, alpha = 0.8, linestyle = '--', label = compnames[i - 5])

    ax_spec.set_xscale('log')
    ax_spec.set_yscale('log')
    ax_spec.set_ylabel(r'counts/cm$^2$/s/keV')
    ax_spec.legend(loc = 'lower left')
    #ax_spec.set_title(case_name)
    ax_spec.set_ylim(bottom = 1E-5)

    # delchi
    ax_delc.errorbar(eng, delc, xerr = eeng, yerr = 1, fmt = '+', \
                            markersize = 1, \
                            color = 'black', \
                            label = 'Source', \
                            alpha = 0.8 )

    ax_delc.set_xscale('log')
    ax_delc.axhline( y = 0, color = 'black' )
    ax_delc.set_xlabel('Energy (keV)')

    at = AnchoredText( 'Reduced c-stat = %1.2f' % (stats[0] / stats[1]), \
                       loc = 'lower right', \
                       frameon = False, \
                       pad = 0.3, \
                       borderpad = 1.3, \
                       prop = dict( size = 15 ) ) # Log(Z) = %4.2f\n %stats[2]
    ax_spec.add_artist( at )

    if display == True:
        plt.show()
    plt.savefig( os.path.join( path_bxa, 'fitted_model.pdf' ), format = 'pdf' )

    return None

# custom cutoff model
def sqrtcutoffpl(engs, params, flux):
    for i in range(len(engs)-1):
        pconst = 1.0 - params[0]
        val = np.power(engs[i+1],pconst)/pconst * np.exp( - np.sqrt( engs[i+1] / params[1] ) ) - np.power(engs[i],pconst)/pconst  * np.exp( - np.sqrt( engs[i+1] / params[1] ) )
        flux[i] = val

# z07 model
def z07(engs, params, flux):
    for i in range(len(engs)-1):
        Ec = params[0]
        val = ( 1. + 0.38 * np.sqrt( engs[i] / Ec ))**(2.75) * np.exp( - np.sqrt( engs[i] / Ec ) )
        #      - np.power( engs[i] / Ec, -2. ) * ( 1. + 0.38 * np.sqrt( engs[i] / Ec ))**(2.75) * np.exp( - np.sqrt( engs[i] / Ec ) )
        flux[i] = val




if __name__ == '__main__':

    # paths, lists & variables
    path_bxa_list = glob.glob( '/home/ellien/CasA/test/out3/*' )
    path_spectra = '/home/ellien/CasA/test'
    path_scripts = '/home/ellien/CasA/scripts'

    dic_models = { 'pow':'Tbabs(powerlaw)', \
                   'srcut':'Tbabs(srcut)', \
                   'cutoffpl':'Tbabs(cutoffpl)', \
                   'sqrtcutoffpl':'Tbabs(sqrtcutoffpl)', \
                   'nei':'Tbabs(nei)', \
                   'vnei':'Tbabs(vnei)', \
                   'pshock':'Tbabs(pshock)', \
                   'npshock':'Tbabs(npshock)', \
                   'pownei':'Tbabs(powerlaw+nei)', \
                   'powpshock':'Tbabs(powerlaw+pshock)', \
                   'pownpshock':'Tbabs(powerlaw+npshock)', \
                   'srcutnei':'Tbabs(srcut+nei)', \
                   'srcutpshock':'Tbabs(srcut+pshock)', \
                   'srcutnpshock':'Tbabs(srcut+npshock)', \
                   'cutoffplnei':'Tbabs(cutoffpl+nei)', \
                   'cutoffplpshock':'Tbabs(cutoffpl+pshock)', \
                   'cutoffplnpshock':'Tbabs(cutoffpl+npshock)', \
                   'sqrtcutoffplnei':'Tbabs(sqrtcutoffpl+nei)', \
                   'sqrtcutoffplpshock':'Tbabs(sqrtcutoffpl+pshock)', \
                   'sqrtcutoffplnpshock':'Tbabs(sqrtcutoffpl+npshock)', \
                   'powneivneivnei':'Tbabs(powerlaw+nei+vnei+vnei)' ,\
                   'powpshockvneivnei':'Tbabs(powerlaw+pshock+vnei+vnei)' ,\
                   'powneivnei':'Tbabs(powerlaw+nei+vnei)' ,\
                   'powpshockvnei':'Tbabs(powerlaw+pshock+vnei)' ,\
                   'pownpshockvnei':'Tbabs(pow+npshock+vnei)',\
                   'powvnei':'Tbabs(powerlaw+vnei)', \
                   'cutoffplvnei':'Tbabs(cutoffpl+vnei)', \
                   'sqrtcutoffplvnei':'Tbabs(sqrtcutoffpl+vnei)', \
                   'srcutvnei':'Tbabs(srcut+vnei)', \
                   'sqrtcutoffplneivnei':'Tbabs(sqrtcutoffpl+nei+vnei)', \
                   'sqrtcutoffplpshockvnei':'Tbabs(sqrtcutoffpl+pshock+vnei)', \
                   'sqrtcutoffplnpshockvnei':'Tbabs(sqrtcutoffpl+npshock+vnei)', \
                   'cutoffplneivnei':'Tbabs(cutoffpl+nei+vnei)', \
                   'cutoffplpshockvnei':'Tbabs(cutoffpl+pshock+vnei)', \
                   'cutoffplnpshockvnei':'Tbabs(cutoffpl+npshock+vnei)', \
                   'srcutneivnei':'Tbabs(srcut+nei+vnei)',\
                   'srcutpshockvnei':'Tbabs(srcut+pshock+vnei)',\
                   'srcutnpshockvnei':'Tbabs(srcut+npshock+vnei)',\
                   'z07neivnei':'Tbabs(z07+nei+vnei)' }

    # Add custom models
    myModelParInfo = ("PhoIndex  \"\" 1.1  -3.  -2.  9.  10.  0.01",
                      "cut   \"\" 15. 0. 0. 1e+20 1e+24 0.01")
    xs.AllModels.addPyMod(sqrtcutoffpl, myModelParInfo, 'add')

    myModelParInfo = ("cut   \"\" 1.5 0. 0. 1e+20 1e+24 0.01",)
    xs.AllModels.addPyMod(z07, myModelParInfo, 'add')

    for path_bxa in path_bxa_list:

        print(path_bxa)

        dir = path_bxa.split('/')[-1]
        model_name = dir.split('_')[-1]
        try:
            model_xs = dic_models[model_name]
        except:
            try:
                model_name = dir.split('_')[-2]
                model_xs = dic_models[model_name]
            except:
                print('Key error with %s.' %(dir))
                continue

        stats = make_qdp( path_bxa, path_spectra, path_scripts, model_xs, display = False )
        print(stats[0])
        try:
            data, delchi = read_qdp( path_bxa )
        except:
            continue
        plot_bxa_fit_model( data, delchi, path_bxa, stats, model_xs, display = False )
