# qPyCR fitter for Python 2.7


import numpy as np
import pandas as pd
import scipy.optimize as opt
import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm




# calculate weight from brightness, or return 1's if disabled
def calcWeight(yieldY, weight_opt):
    weightY = []
    for y in yieldY:
        if weight_opt=="False":
            # disable weight, set all to 1
            w=1
        else:
            w = 1 / abs(y)
            # set upper limit for weight
            if w > 100:
                w = 100
        weightY.append(w)
    return weightY





# EQUATION 6 from Moore/Caar 2012
# calculate yield from previous cycle to optimize max and Kd
def eqn6_yield(prevX, maxZ, KdZ):
    return prevX * (1 + (maxZ - prevX) / maxZ - prevX / (KdZ + prevX))





# squared differences, just for fun
def sq_diff(x,y):
    return (x-y)**2





# optimize SSD for seed, given max, Kd, and real yield
def seedSSD(seed, maxZ, KdZ, prevX, yieldY):
    SSD = sq_diff(seed, prevX[0])
    prevZ=seed
    for Y in yieldY:
        y=eqn6_yield(prevZ, maxZ, KdZ)
        SSD = SSD + sq_diff(y,Y)
        prevZ=y

    return(SSD)





# sum of squares
def sum_of_squares(num_list):
    ss=0
    for n in num_list:
        ss=ss+(n**2)
    return ss





# main fitting function, puts it all together
def qpcr_fit(prevX, yieldY, weight_opt):

    # set variables for profile fit
    maxY = max(yieldY)
    guess0 = [5*maxY, maxY/5]
    boundZ = ( (0,0), (25*maxY,maxY) )

    # calculate weight
    weightY = calcWeight(yieldY, weight_opt)

    # optimize Max and Kd
    popt, pcov = opt.curve_fit(eqn6_yield, prevX, yieldY,
        p0=guess0, sigma=weightY, bounds=boundZ)
    maxZ = popt[0]
    KdZ = popt[1]

    # optimize Seed for SSD
    argZ = (maxZ, KdZ, prevX, yieldY)
    boundZ = (0, max(yieldY))
    res = opt.minimize_scalar(seedSSD, args=argZ,
        bounds=boundZ, method='bounded', tol=1e-100)
    seedZ = res.x
    SSDZ = res.fun

    # return the values
    return(maxZ, KdZ, seedZ, SSDZ)








# calculate yield from seed, max, Kd, and number of cycles
def calcYield(seed, maxZ, KdZ, cycs):
    ylist=[seed,]
    for cyc in range(0,cycs):
        prev=ylist[-1]
        y=eqn6_yield(prev, maxZ, KdZ)
        ylist.append(y)

    return ylist












def get_stats(csvDF, weight_opt):
    # make empty lists
    samp_list=[]
    max_list=[]
    Kd_list=[]
    seed_list=[]
    SSD_list=[]

    # read dataframe without the Cycle column
    for column in csvDF.drop(columns=['Cycle']):

        # get data and header
        rawYield = csvDF[column].tolist()
        sampZ = csvDF[column].name
        prevX = rawYield[:-1]
        yieldY = rawYield[1:]

        # do the fitting
        maxZ, KdZ, seedZ, SSDZ = qpcr_fit(prevX, yieldY, weight_opt)

        # add data to lists
        samp_list.append(sampZ)
        max_list.append(maxZ)
        Kd_list.append(KdZ)
        seed_list.append(seedZ)
        SSD_list.append(SSDZ)


    # organize dataframe for stats.csv
    statsDF=pd.DataFrame()
    statsDF['Sample']=samp_list
    statsDF['Max']=max_list
    statsDF['Kd']=Kd_list
    statsDF['Seed']=seed_list
    statsDF['SSD']=SSD_list

    return statsDF






def get_fit(statsDF, num_cycs):
    fitDF=pd.DataFrame()
    fitDF['Cycle'] = range(1,num_cycs+1)
    for index,row in statsDF.iterrows():

        # get the variables for fitting
        sampY = row['Sample']
        maxY = row['Max']
        KdY = row['Kd']
        seedY = row['Seed']

        # generate fit
        fitDF[sampY] = calcYield(seedY, maxY, KdY, num_cycs-1)

    return fitDF













def groupSamples(csvDF, statsDF, weight_opt, num_cycs):

    # make a list of unique groups "<group>-<name>"
    samp_list=list(csvDF)
    if 'Cycle' in samp_list:
        samp_list.remove('Cycle')

    group_list=[]
    for s in samp_list:
        g, n = s.split('-')
        if g not in group_list:
            group_list.append(g)


    # make empty lists
    names = []
    num_samples = []
    num_cycles = []
    max_avg = []
    max_std = []
    kd_avg = []
    kd_std = []
    seed_avg = []
    seed_std = []
    ssd_avg = []
    ssd_std = []
    fit_avg = []
    fit_std = []
    grp_fit = []

    for group in group_list:

        # find the group's samples
        df0 = csvDF.filter( regex=str(group+'-') )

        # get sample names and count
        sampZ = list(df0)
        num_samples.append(len(sampZ))
        nameZ = '_'.join(sampZ)
        names.append(nameZ)

        # set variables for profile fit
#        prevX=[]
#        for column in df0:
#            prevX = prevX + ( df0[column].values.tolist()[:-1] )

#        yieldY=[]
#        for column in df0:
#            yieldY = yieldY + ( df0[column].values.tolist()[1:] )

        # fit the datums
#        maxZ, KdZ, seedZ, SSDZ = qpcr_fit(prevX, yieldY, weight_opt)

        # get the summary data
#        max_avg.append(maxZ)
#        kd_avg.append(KdZ)
#        seed_avg.append(seedZ)
        #ssd_avg.append(SSDZ)


        # get the group stats
        df = statsDF.loc[statsDF['Sample'].str.contains(str(group+'-'))]
        max_avg.append(df['Max'].describe()['mean'])
        max_std.append(df['Max'].describe()['std'])
        kd_avg.append(df['Kd'].describe()['mean'])
        kd_std.append(df['Kd'].describe()['std'])
        seed_avg.append(df['Seed'].describe()['mean'])
        seed_std.append(df['Seed'].describe()['std'])
        ssd_avg.append(df['SSD'].describe()['mean'])
        ssd_std.append(df['SSD'].describe()['std'])

        # generate group fit
        grp_fit.append( calcYield(
                df['Seed'].describe()['mean'],
                df['Max'].describe()['mean'],
                df['Kd'].describe()['mean'],
                num_cycs-1) )

    # organize dataframe for group_stats.csv
    statsDFgrouped = pd.DataFrame()
    statsDFgrouped['Group'] = group_list
    statsDFgrouped['Samples'] = names
    statsDFgrouped['n'] = num_samples
    statsDFgrouped['Max_avg'] = max_avg
    statsDFgrouped['Max_std'] = max_std
    statsDFgrouped['Kd_avg'] = kd_avg
    statsDFgrouped['Kd_std'] = kd_std
    statsDFgrouped['Seed_avg'] = seed_avg
    statsDFgrouped['Seed_std'] = seed_std
    statsDFgrouped['SSD_avg'] = ssd_avg
    statsDFgrouped['SSD_std'] = ssd_std


    # organize dataframe for group_fit.csv
    fitDFgrouped=pd.DataFrame()
    fitDFgrouped['Cycle']=range(1,num_cycs+1)
    for (groupZ, yieldZ) in zip(group_list, grp_fit):
        fitDFgrouped[groupZ]=yieldZ

    return statsDFgrouped, fitDFgrouped









def makePlot(rawDF, fitDF, fitDFgrouped, title0):

# get a list of groups
    group_list=list(fitDFgrouped)[1:]

    # make a colormap to iterate through
    cmap = cm.get_cmap( 'jet', len(group_list) )
    color_list=[]
    for i in range(cmap.N):
        rgb = cmap(i)[:3]
        color_list.append( matplotlib.colors.rgb2hex(rgb) )


    for group,col in zip(group_list,color_list):

        # find the group's samples
        smpDF = rawDF.filter(regex=group)
        if 'Cycle' in list(smpDF):
            smpDF=smpDF.drop(columns=['Cycle'])

        # plot the sample points
        x=rawDF['Cycle']
        y=smpDF
        plt.plot(x, y, 'o', color=col, markersize=2)

        # calculate Y error from sample fit
        yDF = fitDF.filter(regex=group)
        if 'Cycle' in list(yDF):
            yDF=yDF.drop(columns=['Cycle'])
        y_err=[]
        for row in yDF.iterrows():
            err = np.std( row[1].tolist() )
            y_err.append(err)

        # find the group stats
        grpDF = pd.DataFrame()
        grpDF[group] = fitDFgrouped[group]
        grpDF['Cycle'] = fitDFgrouped['Cycle']
        grpDF = grpDF.reindex( columns=['Cycle']+grpDF.columns[:-1].tolist() )

        # plot the group fit line
        x=grpDF['Cycle'].values
        y=grpDF.drop(columns = ['Cycle']).values
        plt.errorbar( x, y, yerr=y_err, color=col, label=group,
            linewidth=1, elinewidth=0.5, capsize=5, capthick=1 )

    plt.legend(title='Groups', loc=2, fontsize='small')

    plt.title(title0)

    plt.xlabel('Cycle')
    plt.ylabel('Fluorescence')

    plt.savefig(str(title0+'.png'))
    plt.savefig(str(title0+'.pdf'))

    plt.show()













#
