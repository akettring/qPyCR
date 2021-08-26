# qPyCR fitter for Python 2.7

import pandas as pd

# custom pPCR functions
import qPyCR_functions as qp

from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument('-f', '--file', dest='filename',
                    help='--file=<your-file.csv>')

parser.add_argument('-i', '--in_dir', dest='indir',
                    help='directory for incoming files, default = ./infiles/',
                    default='./infiles/')

parser.add_argument('-o', '--out_dir', dest='outdir',
                    help='directory for outgoing files, default = ./outfiles/',
                    default='./outfiles/')

parser.add_argument('-n', '--norm', dest='normalize',
                    help='limit (default), max, global',
                    default='limit')

parser.add_argument('-w', '--weight', dest='weight',
                    help='if True, use weighting function in fitting',
                    default='False')

parser.add_argument('-c', '--cycles', dest='cycles',
                    help='Number of cycles for fit',
                    default='from_input')

args = parser.parse_args()



inFile=args.filename
inDir=args.indir
outDir=args.outdir
norm_opt=args.normalize
weight_opt=args.weight
cyc_opt=args.cycles



# read the input csv file to dataframe
csvDF = pd.read_csv(inDir+inFile)


# get number of cycles
if cyc_opt=='from_input':
    num_cycs = len( csvDF['Cycle'].tolist() )
else:
    num_cycs = int(cyc_opt)




# generate raw stats
statsDF = qp.get_stats(csvDF, weight_opt)
statsDF.to_csv(outDir+inFile+'-raw_stats.csv', index=False)


# generate raw_fit
fitDF = qp.get_fit(statsDF, num_cycs)
fitDF.to_csv(outDir+inFile+'-raw_fit.csv', index=False)






# generate normalization values
max_list=csvDF.drop(columns=['Cycle']).max().values
if norm_opt=='max':
        norm_list=max_list

elif norm_opt=='global':
    max_global = max(max_list)
    norm_list = [max_global for i in max_list]

elif norm_opt=='limit':
    # find limits
    norm_list=[]
    for index,row in statsDF.iterrows():

        # get the variables for fitting
        sampY = row['Sample']
        maxY = row['Max']
        KdY = row['Kd']
        seedY = row['Seed']

        lim = max( qp.calcYield(seedY, maxY, KdY, num_cycs*10) )
        norm_list.append(lim)






# normalize the original DF
counter=0
normDF=pd.DataFrame()
normDF['Cycle']=csvDF['Cycle'].values
for column in csvDF.drop(columns=['Cycle']):
    raw_vals = csvDF[column].values
    normY = norm_list[counter]
    norm_vals = [ (val/normY) for val in raw_vals ]
    counter += 1
    normDF[column]=norm_vals





# repeat the fitting with normalized DF

# generate norm stats
norm_statsDF = qp.get_stats(normDF, weight_opt)
norm_statsDF.to_csv(outDir+inFile+'-norm_stats.csv', index=False)

# generate norm_fit
norm_fitDF = qp.get_fit(norm_statsDF, num_cycs)
norm_fitDF.to_csv(outDir+inFile+'-norm_fit.csv', index=False)










# group raw fit
statsDFgrouped, fitDFgrouped = qp.groupSamples(csvDF, statsDF, weight_opt, num_cycs)
statsDFgrouped.to_csv(outDir+inFile+'-raw_stats_group.csv', index=False)
fitDFgrouped.to_csv(outDir+inFile+'-raw_fit_group.csv', index=False)

# group norm fit
norm_statsDFgrouped, norm_fitDFgrouped = qp.groupSamples(normDF, norm_statsDF, weight_opt, num_cycs)
norm_statsDFgrouped.to_csv(outDir+inFile+'-norm_stats_group.csv', index=False)
norm_fitDFgrouped.to_csv(outDir+inFile+'-norm_fit_group.csv', index=False)






# generate the raw fit plot
title1=str(outDir+inFile+'_raw')
qp.makePlot(csvDF, fitDF, fitDFgrouped, title1)

# generate the normalized plots
title2=str(outDir+inFile+'_norm_'+norm_opt)
qp.makePlot(normDF, norm_fitDF, norm_fitDFgrouped, title2)








#
