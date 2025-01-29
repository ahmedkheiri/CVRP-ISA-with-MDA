from IS_class import InstanceSpace
import numpy as np
import pandas as pd
import sys

import seaborn as sns
import matplotlib.pyplot as plt

from scipy.spatial import distance
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('path', type=str)
parser.add_argument('idx', type=int)
parser.add_argument('pr', type=str)
parser.add_argument('-ns','--nosifted', action='store_true')
parser.add_argument('-mda','--mda', action='store_true')

args = parser.parse_args()

param_path = args.path
param_idx = args.idx - 1
pr = args.pr
from_mda = args.mda
no_sifted = args.nosifted


params = pd.read_csv(param_path)
params['maxPerf'] = params['maxPerf'].astype(bool)
params['absPerf'] = params['absPerf'].astype(bool)

### all modded instances =======================================

all_mod_features = pd.read_csv('~/xx_path/metadata/modded/modded_features_full.csv',
                               index_col=0)
all_mod_features.index.name = 'instance'
all_mod_features.rename(columns={c:'feature_'+c for c in all_mod_features.columns},inplace=True)


all_mod_features['og_instance'] = pd.Series(all_mod_features.index
                ).apply(lambda x: x.removeprefix('mod_').split('_m')[0]).values
###==============================================================

def modify_instance_listMDA(outpath, mc, proj_file, proj_mod_file):
    
    metadata = pd.read_csv(outpath + 'processed_full.csv', index_col=0)
    proj_init = pd.read_csv(f'{outpath}{proj_file}', index_col='instances')

    # calculate distance
    proj_dist = distance.squareform(distance.pdist(
        proj_init[['Z1','Z2']], 'euclidean'))

    # nearest neighbour with different Best
    nn = proj_dist.argsort()[:,1]
    nn_dist = pd.Series(proj_dist[np.arange(len(proj_dist)),nn],
                        index=proj_init.index)

    diffNN = np.where(metadata['best'] != metadata['best'].iloc[nn].values)

    # number of neighbours within distance
    dl = np.quantile(nn_dist.values,0.9)  #np.quantile(proj_dist,0.02)   
    neigh = np.sum(proj_dist < dl, axis=1)
    lonely = np.where(neigh < mc)

    # instances to modify
    #modList = iSpace.performance.index[np.union1d(diffNN,lonely)]
    modList = pd.DataFrame(
        {'instances': list(proj_init.index[lonely]) + list(proj_init.index[diffNN]),
            'type': ['dis']*len(lonely[0]) + ['visa']*len(diffNN[0])})
    # remove duplicates rows in instances
    modList = modList.drop_duplicates(subset='instances')     

    all_mod_proj = pd.read_csv(f'{outpath}{proj_mod_file}', index_col=0)
    all_mod_proj['og'] = pd.Series(all_mod_proj.index
                ).apply(lambda x: x.removeprefix('mod_').split('_m')[0]).values  
    
    # join with modList
    typeList = all_mod_proj.merge(modList,left_on='og',right_on='instances',how='left')['type']
    all_mod_proj['type'] = typeList.values
    
            
    projMod = all_mod_proj[all_mod_proj['og'].isin(modList['instances'])]
     
    projMod['keep'] = projMod.apply(lambda x: distance.euclidean(x[['Z1','Z2']].values,
                    proj_init.loc[x['og'],['Z1','Z2']].values) >= dl,
                    axis=1)
    
    # distance matrix for all
    dist_all = distance.squareform(distance.pdist(
        pd.concat([proj_init[['Z1','Z2']],projMod[['Z1','Z2']]]), 'euclidean'))
    projMod['nn_dist'] = dist_all[len(proj_init):, :len(proj_init)].min(axis=1)
    projMod['keep2'] = projMod['nn_dist'] >= dl
    projMod['keep2'] = projMod.apply(lambda x: x['keep'] if x['type']=='dis' else x['keep2'],axis=1)

    print('keep2',projMod['keep2'].sum())
    print('keep',projMod['keep'].sum())


    pltData = proj_init.copy()
    pltData['mod'] = [x in modList['instances'] for x in proj_init.index]
    sns.scatterplot(data=pltData,x='Z1',y='Z2',hue='mod', alpha=0.6)
    sns.scatterplot(data=projMod.loc[~projMod['keep']],x='Z1',y='Z2',alpha=0.3,
                color='black',marker='x',s=60)
    sns.scatterplot(data=projMod.loc[projMod['keep']],x='Z1',y='Z2',alpha=0.5,
                color='red',marker='x')
    plt.title(outpath.split('/')[-2] + f'- mod {len(modList)} - new {len(projMod)}')
    plt.savefig(f'{outpath}instance modList.png')

    # modList.sort_index().to_csv(f'{outpath}instance modList.csv',index=False)
    projMod[['og','type','keep','keep2']].sort_values('og'
            ).to_csv(f'{outpath}instance mod keep.csv')   



mc = 3 # min cluster size

if from_mda:
    modify_instance_listMDA(params.loc[param_idx,'outfolder'],
                            mc,
                            f'{pr}_proj_full.csv',
                            f'modded_{pr}_proj.csv')
# elif no_sifted:
#    pass
# else:
#    pass
