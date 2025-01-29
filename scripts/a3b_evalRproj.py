import predictions
import pickle as pkl
import pandas as pd
import numpy as np
import sys
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('path', type=str)
parser.add_argument('idx', type=int)
parser.add_argument('proj', type=str)
parser.add_argument('proj_file', type=str)
parser.add_argument('-fol', type=str, default='')

args = parser.parse_args()

param_path = args.path
param_idx = args.idx - 1
proj = args.proj
proj_file = args.proj_file
fol = args.fol


params = pd.read_csv(param_path)

def genRegret(folder,method,best='best1' ):

    Ip = pd.read_pickle(f'{folder}instanceSpace_prelim.pkl')

    if method == 'MDA':
        preds = pd.read_csv(f'{folder}mda_proj.csv')
    else:
        preds = pd.read_csv(f'{folder}pythia_preds.csv')

    preds = preds.loc[preds['proj']==method]
    preds = preds.set_index('instances')

    ind = {t: list(Ip.split_data[f'Yb_{t}'].index) for t in ['train','test']}
    cols = [a.split('_')[1] for a in Ip.algorithms]

    Yr = {t: pd.DataFrame(Ip.split_data[f'Yr_{t}'], columns = cols, index = ind[t]) for t in ['train','test']}
    Yr = pd.concat(Yr, names=['group','instances']).reset_index()
    Yr = Yr.set_index('instances')
    Yr['bestP'] = preds[best].apply(lambda x: 'hgs' if x==1 else 'filo')
    Yr['bestA'] = pd.concat([Ip.split_data['Yb_train'],Ip.split_data['Yb_test']])

    Yr['reg'] = Yr.apply(lambda x: x['hgs'] if x['bestP']=='hgs' else x['filo'], axis=1)
    Yr['reg_sub'] = Yr.apply(lambda x: x['reg'] if x['bestA']!=x['bestP'] else np.nan, axis=1)

    return Yr

def genRegret_abs(folder,method,min,best='best1' ):

    Ip = pd.read_pickle(f'{folder}instanceSpace_prelim.pkl')

    if min:
        Ydiff = Ip.performance[Ip.algorithms].apply(
            lambda row: row - row.min(), axis=1
        ).fillna(0)#.values
    else:
        Ydiff = Ip.performance[Ip.algorithms].apply(
            lambda row: row - row.max(), axis=1
        ).fillna(0)#.values

    if method == 'MDA':
        preds = pd.read_csv(f'{folder}mda_proj.csv')
    else:
        preds = pd.read_csv(f'{folder}pythia_preds.csv')
    
    preds = preds.loc[preds['proj']==method]
    preds = preds.set_index('instances')

    ind = {t: list(Ip.split_data[f'Yb_{t}'].index) for t in ['train','test']}
    cols = [a.split('_')[1] for a in Ip.algorithms]

    Yr = {t: pd.DataFrame(Ydiff.loc[ind[t]].values, columns = cols, index = ind[t]) for t in ['train','test']}
    Yr = pd.concat(Yr, names=['group','instances']).reset_index()
    Yr = Yr.set_index('instances')
    Yr['bestP'] = preds[best].apply(lambda x: 'hgs' if x==1 else 'filo')
    Yr['bestA'] = pd.concat([Ip.split_data['Yb_train'],Ip.split_data['Yb_test']])

    Yr['abs_reg'] = Yr.apply(lambda x: x['hgs'] if x['bestP']=='hgs' else x['filo'], axis=1)
    Yr['abs_reg_sub'] = Yr.apply(lambda x: x['abs_reg'] if x['bestA']!=x['bestP'] else np.nan, axis=1)

    return Yr


def eval_predictions(outpath, proj_file, maxPerf, absPerf, perfThresh):
    iSpace = pd.read_pickle(outpath + 'instanceSpace_prelim.pkl')

    alg_labels = [a.removeprefix('algo_') for a in iSpace.algorithms]

    Ybin_full = iSpace.getBinaryPerf(abs=absPerf, min= not maxPerf, thr=perfThresh)
    bin_train = Ybin_full.loc[iSpace.split_data['Yb_train'].index].values
    bin_test = Ybin_full.loc[iSpace.split_data['Yb_test'].index].values
        
    Ybest_train = iSpace.split_data['Yb_train'].apply(lambda x: alg_labels.index(x)+1)
    Ybest_test = iSpace.split_data['Yb_test'].apply(lambda x: alg_labels.index(x)+1)

    preds = pd.read_csv(outpath + proj_file)
    best_labels = [c for c in preds.columns if c.startswith('best')]

    preds_train = preds.loc[preds['group']=='train']
    preds_test = preds.loc[preds['group']=='test']

    stats_out = pd.concat([predictions.summary_statsM(alg_labels, bin_train, preds_train[alg_labels].values,
                            best_labels, Ybest_train, preds_train[best_labels].values, 'train'),
                        predictions.summary_statsM(alg_labels, bin_test, preds_test[alg_labels].values,
                            best_labels, Ybest_test, preds_test[best_labels].values, 'test')
    ])

    # out_fol = proj_file.split('/')[0]
    stats_out.to_csv(f'{outpath}{proj.lower()}_stats.csv',index=False)

    Yr = genRegret(outpath,proj,'bestS').join(
        genRegret_abs(outpath,proj,True,'bestS')[['abs_reg','abs_reg_sub']])

    Yr.to_csv(f'{outpath}regret.csv')

eval_predictions(params.loc[param_idx,'outfolder']+fol, 
                 proj_file, 
                 params.loc[param_idx,'maxPerf'], 
                 params.loc[param_idx,'absPerf'], 
                 params.loc[param_idx,'epsilon'])
