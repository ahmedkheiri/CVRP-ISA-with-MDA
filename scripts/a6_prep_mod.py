from IS_class import InstanceSpace
import pandas as pd
import pickle as pkl
import sys
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('path', type=str)
parser.add_argument('idx', type=int)
parser.add_argument('mpath', type=str)
parser.add_argument('stage', type=int)
parser.add_argument('-do', type=str)

args = parser.parse_args()
param_path = args.path
param_idx = args.idx - 1
mpath = args.mpath
stage = args.stage

if args.do is not None:
    do_proj = args.do.split(',')
else:
    do_proj = []



params = pd.read_csv(param_path)
params['maxPerf'] = params['maxPerf'].astype(bool)
params['absPerf'] = params['absPerf'].astype(bool)

print(params.loc[param_idx,'dataname'],
      params.loc[param_idx,'infolder'],
        params.loc[param_idx,'outfolder'],
        params.loc[param_idx,'maxPerf'],
        params.loc[param_idx,'absPerf'],
        params.loc[param_idx,'epsilon'])


def prepNewMetadata(dataname,inpath,outpath,maxPerf,absPerf,perfThresh, sel):    
    
    # check if the output directory exists
    if not os.path.exists(outpath):
        print('Output directory does not exist. Exiting.')
        return
    
    if not os.path.exists(outpath+mpath):
        os.makedirs(outpath+mpath)
    # check if output directory is empty
    if len(os.listdir(outpath+mpath)) > 0:
        # error    
        print('Mod output directory is not empty. Exiting.')
        return 
    
    meta_train = pd.read_csv(inpath + dataname + '_train.csv', index_col=0)
    meta_test = pd.read_csv(inpath + dataname + '_test.csv', index_col=0)    

    fm = {'2f':'full', '2s':'short'}
    modName = dataname.split('_')[0] + '_' + fm[dataname.split('_')[1]] + '-mod.csv'
    allModded = pd.read_csv(f'{inpath}modded/{modName}', index_col=0)

    instModList = pd.read_csv(f'{outpath}instance mod keep.csv')
    instModList = instModList.loc[instModList['keep2'],'instances']

    mod_testList = pd.read_csv(f'{outpath}instance mod test.csv')['instances']

    # mod_trainList = instModList.loc[instModList['og'].isin(meta_train.index)].loc[
    #     instModList['keep'],'instance']
    # mod_testList = instModList.loc[instModList['og'].isin(meta_test.index)].loc[
    #     instModList['keep'],'instance']
    
    mod_meta_train = allModded.loc[instModList]
    mod_meta_test = allModded.loc[mod_testList]

    full_train  =  pd.concat([meta_train, meta_test, mod_meta_train])    

    if sel:
        selected_features = list(pd.read_csv(
                outpath + 'selected_features.csv', index_col=0).iloc[0])
        
        full_train = full_train[
            [col for col in full_train.columns if not (col.startswith('feature_')) ] + selected_features]
        
        mod_meta_test = mod_meta_test[
            [col for col in full_train.columns if not (col.startswith('feature_')) ] + selected_features]
        
    iSpace = InstanceSpace()
    iSpace.fromMetadata(pd.concat([full_train,mod_meta_test]), scaler='s',best='best',source='source')
    iSpace.getRelativePerf(min= not maxPerf)
    iSpace.splitData_known(full_train.index,mod_meta_test.index, scale=True)
    
    ### same as a1_prelim.py
    binPerf = iSpace.getBinaryPerf(abs=absPerf, min= not maxPerf, thr=perfThresh)
    alg_list = [a.removeprefix('algo_') for a in iSpace.algorithms]

    # write processed data to csv
    train_out = pd.concat([
        pd.DataFrame(iSpace.split_data['Y_train'], index = full_train.index, 
                     columns = iSpace.algorithms),
        binPerf.loc[full_train.index,:].astype(int),
        pd.DataFrame(iSpace.split_data['X_train'], index = full_train.index, 
                     columns = iSpace.featureNames)    
    ], axis=1)

    train_out.insert(0, 'source', full_train['source'])
    train_out.insert(0, 'best', full_train['best'].apply(lambda x: alg_list.index(x)+1))

    test_out = pd.concat([
        pd.DataFrame(iSpace.split_data['Y_test'], index = mod_meta_test.index, columns = iSpace.algorithms),
        binPerf.loc[mod_meta_test.index,:].astype(int),
        pd.DataFrame(iSpace.split_data['X_test'], index = mod_meta_test.index, columns = iSpace.featureNames)    
    ], axis=1)

    test_out.insert(0, 'source', mod_meta_test['source'])
    test_out.insert(0, 'best', mod_meta_test['best'].apply(lambda x: alg_list.index(x)+1))

    train_out.to_csv(outpath + f'{mpath}processed_train.csv')
    test_out.to_csv(outpath + f'{mpath}processed_test.csv')

    ## save the instance space to pkl
    with open(f'{outpath}{mpath}instanceSpace_prelim.pkl', 'wb') as f:
        pkl.dump(iSpace, f)


if __name__ == '__main__':

    if stage == 1:
        # just prelim
        prepNewMetadata(params.loc[param_idx,'dataname'],
            params.loc[param_idx,'infolder'],
            params.loc[param_idx,'outfolder'],
            params.loc[param_idx,'maxPerf'],
            params.loc[param_idx,'absPerf'],
            params.loc[param_idx,'epsilon'],
            False)
    
    
    
