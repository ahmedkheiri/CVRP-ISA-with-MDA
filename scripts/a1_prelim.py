from IS_class import InstanceSpace
import pandas as pd
import pickle as pkl
import sys
import os

param_path = sys.argv[1]
param_idx = int(sys.argv[2]) - 1


params = pd.read_csv(param_path)
params['maxPerf'] = params['maxPerf'].astype(bool)
params['absPerf'] = params['absPerf'].astype(bool)

print(params.loc[param_idx,'dataname'],
      params.loc[param_idx,'infolder'],
        params.loc[param_idx,'outfolder'],
        params.loc[param_idx,'maxPerf'],
        params.loc[param_idx,'absPerf'],
        params.loc[param_idx,'epsilon'])


def runPrelim(dataname,inpath,outpath,
              maxPerf,absPerf,perfThresh):
    
    # check if the output directory exists
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    # check if output directory is empty
    if len(os.listdir(outpath)) > 0:
        # error    
        print('Output directory is not empty. Exiting.')
        return

    # read the data
    meta_train = pd.read_csv(inpath + dataname + '_train.csv', index_col=0)
    meta_test = pd.read_csv(inpath + dataname + '_test.csv', index_col=0)

    metadata = pd.concat([meta_train, meta_test])

    for split_ in [True, False]:
        iSpace = InstanceSpace()
        iSpace.fromMetadata(metadata, scaler='s',best='best',source='source')
        iSpace.getRelativePerf(min= not maxPerf)
        
        if split_:
            iSpace.splitData_known(meta_train.index, meta_test.index, scale=True)

        binPerf = iSpace.getBinaryPerf(abs=absPerf, min= not maxPerf, thr=perfThresh)
        alg_list = [a.removeprefix('algo_') for a in iSpace.algorithms]

        # write processed data to csv
        if split_:
            train_out = pd.concat([
                pd.DataFrame(iSpace.split_data['Y_train'], index = meta_train.index, columns = iSpace.algorithms),
                binPerf.loc[meta_train.index,:].astype(int),
                pd.DataFrame(iSpace.split_data['X_train'], index = meta_train.index, columns = iSpace.featureNames)    
            ], axis=1)

            train_out.insert(0, 'source', meta_train['source'])
            train_out.insert(0, 'best', meta_train['best'].apply(lambda x: alg_list.index(x)+1))

            test_out = pd.concat([
                pd.DataFrame(iSpace.split_data['Y_test'], index = meta_test.index, columns = iSpace.algorithms),
                binPerf.loc[meta_test.index,:].astype(int),
                pd.DataFrame(iSpace.split_data['X_test'], index = meta_test.index, columns = iSpace.featureNames)    
            ], axis=1)

            test_out.insert(0, 'source', meta_test['source'])
            test_out.insert(0, 'best', meta_test['best'].apply(lambda x: alg_list.index(x)+1))

            train_out.to_csv(outpath + 'processed_train.csv')
            test_out.to_csv(outpath + 'processed_test.csv')

        ## save the instance space to pkl
            with open(outpath + 'instanceSpace_prelim.pkl', 'wb') as f:
                pkl.dump(iSpace, f)

        else:
            full_out = pd.concat([
                pd.DataFrame(iSpace.Y_s, index = metadata.index, columns = iSpace.algorithms),
                binPerf.loc[metadata.index,:].astype(int),
                pd.DataFrame(iSpace.X_s, index = metadata.index, columns = iSpace.featureNames)    
            ], axis=1)

            full_out.insert(0, 'source', metadata['source'])
            full_out.insert(0, 'best', metadata['best'].apply(lambda x: alg_list.index(x)+1))

            full_out.to_csv(outpath + 'processed_full.csv')

            with open(outpath + 'instanceSpace_prelimF.pkl', 'wb') as f:
                pkl.dump(iSpace, f)

runPrelim(params.loc[param_idx,'dataname'],
        params.loc[param_idx,'infolder'],
        params.loc[param_idx,'outfolder'],
        params.loc[param_idx,'maxPerf'],
        params.loc[param_idx,'absPerf'],
        params.loc[param_idx,'epsilon'])
