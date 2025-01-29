import pandas as pd
import numpy as np
import pickle as pkl
import os

from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import decomposition, cross_decomposition
from scipy.spatial import distance
from scipy import stats
from scipy.linalg import eig

from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score, cross_validate, train_test_split


import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('mode.chained_assignment',None)

    
    
class InstanceSpace():
    
    def __init__(self):
        
        # attributes
        self.features = pd.DataFrame()
        self.performance = pd.DataFrame()
        self.featureNames = []
        self.algorithms = []
        self.source = []

        self.n, self.m, self.a = 0, 0, 0 

        # standardized arrays
        self.X_s = []
        self.Y_s = []

        # centered arrays
        # self.X_c = []
        # self.Y_c = []
        

        # projection spaces
        self.PlotProj = {} # projection of data points - 2D
        self.projections = {}   # projection objects
        # self.proj = {}         # projection component names

        # training-test split
        self.split_data = {}
        self.Y_rel = []

        # eval
        self.eval = {}
        self.footprints = {}

        # scaler
        self.scaler = None
        self.x_scaler = None
        self.y_scaler = None

    def fromMetadata(self, metadata, prefixes=['feature_','algo_'], scaler='s', best=None, source=None):
        self.featureNames = [x for x in metadata.columns if x.startswith(prefixes[0])]
        self.algorithms = [x for x in metadata.columns if x.startswith(prefixes[1])]

        self.features = metadata[self.featureNames]
        self.performance = metadata[self.algorithms]

        if best is not None and best in metadata.columns:
            self.performance['Best'] = metadata[best]
        if source is not None and source in metadata.columns:
            self.source = metadata[source]

        # put instance name in PlotProj
        #self.PlotProj['instance'] = metadata.index
        #self.PlotProj.set_index(metadata.index,inplace=True)

        self.n, self.m = self.features.shape
        self.a = len(self.algorithms)

        self.scaler = scaler
        if scaler == 's':
            # self.scaler = StandardScaler().fit
            self.x_scaler = StandardScaler().fit(self.features.values)
            self.y_scaler = StandardScaler().fit(self.performance[self.algorithms].values.reshape(-1,1))            
            
        elif scaler == 'r5':
            # self.scaler = RobustScaler(quantile_range=(5.0,95.0),unit_variance=True).fit
            self.x_scaler = RobustScaler(quantile_range=(5.0,95.0),unit_variance=True).fit(self.features.values)
            self.y_scaler = RobustScaler(quantile_range=(5.0,95.0),unit_variance=True
                                         ).fit(self.performance[self.algorithms].values.reshape(-1,1))            
        
        elif scaler == 'r25':
            # self.scaler = RobustScaler(quantile_range=(25.0,75.0),unit_variance=True).fit
            self.x_scaler = RobustScaler(quantile_range=(25.0,75.0),unit_variance=True).fit(self.features.values)
            self.y_scaler = RobustScaler(quantile_range=(25.0,75.0),unit_variance=True
                                         ).fit(self.performance[self.algorithms].values.reshape(-1,1))

        elif scaler == 'p':
            # self.scaler = PowerTransformer().fit
            # minX = self.features.min().apply(lambda x: x + 1 if x <= 0 else 0)
            # minY = self.performance.min().apply(lambda x: x + 1 if x <= 0 else 0)
            self.x_scaler = PowerTransformer().fit(self.features.values)
            self.y_scaler = PowerTransformer().fit(self.performance[self.algorithms].values.reshape(-1,1))
            # self.X_s = PowerTransformer().fit_transform(self.features.values) # + minX)
            # self.Y_s = PowerTransformer().fit_transform(self.performance[self.algorithms].values) # + minY)
        
        if self.x_scaler is not None:
            # self.x_scaler = self.scaler(self.features.values)
            # self.y_scaler = self.scaler(self.performance[self.algorithms].values)

            self.X_s = self.x_scaler.transform(self.features.values)
            self.Y_s = np.apply_along_axis(
                lambda x: self.y_scaler.transform(x.reshape(-1,1)), 0, self.performance[self.algorithms].values).reshape(-1,self.a)
            
        
        # self.X_c = StandardScaler(with_std=False).fit_transform(self.features.values)
        # self.Y_c = StandardScaler(with_std=False).fit_transform(self.performance[self.algorithms].values)

    def getSource(self, source):
        if callable(source):
            self.source = source(self.features)
            self.PlotProj['Source'] = self.source
            print('source of data available')

        elif len(source) == self.n:
            self.source = source
            self.PlotProj['Source'] = self.source
            print('source of data available')

        else:
            print('Cannot assign source. Check the length of the list or the function')

    def getBest(self, best, axis=1):
        """Actual best performance based on performance metrics

        Args:
            expr: function with rule for best algorithm or a list type with the same length as the number of instances 
        """
        if callable(best):
            self.performance['Best'] = self.performance.apply(best,axis=axis)
            # self.PlotProj['Best'] = self.performance['Best']#.values
            print('classification data available')

        elif len(best) == self.n:
            self.performance['Best'] = best
            # self.PlotProj['Best'] = self.performance['Best']#.values
            print('classification data available')

        else:
            print('Cannot assign best algorithm. Check the length of the list or the function')
    
    def getBinaryPerf(self, abs, min, thr, pref='algo_'):
        if abs:
            if min:
                binPerf = self.performance[self.algorithms].apply(
                    lambda row: row <= thr, axis=1)
            else:
                binPerf = self.performance[self.algorithms].apply(
                    lambda row: row >= thr, axis=1)
        else:
            if min:
                binPerf = self.Y_rel.apply(
                    lambda row: row <= thr, axis=1)
            else:
                binPerf = self.Y_rel.apply(
                    lambda row: row >= thr, axis=1)
                
        # rename columns
        binPerf.rename(columns = {c: 'bin_'+c.removeprefix(pref) for c in binPerf.columns}, inplace=True)
        return binPerf

    def getRelativePerf(self, min):

        if min:
            self.Y_rel = self.performance[self.algorithms].apply(
                lambda row: row if row.min()==0 else (row - row.min())/row.min(), axis=1
            ).fillna(0)#.values
        else:
            self.Y_rel = self.performance[self.algorithms].apply(
                lambda row: row if row.max()==0 else (row - row.max())/row.max(), axis=1
            ).fillna(0)#.values       
        
        print(f'Relative performance data available')

    def splitData(self, test_size, random_state, scale, stratified = True):
        """Split data into training and test sets.

        Args:
            test_size (float): proportion of data to be in test set
            random_state (int): seed for random number generator
            scale (bool): whether to scale the data
            stratified (bool, optional): whether to stratify the split. Defaults to True.
        """


        self.split_data = dict(zip(
            ['X_train', 'X_test', 'Y_train', 'Y_test', 'Yb_train', 'Yb_test'],
            
            train_test_split(self.features.values, self.performance[self.algorithms].values, 
                             self.performance['Best'],
                             test_size=test_size, random_state=random_state, 
                             stratify= self.performance['Best'] if stratified else None)
    
        ))

        if scale:
            # self.x_scaler = self.scaler(self.split_data['X_train'])
            # self.y_scaler = self.scaler(self.split_data['Y_train'])

            if self.scaler == 's':
                self.x_scaler = StandardScaler().fit(self.split_data['X_train'])
                self.y_scaler = StandardScaler().fit(self.split_data['Y_train'].reshape(-1,1))
            elif self.scaler == 'r5':
                self.x_scaler = RobustScaler(quantile_range=(5.0,95.0),unit_variance=True).fit(self.split_data['X_train'])
                self.y_scaler = RobustScaler(quantile_range=(5.0,95.0),unit_variance=True).fit(
                    self.split_data['Y_train'].reshape(-1,1))
            elif self.scaler == 'r25':
                self.x_scaler = RobustScaler(quantile_range=(25.0,75.0),unit_variance=True).fit(self.split_data['X_train'])
                self.y_scaler = RobustScaler(quantile_range=(25.0,75.0),unit_variance=True
                                             ).fit(self.split_data['Y_train'].reshape(-1,1))
            elif self.scaler == 'p':
                # minX = self.features.min().apply(lambda x: x + 1 if x <= 0 else 0)
                # minY = self.performance.min().apply(lambda x: x + 1 if x <= 0 else 0)
                self.x_scaler = PowerTransformer().fit(self.split_data['X_train'])
                self.y_scaler = PowerTransformer().fit(self.split_data['Y_train'].reshape(-1,1))
                        
            self.split_data['X_train'] = self.x_scaler.transform(self.split_data['X_train'])
            self.split_data['X_test'] = self.x_scaler.transform(self.split_data['X_test'])
            self.split_data['Y_train'] = np.apply_along_axis(
                lambda x: self.y_scaler.transform(x.reshape(-1,1)), 0, self.split_data['Y_train']).reshape(-1,self.a)
            self.split_data['Y_test'] = np.apply_along_axis(
                lambda x: self.y_scaler.transform(x.reshape(-1,1)), 0, self.split_data['Y_test']).reshape(-1,self.a)
        
        else:
            self.split_data['X_train'] = self.split_data['X_train'].astype(float)
            self.split_data['X_test'] = self.split_data['X_test'].astype(float)

        if len(self.Y_rel) > 0:
            self.split_data['Yr_train'], self.split_data['Yr_test'] = train_test_split(
                self.Y_rel.values, test_size=test_size, random_state=random_state, 
                stratify= self.performance['Best'] if stratified else None
            )

        ps = 'Scaled' if scale else 'Unscaled'
        print(f'{ps} data split into training (size {len(self.split_data["X_train"])}) and test (size {len(self.split_data["X_test"])}) sets, stratified: {stratified}')
   
    def splitData_known(self, train_ind, test_ind, scale):

        self.split_data = {
            'X_train': self.features.loc[train_ind,:].values,
            'X_test': self.features.loc[test_ind,:].values,
            'Y_train': self.performance.loc[train_ind,self.algorithms].values,
            'Y_test': self.performance.loc[test_ind,self.algorithms].values,
            'Yb_train': self.performance.loc[train_ind,'Best'],
            'Yb_test': self.performance.loc[test_ind,'Best']
        }


        if scale:
            # self.x_scaler = self.scaler(self.split_data['X_train'])
            # self.y_scaler = self.scaler(self.split_data['Y_train'])
            if self.scaler == 's':
                self.x_scaler = StandardScaler().fit(self.split_data['X_train'])
                self.y_scaler = StandardScaler().fit(self.split_data['Y_train'].reshape(-1,1))
            elif self.scaler == 'r5':
                self.x_scaler = RobustScaler(quantile_range=(5.0,95.0),unit_variance=True).fit(self.split_data['X_train'])
                self.y_scaler = RobustScaler(quantile_range=(5.0,95.0),unit_variance=True
                                             ).fit(self.split_data['Y_train'].reshape(-1,1))
            elif self.scaler == 'r25':
                self.x_scaler = RobustScaler(quantile_range=(25.0,75.0),unit_variance=True).fit(self.split_data['X_train'])
                self.y_scaler = RobustScaler(quantile_range=(25.0,75.0),unit_variance=True
                                             ).fit(self.split_data['Y_train'].reshape(-1,1))
            elif self.scaler == 'p':
                # minX = self.features.min().apply(lambda x: x + 1 if x <= 0 else 0)
                # minY = self.performance.min().apply(lambda x: x + 1 if x <= 0 else 0)
                self.x_scaler = PowerTransformer().fit(self.split_data['X_train'])
                self.y_scaler = PowerTransformer().fit(self.split_data['Y_train'].reshape(-1,1))
                        
            self.split_data['X_train'] = self.x_scaler.transform(self.split_data['X_train'])
            self.split_data['X_test'] = self.x_scaler.transform(self.split_data['X_test'])
            self.split_data['Y_train'] = np.apply_along_axis(
                lambda x: self.y_scaler.transform(x.reshape(-1,1)), 0,self.split_data['Y_train']).reshape(-1,self.a)
            self.split_data['Y_test'] = np.apply_along_axis(
                lambda x: self.y_scaler.transform(x.reshape(-1,1)), 0, self.split_data['Y_test']).reshape(-1,self.a)
        
        else:
            self.split_data['X_train'] = self.split_data['X_train'].astype(float)
            self.split_data['X_test'] = self.split_data['X_test'].astype(float)

        if len(self.Y_rel) > 0:
            self.split_data['Yr_train'] = self.Y_rel.loc[train_ind].values
            self.split_data['Yr_test'] = self.Y_rel.loc[test_ind].values

        ps = 'Scaled' if scale else 'Unscaled'
        print(f'{ps} data split into training (size {len(self.split_data["X_train"])}) and test (size {len(self.split_data["X_test"])}) sets')
   
    def dropFeatures(self, keep):
        keep_ind = [self.featureNames.index(f) for f in keep]
        self.m = len(keep)

        self.featureNames = keep
        self.features = self.features.loc[:,keep]
            
        self.X_s = self.X_s[:,keep_ind]
        # self.X_c = self.X_c[:,keep_ind]
        
        if len(self.split_data) > 0:
            self.split_data['X_train'] = self.split_data['X_train'][:,keep_ind]
            self.split_data['X_test'] = self.split_data['X_test'][:,keep_ind]

        train_ind = self.split_data['Yb_train'].index if len(self.split_data) > 0 else self.performance.index
            
        if self.scaler == 's':
            self.x_scaler = StandardScaler().fit(self.features.loc[train_ind,:].values)
        elif self.scaler == 'r5':
            self.x_scaler = RobustScaler(quantile_range=(5.0,95.0),unit_variance=True).fit(
                self.features.loc[train_ind,:].values)
        elif self.scaler == 'r25':
            self.x_scaler = RobustScaler(quantile_range=(25.0,75.0),unit_variance=True).fit(
                self.features.loc[train_ind,:].values)
        elif self.scaler == 'p':
            self.x_scaler = PowerTransformer().fit(self.features.loc[train_ind,:].values)
        

        print(f'Features dropped. Remaining features: {self.m}')        


    def plot(self, proj, hue=None, legend=True):

        pltData = self.PlotProj[proj].copy()

        split = None if 'group' not in pltData.columns else 'group'

        if hue == 'Best':
            if split == 'group':
                pltData['Best'] = pd.concat([
                    self.split_data['Yb_train'], self.split_data['Yb_test']
                ])
            else:
                pltData['Best'] = self.performance['Best']
        else:
            hue = None

        if len([col for col in pltData.columns if col.startswith('Z')]) == 1:
            # density plot - group by split            
            sns.kdeplot(x=pltData['Z1'],hue=hue, data=pltData, fill=True, legend=legend)
        else: 
            sns.scatterplot(data=pltData, x='Z1', y='Z2', alpha=0.7, s=20,
                hue=hue, style=split, legend=legend)
            

    # adding projections
    def addProj(self, proj, proj_data):
        print(f'Manually adding {proj} projection')
        self.PlotProj[proj] = proj_data


            
    def projectNewInstances(self, X_new, proj):
        if proj not in self.projections.keys():
            print(f"{proj} projection not available")
            return
        
        # keep only features used in training
        X_new = X_new[self.featureNames]
        X_new_ind = X_new.index

        # scale
        if self.x_scaler is not None:
            X_new = self.x_scaler.transform(X_new.values)

        # project
        
        Z_new = pd.DataFrame(
                self.projections[proj].transform(X_new),
                columns=['Z1','Z2'], index=X_new_ind)
        
        return Z_new

    # delete projections
    def delProj(self, method):
        """Deletes projection from PlotProj and projections dict.

        Args:
            method (str): name of the projection to delete
        """
        if method in self.projections.keys():
            del self.projections[method]
            del self.PlotProj[method]
            print(f"{method} projection deleted")
        else:
            print(f"{method} projection not defined")    
    


if __name__ == "__main__":
    
    pass
