## ----- Adapted from
# M.A. Muñoz and K. Smith-Miles. Instance Space Analysis: A toolkit for the assessment of algorithmic power. andremun/InstanceSpace on Github. Zenodo, DOI:10.5281/zenodo.4484107, 2020.
### !!!! will change to reference to Python package when available
## -------------------

from IS_class import InstanceSpace
import pandas as pd
import numpy as np
from numpy.typing import NDArray

import pickle as pkl
import sys
import os

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score

from scipy import stats
import time


parallel_options = {'flag': False, 'n_cores': 3}

def compute_znorm(z: NDArray[np.double],
    ) -> tuple[list[float], list[float], NDArray[np.double]]:
    """Compute mormalized z, standard deviations and mean.

    Parameters
    ----------
    z : NDArray[np.double]
        The feature coordinates.

    Returns
    -------
    tuple[list[float], list[float], NDArray[np.double]]
    The mean, standard deviation and normalized feature coordinates.
    """
    z = stats.zscore(z, ddof=1)
    mu = np.mean(z, axis=0)
    sigma = np.std(z, ddof=1, axis=0)
    return (mu, sigma, z)

def generate_params(rng: np.random.Generator, nvals:int):
    """Generate hyperparameters for the SVM models.

    Parameters
    ----------
    rng : np.random.Generator
        The random number generator.
    """
    
    maxgrid, mingrid = 4, -10
    # Number of samples
    

    # Generate params space through latin hypercube samples for grid search
    lhs = stats.qmc.LatinHypercube(d=2, seed=rng)
    samples = lhs.random(nvals)
    c = 2 ** ((maxgrid - mingrid) * samples[:, 0] + mingrid)
    gamma = 2 ** ((maxgrid - mingrid) * samples[:, 1] + mingrid)
    return {"C": list(c)+list(c), "gamma": list(gamma)+list(gamma),
            "kernel": ["poly"]*nvals + ["rbf"]*nvals}

def fitmatsvm(
    z: NDArray[np.double],
    y_bin: NDArray[np.bool_],
    skf: StratifiedKFold,
    # is_poly_kernel: bool,
    param_space: dict[str, list[float]] | None,
    parallel_options: dict[bool, int],
    ) -> dict:
    """Train a SVM model based on configuration.

    Parameters
    ----------
    z : NDArray[np.double]
        The instance space.
    y_bin : NDArray[np.bool_]
        The binary labels.
    w : NDArray[np.double]
        The sample weights.
    skf : StratifiedKFold
        The stratified k-fold cross-validation object.
    is_poly_kernel : bool
        Whether to use a polynomial kernel.
    param_space : dict | None
        The hyperparameters for the SVM model.
    use_grid_search : bool
        Whether to use grid search for hyperparameter optimization.
    
    Returns
    -------
    _SvmRes
    The SVM result object.
    """
    # kernel = "poly" if is_poly_kernel else "rbf"
    svm_model = SVC(
        # kernel=kernel,
        random_state=1111,
        probability=False,
        degree=2, coef0=1,
    )
    # Perform grid search for hyperparameter optimization
    # The randomizedsearchCV is used to reduce the computational cost
    # by considering a limited number combination of hyperparameters
    optimization = RandomizedSearchCV(
        estimator=svm_model,
        n_iter=30,
        param_distributions=param_space,
        cv=skf,
        verbose=0,
        random_state=1111,  # Ensure reproducibility with a fixed seed
        n_jobs=(parallel_options['n_cores'] if parallel_options['flag'] else 1),
        )
    
    optimization.fit(z, y_bin)
    best_svm = optimization.best_estimator_
    c = optimization.best_params_["C"]
    g = optimization.best_params_["gamma"]
    k = optimization.best_params_["kernel"]

    # Perform cross-validated predictions using the best SVM model
    y_cv = cross_val_predict(best_svm, z, y_bin, cv=skf, method="predict")
    # Predict the labels and probabilities for the entire dataset
    y_hat = best_svm.predict(z)
    
    # p_sub = cross_val_predict(best_svm, z, y_bin, cv=skf, method="predict_proba")[:,1,]
    # p_hat = best_svm.predict_proba(z)[:, 1]

    return {'svm':best_svm,'Yhat':y_hat,'Ycv':y_cv,'c':c,'g':g,'k':k,}
        # 'Psub'=p_sub,'Phat'=p_hat,}

def determine_selections(nalgos: int, precision: list[float],
    y_hat: NDArray[np.bool_], default: int,
) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
    """Determine the selections based on the predicted labels and precision.

    Parameters
    ----------
    nalgos : int
        The number of algorithms.
    precision : list[float]
        The precision metrics.
    y_hat : NDArray[np.bool_]
        The predicted labels.
    y_bin : NDArray[np.bool_]
        The binary labels.
    """
    # Stores the index of the column with the highest mean value
    # for instances with no "good" algorithms
    # default = np.argmax(np.mean(y_bin, axis=0))    

    if nalgos > 1:
        # if ties (if multiple y_bin = 1), choose the one with the highest precision
        precision_array = np.array(precision)
        weighted_yhat = y_hat * precision_array[np.newaxis, :]
        selection0 = np.argmax(weighted_yhat, axis=1) + 1

        # Find the maximum value for each row in weighted_yhat
        best = np.max(weighted_yhat, axis=1)
    else:
        best = y_hat
        selection0 = y_hat.astype(np.int_)

    selection1 = np.copy(selection0)
    selection0[best <= 0] = 0
    selection1[best <= 0] = default
    return (selection0.reshape(-1,1), selection1.reshape(-1,1))   

def fit_pythia(Z, y_bin, algo_labels):

    rng = np.random.default_rng(1111)

    y_cv = np.zeros(y_bin.shape, dtype=bool)
    y_hat = np.zeros(y_bin.shape, dtype=bool)
    # pr0sub = np.zeros(y_bin.shape, dtype=np.double)
    # pr0hat = np.zeros(y_bin.shape, dtype=np.double)

    cp = StratifiedKFold(n_splits=5, shuffle=True, random_state=1111)
    svm = {}
    box_consnt = {}
    k_scale = {}
    kernel = {}
    precision_record = []

    # cvcmat = np.zeros((nalgos, 4), dtype=int)
    # accuracy_record = []
    # recall_record = []

    # Section 1: Normalize the feature matrix
    (mu, sigma, z) = compute_znorm(Z)

    print('-------------------------------------------------------------------------')
    print('PYTHIA with a polynomial kernel using grid search hyperparameter tuning')

    # Section 3: Train SVM model for each algorithm & Evaluate performance.
    overall_start_time = time.perf_counter()

    for i,alg in enumerate(algo_labels):
        # algo_start_time = time.perf_counter()
        param_space = generate_params(rng,30)

        res = fitmatsvm(z=z,y_bin=y_bin[:, i],skf=cp,
                # is_poly_kernel=opts.is_poly_krnl,
                param_space=param_space, parallel_options=parallel_options)

        # Record performance metrics
        y_cv[:, [i]] = res['Ycv'].reshape(-1, 1)
        y_hat[:, [i]] = res['Yhat'].reshape(-1, 1)
        # pr0sub[:, [i]] = res.Psub.reshape(-1, 1)
        # pr0hat[:, [i]] = res.Phat.reshape(-1, 1)
        box_consnt[alg] = res['c']
        k_scale[alg] = res['g']
        kernel[alg] = res['k']
        svm[alg] = res['svm']

        # CV precision
        precision = precision_score(y_bin[:, i], res['Ycv'])
        precision_record.append(precision)
        
        # cm = confusion_matrix(y_bin[:, i], res['Ycv'])
        # tn, fp, fn, tp = cm.ravel()
        # accuracy = accuracy_score(y_bin[:, i], res['Ycv'])
        # recall = recall_score(y_bin[:, i], res['Ycv'])
        # cvcmat[i, :] = [tn, fp, fn, tp]
        # accuracy_record.append(accuracy)
        # recall_record.append(recall)

    print(f"Total elapsed time:  {time.perf_counter() - overall_start_time:.2f}s")

    # predY_0 and predY_0cv good for plots
    default = np.argmax(np.mean(y_bin, axis=0)) + 1
    pred_hat = determine_selections(len(algo_labels), precision_record, y_hat, default)  
    pred_cv = determine_selections(len(algo_labels), precision_record, y_cv, default) 

    return {'bin_hat':y_hat, 'bin_cv':y_cv, 
            'best_hat':pred_hat, 'best_cv':pred_cv,
            'svm':svm, 'cv_precision':precision_record,'default_alg':default,
            'hyper_params':{'box':box_consnt, 'scale':k_scale, 'kernel':kernel},
            'norm_params':{'mu':mu, 'sigma':sigma},}

def predict_pythia(Z, normZ, svms, cv_precision, default_alg, algo_labels):
    # normalize Z with mu and sigma
    z = (Z - normZ['mu']) / normZ['sigma']
    y_hat = np.zeros((Z.shape[0], len(algo_labels)), dtype=bool)

    for i,alg in enumerate(algo_labels):
        y_hat[:, i] = svms[alg].predict(z)

    pred_hat = determine_selections(len(algo_labels), cv_precision, y_hat, default_alg)  
    
    return {'bin_hat':y_hat, 'best_hat':pred_hat}

def summary_stats(algo_labels,bin_act, bin_pred, best_act, best_pred, suf=''):

    stats = {}
    for i, alg in enumerate(algo_labels):
        stats[alg] = {'accuracy': accuracy_score(bin_act[:, i], bin_pred[:, i]),
            'precision': precision_score(bin_act[:, i], bin_pred[:, i]),
            'recall': recall_score(bin_act[:, i], bin_pred[:, i])}
        
    stats['best'] = {'accuracy': accuracy_score(best_act, best_pred),
        'precision': precision_score(best_act, best_pred,zero_division=1,average='weighted'),
        'recall': recall_score(best_act, best_pred,zero_division=1,average='macro')}
    
    stats = pd.DataFrame(stats).T
    stats.reset_index(inplace=True, names='algorithm')    
    stats['type'] = suf
    
    return stats

def summary_statsM(algo_labels,bin_act, bin_pred, best_labels, best_act, best_pred, suf=''):

    stats = {}
    for i, alg in enumerate(algo_labels):
        stats[alg] = {'accuracy': accuracy_score(bin_act[:, i], bin_pred[:, i]),
            'precision': precision_score(bin_act[:, i], bin_pred[:, i]),
            'recall': recall_score(bin_act[:, i], bin_pred[:, i])}
    for j, b in enumerate(best_labels):    
        stats[b] = {'accuracy': accuracy_score(best_act, best_pred[:, j]),
            'precision': precision_score(best_act, best_pred[:, j],zero_division=1,average='weighted'),
            'recall': recall_score(best_act, best_pred[:, j],zero_division=1,average='macro')}
        
    stats = pd.DataFrame(stats).T
    stats.reset_index(inplace=True, names='algorithm')    
    stats['type'] = suf
    
    return stats
##=============================================================
def predictionsIS(outpath, maxPerf, absPerf, perfThresh):
    iSpace = pd.read_pickle(outpath + 'instanceSpace_proj.pkl')

    alg_labels = [a.removeprefix('algo_') for a in iSpace.algorithms]

    Ybin_full = iSpace.getBinaryPerf(abs=absPerf, min= not maxPerf, thr=perfThresh)
    bin_train = Ybin_full.loc[iSpace.split_data['Yb_train'].index].values
    bin_test = Ybin_full.loc[iSpace.split_data['Yb_test'].index].values
        
    Ybest_train = iSpace.split_data['Yb_train'].apply(lambda x: alg_labels.index(x)+1)
    Ybest_test = iSpace.split_data['Yb_test'].apply(lambda x: alg_labels.index(x)+1)

    stats_out = []
    params_out = {}
    svms_out = {}
    preds_out = {}

    for proj in iSpace.PlotProj.keys():
        Zfull = iSpace.PlotProj[proj]

        Ztrain = Zfull.loc[Zfull['group']=='train'].drop(columns='group').values
        Ztest = Zfull.loc[Zfull['group']=='test'].drop(columns='group').values

        
        pythia_out = fit_pythia(Ztrain, bin_train, alg_labels)
        pythia_test = predict_pythia(Ztest, pythia_out['norm_params'], pythia_out['svm'],
                        pythia_out['cv_precision'], pythia_out['default_alg'], alg_labels)
        
        preds = np.vstack([np.hstack([pythia_out['bin_hat'],pythia_out['best_hat'][0],pythia_out['best_hat'][1]]),
                  np.hstack([pythia_test['bin_hat'],pythia_test['best_hat'][0],pythia_test['best_hat'][1]])])
        preds = pd.DataFrame(preds, columns=alg_labels+['best0','best1'], index=Zfull.index)
        preds['group'] = Zfull['group']
        
        out_tab = pd.concat([
            summary_stats(alg_labels, bin_train, pythia_out['bin_hat'], 
                    Ybest_train, pythia_out['best_hat'][1], 'train'),
            summary_stats(alg_labels, bin_train, pythia_out['bin_cv'],
                    Ybest_train, pythia_out['best_cv'][1], 'CV'),
            summary_stats(alg_labels, bin_test, pythia_test['bin_hat'],
                    Ybest_test, pythia_test['best_hat'][1], 'test')
        ])
        out_tab['projection'] = proj

        stats_out.append(out_tab)
        params_out[proj] = pythia_out['hyper_params']
        svms_out[proj] = pythia_out['svm']
        preds_out[proj] = preds

    stats_out = pd.concat(stats_out).reset_index(drop=True)
    params_out = pd.concat(
        {k: pd.DataFrame(v) for k,v in params_out.items()},
        names=['proj','algorithm']).reset_index()
    preds_out = pd.concat(
        {k: v for k,v in preds_out.items()},
        names=['proj','instances']).reset_index()

    return (stats_out, params_out, svms_out, preds_out)


