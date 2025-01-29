import numpy as np
import pandas as pd

import os
import shutil

import vrplib
from sklearn.cluster import DBSCAN
from math import sqrt
from copy import deepcopy

import matplotlib.pyplot as plt
import seaborn as sns

# name = 'A-n32-k5'
# M = [0.5]
# replace = False
# path_in = '../Instances/'


def modify_instance(name, M, replace, path_in, path_out, plot=False):
    # read the instance
    og_instance  = vrplib.read_instance(f'{path_in}{name}.vrp', 
                                        compute_edge_weights=True)

    # dist_depot = og_instance['edge_weight'][0,1:]
    dist = og_instance['edge_weight'][1:,1:]

    depot_coords = og_instance['node_coord'][int(og_instance['depot'][0])]
    coords = og_instance['node_coord'][1:] # assumes depot is node 0    

    # DBSCAN clustering
    eps = (np.max(coords) - np.min(coords))/sqrt(len(coords))  # uniform
    # eps = np.percentile(dist[dist>0],0.5)  # percentile    

    db = DBSCAN(eps=eps, min_samples=4, metric='precomputed').fit(dist)
    clusters = np.unique(db.labels_)
    clusters = np.delete(clusters, np.where(clusters==-1))

    if len(clusters) == 0:
        print('No clusters found')
        return {name: 0}

    new_coords = {m: [] for m in M}
    new_demands = []
    rep_ind = []
    new_coords['centroid'] = []

    for c in clusters:
        # get the nodes in the cluster
        cluster_coords = coords[np.where(db.labels_ == c)]
        cluster_demands = og_instance['demand'][1:][np.where(db.labels_ == c)]
        centroid = np.mean(cluster_coords, axis=0)
        new_coords['centroid'].append(centroid)

        # dist_to_depot = np.linalg.norm(centroid - depot_coords)
        
        synth_outlier = {m: np.round(depot_coords + m*(centroid - depot_coords )) for m in M}

        if replace:
            # find the node closest to the centroid
            closest = np.argmin(np.linalg.norm(cluster_coords - centroid, axis=1))
            closest = np.where(db.labels_ == c)[0][closest]
            rep_ind.append(closest)
        else:
            # new demand if not replacing
            new_demands.append(np.median(cluster_demands)) # could be min    

        # new coords
        for m in M:
            new_coords[m].append(synth_outlier[m])        
                

    # plot  
    if plot: 
        plt.figure(figsize=(5*len(M),4))
        for i,m in enumerate(M):
            plt.subplot(1,len(M),i+1)
            plt.scatter(coords[:,0], coords[:,1], c=db.labels_,s=80,edgecolors='tab:grey')
            plt.scatter(depot_coords[0], depot_coords[1], c='r', marker='s', s=120)
            plt.scatter(np.array(new_coords[m])[:,0], np.array(new_coords[m])[:,1], 
                        c='black', marker='x', s=80)
            plt.title(f'm={m}',fontsize='x-large')
            # set xlim and ylim
            plt.xlim([-400, 1200])
            plt.ylim([-400, 1000])
            # remove axis ticks
            plt.xticks([])
            plt.yticks([])
            
            # line between centroid and depot
            line_list = m if m > 1 else 'centroid'
            for p in new_coords[line_list]:
                plt.plot([p[0], depot_coords[0]], [p[1], depot_coords[1]], 
                        'k--', alpha=0.5)
                
        plt.savefig(f'{path_out}/mod_{name}.pdf', bbox_inches='tight')
            
    # write new instances
    for m in M:
        new_instance_ = deepcopy(og_instance)
        if replace:
            for i,ind in enumerate(rep_ind):
                new_instance_['node_coord'][ind] = new_coords[m][i]
            
        else:
            new_instance_['node_coord'] = np.vstack(
                [og_instance['node_coord'], np.array(new_coords[m])])
        
            new_instance_['demand'] = np.concatenate((og_instance['demand'], new_demands))

        r = 'r' if replace else 'a'
        new_name = f'mod_{name}_m{m}{r}'
        comment_add = 'customers moved' if replace else 'new customers'

        # new_instance = {
        #     'NAME\t': new_name,
        #     'COMMENT\t': f'Modified {name} with {len(clusters)} {comment_add} at distance {m}',
        #     'TYPE\t': 'CVRP',
        #     'DIMENSION\t': len(new_instance_['node_coord']),
        #     'EDGE_WEIGHT_TYPE\t': 'EUC_2D',
        #     'CAPACITY\t': og_instance['capacity'],
        #     'NODE_COORD_SECTION\t': new_instance_['node_coord'].astype(int),
        #     'DEMAND_SECTION\t': new_instance_['demand'].astype(int),
        #     'DEPOT_SECTION\t': [1, -1],
        # }        
        # vrplib.write_instance(f'{path_out}/{new_name}.vrp',new_instance)

        f = open(f'{path_out}/{new_name}.vrp', 'w')
        f.write('NAME : ' + new_name + '\n')
        f.write('COMMENT : ' f'Modified {name} with {len(clusters)} {comment_add} at distance {m}\n')
        f.write('TYPE : CVRP\n')
        f.write('DIMENSION : ' + str(len(new_instance_['node_coord'])) + '\n')
        f.write('EDGE_WEIGHT_TYPE : EUC_2D\n')
        f.write('CAPACITY : ' + str(int(og_instance['capacity'])) + '\n')
        f.write('NODE_COORD_SECTION\n')

        for i,v in enumerate(new_instance_['node_coord']):
            f.write('{:<4}'.format(i+1)+' '+'{:<4}'.format(v[0])+' '+'{:<4}'.format(v[1])+'\n')

        f.write('DEMAND_SECTION\n')
        for i,d in enumerate(new_instance_['demand'].astype(int)):
            f.write('{:<4}'.format(i+1)+' '+'{:<4}'.format(d)+'\n')

        f.write('DEPOT_SECTION\n1\n-1\nEOF\n')
        f.close()

    return {name: len(clusters)}

# with modification when no clusters are found 
# need to skip if cluster centroid == depot
def modify_instance0(name, M, replace, path_in, path_out, plot=False):
    # read the instance
    og_instance  = vrplib.read_instance(f'{path_in}{name}.vrp', 
                                        compute_edge_weights=True)

    # dist_depot = og_instance['edge_weight'][0,1:]
    dist = og_instance['edge_weight'][1:,1:]

    depot_coords = og_instance['node_coord'][int(og_instance['depot'][0])]
    coords = og_instance['node_coord'][1:] # assumes depot is node 0    

    # DBSCAN clustering
    eps = (np.max(coords) - np.min(coords))/sqrt(len(coords))  # uniform
    # eps = np.percentile(dist[dist>0],0.5)  # percentile    

    db = DBSCAN(eps=eps, min_samples=4, metric='precomputed').fit(dist)
    clusters = np.unique(db.labels_)
    if len(clusters) == 1:
        print('No clusters found')
    else:
        clusters = np.delete(clusters, np.where(clusters==-1))

        

    new_coords = {m: [] for m in M}
    new_demands = []
    rep_ind = []
    new_coords['centroid'] = []

    for c in clusters:
        # get the nodes in the cluster
        cluster_coords = coords[np.where(db.labels_ == c)]
        cluster_demands = og_instance['demand'][1:][np.where(db.labels_ == c)]
        centroid = np.mean(cluster_coords, axis=0)
        new_coords['centroid'].append(centroid)

        # dist_to_depot = np.linalg.norm(centroid - depot_coords)
        
        synth_outlier = {m: np.round(depot_coords + m*(centroid - depot_coords )) for m in M}

        if replace:
            # find the node closest to the centroid
            closest = np.argmin(np.linalg.norm(cluster_coords - centroid, axis=1))
            closest = np.where(db.labels_ == c)[0][closest]
            rep_ind.append(closest)
        else:
            # new demand if not replacing
            new_demands.append(np.median(cluster_demands)) # could be min    

        # new coords
        for m in M:
            new_coords[m].append(synth_outlier[m])        
                

    # plot  
    if plot: 
        for m in M:
            plt.figure()
            plt.scatter(coords[:,0], coords[:,1], c=db.labels_)
            plt.scatter(depot_coords[0], depot_coords[1], c='r', marker='s', s=100)
            plt.scatter(np.array(new_coords[m])[:,0], np.array(new_coords[m])[:,1], 
                        c='black', marker='x', s=80)
            plt.title(m)
            
            # line between centroid and depot
            line_list = m if m > 1 else 'centroid'
            for p in new_coords[line_list]:
                plt.plot([p[0], depot_coords[0]], [p[1], depot_coords[1]], 
                        'k--', alpha=0.5)
        
            
    # write new instances
    for m in M:
        new_instance_ = deepcopy(og_instance)
        if replace:
            for i,ind in enumerate(rep_ind):
                new_instance_['node_coord'][ind] = new_coords[m][i]
            
        else:
            new_instance_['node_coord'] = np.vstack(
                [og_instance['node_coord'], np.array(new_coords[m])])
        
            new_instance_['demand'] = np.concatenate((og_instance['demand'], new_demands))

        r = 'r' if replace else 'a'
        new_name = f'mod_{name}_m{m}{r}'
        comment_add = 'customers moved' if replace else 'new customers'

        # new_instance = {
        #     'NAME\t': new_name,
        #     'COMMENT\t': f'Modified {name} with {len(clusters)} {comment_add} at distance {m}',
        #     'TYPE\t': 'CVRP',
        #     'DIMENSION\t': len(new_instance_['node_coord']),
        #     'EDGE_WEIGHT_TYPE\t': 'EUC_2D',
        #     'CAPACITY\t': og_instance['capacity'],
        #     'NODE_COORD_SECTION\t': new_instance_['node_coord'].astype(int),
        #     'DEMAND_SECTION\t': new_instance_['demand'].astype(int),
        #     'DEPOT_SECTION\t': [1, -1],
        # }        
        # vrplib.write_instance(f'{path_out}/{new_name}.vrp',new_instance)

        f = open(f'{path_out}/{new_name}.vrp', 'w')
        f.write('NAME : ' + new_name + '\n')
        f.write('COMMENT : ' f'Modified {name} with {len(clusters)} {comment_add} at distance {m}\n')
        f.write('TYPE : CVRP\n')
        f.write('DIMENSION : ' + str(len(new_instance_['node_coord'])) + '\n')
        f.write('EDGE_WEIGHT_TYPE : EUC_2D\n')
        f.write('CAPACITY : ' + str(int(og_instance['capacity'])) + '\n')
        f.write('NODE_COORD_SECTION\n')

        for i,v in enumerate(new_instance_['node_coord']):
            f.write('{:<4}'.format(i+1)+' '+'{:<4}'.format(v[0])+' '+'{:<4}'.format(v[1])+'\n')

        f.write('DEMAND_SECTION\n')
        for i,d in enumerate(new_instance_['demand'].astype(int)):
            f.write('{:<4}'.format(i+1)+' '+'{:<4}'.format(d)+'\n')

        f.write('DEPOT_SECTION\n1\n-1\nEOF\n')
        f.close()

    return {name: len(clusters)}

        
def modify_instance_folder(path_in, path_out, M, replace):

    # create output folder
    if not os.path.exists(path_out):
        os.makedirs(path_out)

    os.makedirs(f'{path_out}new-instances/', exist_ok=True)

    instances = [f.removesuffix('.vrp') for f in os.listdir(path_in) if f.endswith('.vrp')]
    mod_list = {}

    # remove value 1 from M
    M_ = [m for m in M if m != 1]
            

    for name in instances:
        print(name, end='\r')

        if replace == 'both':
            mr = modify_instance(name, M_, True, path_in, f'{path_out}new-instances/')
            mn = modify_instance(name, M, False, path_in, f'{path_out}new-instances/')
            mod_list.update(mr)
        else:
            useM = M_ if replace else M
            mod_list.update(
                modify_instance(name, useM, replace, path_in, f'{path_out}new-instances/'))
            
    # zip path_out
    shutil.make_archive(f'{path_out}new-instances', 'zip', f'{path_out}new-instances/')
    shutil.rmtree(f'{path_out}new-instances/')

    pd.DataFrame.from_dict(mod_list, orient='index', columns=['changes']).to_csv(
        f'{path_out}/mod_count.csv', index_label='instances')


# modify_instance_folder(
#     path_in='../Instances/',
#     path_out='../Instances/mod_test/',
#     M=[0.5, 1, 2],
#     replace='both'
# )


# modify_instance(
#     name='A-n32-k5',
#     M=[0.5, 1, 2],
#     replace=False,
#     path_in='../Instances/',
#     path_out='../Instances/mod_test/',
#     plot=False
# )


