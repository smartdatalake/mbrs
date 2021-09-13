# METHODS USED IN GRAPH EXPANSION


from rtree import index
from tqdm.notebook import tqdm
import networkx as nx
import pandas as pd
import random
from scipy.stats import entropy
import math
import numpy as np
from sklearn.cluster import KMeans
import time
import heapq
import geopandas as gpd
from random import sample
from statistics import mean, median


def load_data(input_file, min_lon=None, max_lon=None, min_lat=None, max_lat=None):
    
    col_sep = ';'
    header = None
    col_names = ['id', 'name', 'lon', 'lat', 'kwds']
    df = pd.read_csv(input_file, delimiter=col_sep, header=header, names=col_names, error_bad_lines=False)

    df = df[pd.to_numeric(df['lon'], errors='coerce').notnull()]
    df = df[pd.to_numeric(df['lat'], errors='coerce').notnull()]
    df['lon'] = df['lon'].apply(lambda x: float(x))
    df['lat'] = df['lat'].apply(lambda x: float(x))
    df['cat1'] = df['kwds'].apply(lambda x: x.split(',')[0])
    df['cat2'] = df['kwds'].apply(lambda x: x.split(',')[1])
    df = df.drop('kwds', 1)

    if min_lon != None:
        df = df.loc[df['lon'] >= min_lon]
        df = df.loc[df['lon'] <= max_lon]
        df = df.loc[df['lat'] >= min_lat]
        df = df.loc[df['lat'] <= max_lat]

    df.reset_index(drop=True, inplace=True)

    types1 = set(df['cat1'].tolist())
    type1_colors = {t: "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for t in types1}
    types2 = set(df['cat2'].tolist())
    type2_colors = {t: "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for t in types2}

    # select category level
    cat_level = 1
    if cat_level == 1:
        cat = 'cat1'
        types = types1
        colors = type1_colors
    else:
        cat = 'cat2'
        types = types2
        colors = type2_colors

    print('#types: ' + str(len(types)))

    type_count_global = {t: len(df.loc[df[cat] == t].index) for t in types}
    n_total_global = len(df.index)
    print('#points: ' + str(n_total_global))

    global_distr = [type_count_global[t] / n_total_global for t in type_count_global]
    print('global entropy: ' + str(entropy(global_distr)))

    max_se = math.log(len(types))
    print('max entropy: ' + str(max_se))

    return df, types, colors


def compute_score(init, params):
    
    size = sum(init)
    distr = [x / size for x in init]    
    
    rel_se = entropy(distr) / params['settings']['max_se']
    rel_size = size / params['variables']['max_size']['current']

    if params['entropy_mode']['current'] == 'high':
        score = rel_se * (rel_size ** params['variables']['size_weight']['current'])
    else:
        rel_se = 1 - rel_se
        score = rel_se * (rel_size ** params['variables']['size_weight']['current'])
    
    return score, rel_se


def create_graph(df, eps):
    
    # create R-tree index
    rtree = index.Index()

    for idx, row in tqdm(df.iterrows()):
        left, bottom, right, top = row['lon'], row['lat'], row['lon'], row['lat']
        rtree.insert(idx, (left, bottom, right, top))
    
    # construct the graph
    G = nx.Graph()

    for idx, row in tqdm(df.iterrows()):

        # create vertex
        G.add_nodes_from([(idx, {'cat': df.loc[idx]['cat1']})])

        # retrieve neighbors and create edges
        neighbors = list()
        left, bottom, right, top = row['lon'] - eps, row['lat'] - eps, row['lon'] + eps, row['lat'] + eps
        neighbors = [n for n in rtree.intersection((left, bottom, right, top))]
        a = np.array([df.loc[idx]['lon'], df.loc[idx]['lat']])
        for n in neighbors:
            if idx < n:
                b = np.array([df.loc[n]['lon'], df.loc[n]['lat']])
                dist = np.linalg.norm(a - b)
                if dist <= eps:
                    G.add_edge(idx, n)
    
    # check max node degree
    cc = [d for n, d in G.degree()]
    max_degree = sorted(cc)[-1] + 1
    mean_degree = mean(cc)
    median_degree = median(cc)
    print('Max degree: ' + str(max_degree) + ' Mean degree: ' + str(mean_degree) + ' Median degree: ' + str(median_degree))
    
    # check connected components
    print('Max connected component: ' + str([len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)][0]))
    
    return G, rtree


def create_seeds(df, rtree, seeds_ratio):
    
    # Compute clusters with k-means
    num_clusters = int(seeds_ratio * len(df.index))
    X = df[['lon', 'lat']].to_numpy()
    kmeans = KMeans(n_clusters=num_clusters).fit(X)
    cluster_labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_
    
    # Pick seeds from cluster centroids
    seeds = []
    for c in cluster_centers:
        seeds.append(list(rtree.nearest([c[0], c[1]], 1))[0])
    
    return seeds


def init_scores(G, types, params):
    
    for v in G.nodes:

        # create region
        region_core = {v}
        region_border = set(G[v])
        region = region_core.union(region_border)

        # compute 'init'
        categories = [G.nodes[n]['cat'] for n in region]
        init = [categories.count(t) for t in types]

        # compute score
        score, rel_se = compute_score(init, params)

        # store the information in the node
        G.nodes[v]['init_score'] = score


def init_queue(G, seeds, types, params, topk_regions, start_time, updates):

    queue = []

    # for v in tqdm(seeds):
    for v in seeds:
    
        # create region
        region_core = {v}
        region_border = set(G[v])
        region = region_core.union(region_border)
    
        # check if border node is actually core
        border_to_core = set()
        for n in region_border:
            has_new_neighbors = False
            for nn in set(G[n]):
                if nn not in region:
                    has_new_neighbors = True
                    break
            if not has_new_neighbors:
                border_to_core.add(n)
        for n in border_to_core:
            region_border.remove(n)
            region_core.add(n)
        
        # check max size
        if len(region) > params['variables']['max_size']['current']:
            continue
    
        # compute 'init'
        categories = [G.nodes[n]['cat'] for n in region]
        init = [categories.count(t) for t in types]
        
        # compute score
        score, rel_se = compute_score(init, params)
    
        # update top-k regions
        if score > topk_regions[params['settings']['top_k'] - 1][0]:
            topk_regions.append([score, rel_se,
                                 [region_core.copy(), region_border.copy()],
                                 init.copy(), len(region)])
            topk_regions = sorted(topk_regions, key=lambda topk_regions: topk_regions[0], reverse=True)
            topk_regions = topk_regions[:-1]
            updates[time.time() - start_time] = topk_regions[params['settings']['top_k'] - 1][0]
            # print('init ' + str(score))
    
        # add to queue if border is not empty
        if len(region_border) > 0:
            heapq.heappush(queue, (-score, (region_core.copy(), region_border.copy())))
            # print(score, (region_core.copy(), region_border.copy()))
    
    # print('Total queue entries: ' + str(len(queue)))
    
    return queue, topk_regions


def expand_region(G, region_core, region_border, nodes_to_expand, params, types):
    
    new_region_core = region_core.copy()
    new_region_border = region_border.copy()
    
    for n in nodes_to_expand:

        # move selected border node to core
        new_region_border.remove(n)
        new_region_core.add(n)

        # find new neighbors and add them to border
        new_neighbors = set(G[n])
        for nn in new_neighbors:
            if nn not in new_region_core:
                new_region_border.add(nn)

    # get the newly formed region
    new_region = new_region_core.union(new_region_border)

    # check if border node is actually core
    border_to_core = set()
    for n in new_region_border:
        has_extra_neighbors = False
        for nn in set(G[n]):
            if nn not in new_region:
                has_extra_neighbors = True
                break
        if not has_extra_neighbors:
            border_to_core.add(n)
    for n in border_to_core:
        new_region_border.remove(n)
        new_region_core.add(n)

    # compute 'init'
    categories = [G.nodes[n]['cat'] for n in new_region]
    init = [categories.count(t) for t in types]

    # compute score
    score, rel_se = compute_score(init, params)
    
    return new_region, new_region_core, new_region_border, init, score, rel_se


def process_queue(G, queue, topk_regions, params, types, start_time, updates):
    
    # POP THE NEXT REGION TO EXPAND
    t = heapq.heappop(queue)
    score, region_core, region_border = -t[0], t[1][0], t[1][1]

    if params['methods']['current'] == 'ExpSingle':   # FIND THE BEST BORDER NODE TO EXPAND

        best_region_core = set()
        best_region_border = set()
        best_region_score = -1
        best_region_rel_se = -1

        for n in region_border:
            
            # expand region with this border point
            new_region, new_region_core, new_region_border, init, new_score, new_rel_se = expand_region(
                G, region_core, region_border, [n], params, types
            )

            # check max size
            if len(new_region) > params['variables']['max_size']['current']:
                continue

            # update top-k regions
            if new_score > topk_regions[params['settings']['top_k'] - 1][0]:
                topk_regions.append([new_score,
                                     new_rel_se,
                                     [new_region_core.copy(), new_region_border.copy()],
                                     init.copy(),
                                     len(new_region)])
                topk_regions = sorted(topk_regions, key=lambda topk_regions: topk_regions[0], reverse=True)
                topk_regions = topk_regions[:-1]
                updates[time.time() - start_time] = topk_regions[params['settings']['top_k'] - 1][0]
                
            # update current best score
            if new_score > best_region_score and len(new_region_border) > 0:
                best_region_core = new_region_core.copy()
                best_region_border = new_region_border.copy()
                best_region_score = new_score
                best_region_rel_se = new_rel_se

        # ADD THE BEST FOUND NEW REGION TO QUEUE
        if best_region_score > -1:
            heapq.heappush(queue, (-best_region_score, (best_region_core.copy(), best_region_border.copy())))
        
        return best_region_score, topk_regions

    elif params['methods']['current'] == 'ExpAll':   # EXPAND THE ENTIRE BORDER

        # expand region with all border points
        new_region, new_region_core, new_region_border, init, new_score, new_rel_se = expand_region(
            G, region_core, region_border, region_border, params, types
        )
        
        # check max size
        if len(new_region) > params['variables']['max_size']['current']:
            return -1, topk_regions

        # update top-k regions
        if new_score > topk_regions[params['settings']['top_k'] - 1][0]:
            topk_regions.append([new_score,
                                 new_rel_se,
                                 [new_region_core.copy(), new_region_border.copy()],
                                 init.copy(),
                                 len(new_region)])
            topk_regions = sorted(topk_regions, key=lambda topk_regions: topk_regions[0], reverse=True)
            topk_regions = topk_regions[:-1]
            updates[time.time() - start_time] = topk_regions[params['settings']['top_k'] - 1][0]

        # ADD THE NEW REGION TO QUEUE
        if len(new_region_border) > 0:
            heapq.heappush(queue, (-new_score, (new_region_core.copy(), new_region_border.copy())))
        
        return new_score, topk_regions
    
    
def get_region_shapes(df, topk_regions):
    region_shapes = []
    for r in topk_regions:
        region = df.loc[df.index.isin(r[2][0].union(r[2][1]))]
        region = gpd.GeoDataFrame(region, geometry=gpd.points_from_xy(region.lon, region.lat))        
        region = region.unary_union.convex_hull
        region_shapes.append(region)
    region_shapes = pd.DataFrame(region_shapes, columns=['geom'])
    region_shapes = gpd.GeoDataFrame(region_shapes, geometry='geom')
    region_shapes.crs = {'init':'epsg:4326'}
    return region_shapes


def run_exp_hybrid(G, seeds, params, types, topk_regions, start_time, updates):
    
    # create priority queue for regions
    queue = []
    
    # PART I: For each seed, perform ExpAll
    for v in seeds:
        
        # initialize best local region
        best_region_score = 0
        best_region = set()
    
        # initialize region
        region_core = {v}
        region_border = set(G[v])
        region = region_core.union(region_border)
        
        # expand region until max size
        while len(region) <= params['variables']['max_size']['current'] and len(region_border) > 0 and (time.time() - start_time) < params['variables']['time_budget']['current']:
            
            # compute 'init'
            categories = [G.nodes[n]['cat'] for n in region]
            init = [categories.count(t) for t in types]

            # compute score
            score, rel_se = compute_score(init, params)

            # update top-k regions
            if score > topk_regions[params['settings']['top_k'] - 1][0]:
                topk_regions.append([score, rel_se,
                                     [region_core.copy(), region_border.copy()],
                                     init.copy(), len(region)])
                topk_regions = sorted(topk_regions, key=lambda topk_regions: topk_regions[0], reverse=True)
                topk_regions = topk_regions[:-1]
                updates[time.time() - start_time] = topk_regions[params['settings']['top_k'] - 1][0]

            # check if local best
            if score > best_region_score:
                best_region_score = score
                best_region = region.copy()
            
            # check if border node is actually core
            border_to_core = set()
            for n in region_border:
                has_new_neighbors = False
                for nn in set(G[n]):
                    if nn not in region:
                        has_new_neighbors = True
                        break
                if not has_new_neighbors:
                    border_to_core.add(n)
            for n in border_to_core:
                region_border.remove(n)
                region_core.add(n)
            
            # expand region with all border points
            region, region_core, region_border, init, score, rel_se = expand_region(
                G, region_core, region_border, region_border, params, types
            )            
        
        # add best found region to queue
        if len(best_region) > 0:
            heapq.heappush(queue, (-best_region_score, best_region))
        
    # PART II: For each seed region, perform ExpSingle
    while len(queue) > 0 and (time.time() - start_time) < params['variables']['time_budget']['current']:
        
        # get the next seed region
        t = heapq.heappop(queue)
        score, seed_region = -t[0], t[1]
        
        # pick a seed
        v = seed_region.pop()
        
        # initialize region
        region_core = {v}
        region_border = set(G[v])
        region = region_core.union(region_border)
        
        # initialize best local region
        best_local_region_score = 0
        
        # expand region until max size
        while len(region) <= params['variables']['max_size']['current'] and len(region_border) > 0 and (time.time() - start_time) < params['variables']['time_budget']['current']:
        
            # compute 'init'
            categories = [G.nodes[n]['cat'] for n in region]
            init = [categories.count(t) for t in types]

            # compute score
            score, rel_se = compute_score(init, params)

            # update top-k regions
            if score > topk_regions[params['settings']['top_k'] - 1][0]:
                topk_regions.append([score, rel_se,
                                     [region_core.copy(), region_border.copy()],
                                     init.copy(), len(region)])
                topk_regions = sorted(topk_regions, key=lambda topk_regions: topk_regions[0], reverse=True)
                topk_regions = topk_regions[:-1]
                updates[time.time() - start_time] = topk_regions[params['settings']['top_k'] - 1][0]

            # check if border node is actually core
            border_to_core = set()
            for n in region_border:
                has_new_neighbors = False
                for nn in set(G[n]):
                    if nn not in region:
                        has_new_neighbors = True
                        break
                if not has_new_neighbors:
                    border_to_core.add(n)
            for n in border_to_core:
                region_border.remove(n)
                region_core.add(n)

            # find best border point to expand
        
            best_region_core = set()
            best_region_border = set()
            best_region_score = -1
            best_region_rel_se = -1

            for n in region_border:

                # expand region with this border point
                new_region, new_region_core, new_region_border, init, new_score, new_rel_se = expand_region(
                    G, region_core, region_border, [n], params, types
                )

                # check max size
                if len(new_region) > params['variables']['max_size']['current']:
                    continue

                # update top-k regions
                if new_score > topk_regions[params['settings']['top_k'] - 1][0]:
                    topk_regions.append([new_score,
                                         new_rel_se,
                                         [new_region_core.copy(), new_region_border.copy()],
                                         init.copy(),
                                         len(new_region)])
                    topk_regions = sorted(topk_regions, key=lambda topk_regions: topk_regions[0], reverse=True)
                    topk_regions = topk_regions[:-1]
                    updates[time.time() - start_time] = topk_regions[params['settings']['top_k'] - 1][0]

                # update current best score
                if new_score > best_region_score and len(new_region_border) > 0:
                    best_region_core = new_region_core.copy()
                    best_region_border = new_region_border.copy()
                    best_region_score = new_score
                    best_region_rel_se = new_rel_se

            # set current region to best
            region_core = best_region_core
            region_border = best_region_border
            region = region_core.union(region_border)
            
            # update best local score
            if best_region_score > best_local_region_score:
                best_local_region_score = best_region_score
                
        # push back to queue with new score
        if len(seed_region) > 0:
            heapq.heappush(queue, (-best_local_region_score, seed_region))
            
    return topk_regions
        

def run_method(df, G, rtree, types, params, start_time, seeds):

    # Initialize top-k list
    topk_regions = []
    while len(topk_regions) < params['settings']['top_k']:
        topk_regions.append([0, 0, [set(), set()], [], 0])   # [score, rel_se, [region_core, region_border], init, length]

    iterations = 0
    updates = dict()
    
    if params['methods']['current'] == 'ExpHybrid':
        topk_regions = run_exp_hybrid(G, seeds, params, types, topk_regions, start_time, updates)
    else:
        # Initialize queue
        queue, topk_regions = init_queue(G, seeds, types, params, topk_regions, start_time, updates)

        # Process queue
        while (time.time() - start_time) < params['variables']['time_budget']['current'] and len(queue) > 0:
            iterations += 1
            score, topk_regions = process_queue(G, queue, topk_regions, params, types, start_time, updates)
    
    return topk_regions[0][0], topk_regions[0][2][0].union(topk_regions[0][2][1]), updates # return score, region and updates
