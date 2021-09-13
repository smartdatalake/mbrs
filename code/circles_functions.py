# METHODS USED IN CIRCULAR (FIXED_SHAPE) SCAN

import pandas as pd
import random
from shapely.geometry import box
from shapely.geometry import Point
from shapely.ops import cascaded_union, polygonize
from scipy.stats import entropy
from tqdm.notebook import tqdm, trange
import networkx as nx
import math
from rtree import index
import folium
import numpy as np
import collections
import heapq
import json
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay
from shapely import geometry
from shapely.geometry import Polygon, mapping
import time
from math import cos, sin, asin, sqrt, radians



# Progress bar
def progressBar(current, total, queueSize, barLength = 20):
    percent = float(current) * 100 / total
    arrow   = '-' * int(percent/100 * barLength - 1) + '>'
    spaces  = ' ' * (barLength - len(arrow))

    print('Progress: [%s%s] %d %%  Queue size: %d' % (arrow, spaces, percent, queueSize), end='\r')

    
# Pick a sample from the given data frame as seed points
def pickSeeds(df, seeds_ratio):
    
    sample = df.sample(int(seeds_ratio * len(df)))  
    seeds = dict()
    for idx, row in sample.iterrows():
        s = len(seeds) + 1
        seeds[s] = Point(row['lon'], row['lat'])
                
    return seeds



# Check if point p is within distance eps from at least one of the current items in the list
def checkCohesiveness(df, p, items, eps):
    for idx, row in df.loc[df.index.isin(items)].iterrows():
        if (p.distance(Point(row['lon'], row['lat'])) < eps):
            return True
    return False


# Helper functions used in score computation

def get_neighbors(G, region):
    return [n for v in region for n in list(G[v]) if n not in region]


def neighbor_extension(G, region):
    # get new neighbors
    neighbors = get_neighbors(G, region)
    
    region_ext = set(region.copy())
    # update region
    for n in neighbors:
        region_ext.add(n)
    
    return region_ext  


# SCORE from data frame
def get_region_score(df, types, region, params):
    categories = [df.loc[n][cat] for n in region] 
    init = [categories.count(t) for t in types]
    score, entr = compute_score(init, params)
    return score, entr
     

# SCORE from graph
def get_region_score_graph(G, types, region, params):
    categories = [G.nodes[n]['cat'] for n in region]
    init = [categories.count(t) for t in types]
    score, entr = compute_score(init, params)
    return score, entr


# score computation function
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



class PrioritizedObject:
    def __init__(self, data):
        self.data = data
   
    def __lt__(self, other):
        return self.data['score'] < other.data['score']
    
    def __str__(self):
        return f'{self.data}'
        # return f"score={self.data['score']}, length={len(self.data['region'])}"

        
        
## Result Logger: class for collecting results
class Result_Logger(object):
    def __init__(self, topk=1):   # topk=params['settings']['top-k']
        self.topk = topk
        self.reset()
    
    def reset(self):
        self.results = []
        self.iterations = 0
        self.start_time = time.time()
        self.updates = {}
        self.updates['time'] = {}
        self.updates['iterations'] = {}


    ## logs an entry specified by the dictionary data
    ## data *must* contain a key 'score', and can contain any other piece of information
    def log(self, **data):
        self.iterations += 1

        # Put this region into the maxheap of top-k regions according to its score
        if len(self.results) < self.topk or data['score'] > heapq.nsmallest(1,self.results)[0].data['score']:
            ## new result
            heapq.heappush(self.results, PrioritizedObject(data))
            # ... at the expense of the one currently having the lowest score
            if (len(self.results) > self.topk):
                heapq.heappop(self.results)
            topk_score = heapq.nsmallest(1,self.results)[0].data['score']

            # Statistics
            elapsed = time.time() - self.start_time
            self.updates['time'][elapsed] = topk_score
            self.updates['iterations'][self.iterations] = topk_score

    
    def get_top1_result(self):
        return heapq.nlargest(1,self.results)[0]

    def get_top1_score(self):
        return self.get_top1_result().data['score']


########################################################################
# Methods for queue manipulation
########################################################################

## performs heap-based search
## input_data can be a list of points, or a list of regions (list of points)
def queue_search(df, types, G, rtree, seeds, params, start_time, loggers=[]):
    queue = []
    
    ## initialize local logger
    local_logger = Result_Logger()
    loggers.append(local_logger)
    
    queue, neighbors = queue_init(df, types, G, rtree, seeds, params, loggers)
    
    iterations = 0
    while (time.time() - start_time) < params['variables']['time_budget']['current'] and len(queue) > 0:
        iterations += 1
        queue, t = queue_pop(df, types, G, neighbors, seeds, queue, params, loggers)
#        progressBar((time.time() - start_time), params['variables']['time_budget']['current'], len(queue))
    
    return queue


# Queue initialization (with seeds)
def queue_init(df, types, G, rtree, seeds, params, loggers=[]):
    
    # Priority queue for seeds
    queue = []
    
    # Keeps a list per seed of all its (max_size) neighbors by ascending distance
    neighbors = dict()
    
    local_size = 2   # Check the seed and its 1-NN

    for s in seeds:
        # Keep all (max_size) neighbors around this seed for retrieval during iterations
        neighbors[s] = list(rtree.nearest((seeds[s].x, seeds[s].y, seeds[s].x, seeds[s].y), params['variables']['max_size']['current'])).copy()    
    
        # Retrieve 2-NN points to the current seed
        region = neighbors[s][0:local_size]
        n1 = Point(df.loc[region[local_size-2]]['lon'], df.loc[region[local_size-2]]['lat'])
        n2 = Point(df.loc[region[local_size-1]]['lon'], df.loc[region[local_size-1]]['lat'])
        dist_farthest = seeds[s].distance(n2)

        # Drop this seed if its two closest neighbors are more than eps away 
        if (n1.distance(n2) > params['variables']['eps']['current']):
            continue
            
        # SCORE ESTIMATION
        if (params['methods']['current'] == 'ExpCircles'):
            region_ext = neighbor_extension(G, region)
            if len(region_ext) > params['variables']['max_size']['current']:
                continue 
            score, entr = get_region_score_graph(G, types, region_ext, params)   # Estimate score by applying EXPANSION with neighbors; the points in the circle still constitute the region
        else:
            score, entr = get_region_score_graph(G, types, region, params)    # Estimate score with only the new point included in the region   

        # update top-k score
        # topk_update(score, region, init) ## deprecated
        for logger in loggers:
            logger.log(score=score, region=region.copy(), seed=s, distance=dist_farthest)
    
        # Push this seed into a priority queue
        heapq.heappush(queue, (-score, (s, local_size, dist_farthest)))
    
    return queue, neighbors


# Pop a sed from priority queue and perform expansion
def queue_pop(df, types, G, neighbors, seeds, queue, params, loggers=[]):
   
    # Examine the seed currently at the head of the priority queue
    t = heapq.heappop(queue)
    score, s, local_size, dist_last = -t[0], t[1][0], t[1][1], t[1][2]    
#    print(score, s, local_size, dist_last)
    
    # number of neighbos to examine next
    local_size += 1 

    # check max size
    if local_size > params['variables']['max_size']['current'] or local_size > len(neighbors[s]):
        return queue, t  #continue

    # get one more point from its neighbors to construct the new region
    region = neighbors[s][0:local_size]
    p = Point(df.loc[region[local_size-1]]['lon'], df.loc[region[local_size-1]]['lat'])
    # its distance for the seed
    dist_farthest = seeds[s].distance(p)
        
    # SCORE ESTIMATION
    if (params['methods']['current'] == 'ExpCircles'):
        # COHESIVENESS CONSTRAINT: if next point is > eps away from all points in the current region of this seed, 
        # skip this point, but keep the seed in the priority queue for further search
        if not checkCohesiveness(df, p, neighbors[s][0:local_size-1], params['variables']['eps']['current']):   
            del neighbors[s][local_size-1]   # Remove point from neighbors
            heapq.heappush(queue, (-score, (s, local_size-1, dist_last)))
            return queue, t  #continue
        
        # RADIUS CONSTRAINT: if next point is > eps away from the most extreme point in the current region, 
        # discard this seed, as no better result can possibly come out of it    
        if (dist_farthest - dist_last > params['variables']['eps']['current']):
            return queue, t  #continue
            
        # COMPLETENESS CONSTRAINT: Skip this seed if extented region exceeds max_size
        region_ext = neighbor_extension(G, region)
        if len(region_ext) > params['variables']['max_size']['current']:
            return queue, t    #continue 
        score, entr = get_region_score_graph(G, types, region_ext, params)   # Estimate score by applying EXPANSION with neighbors; the points in the circle still constitute the region
    else:
        score, entr = get_region_score_graph(G, types, region, params)    # Estimate score with only the new point included in the region
    
    # update top-k score and region
    for logger in loggers:
        logger.log(score=score, region=region.copy(), seed=s, distance=dist_farthest)

    # Push this seed back to the queue
    heapq.heappush(queue, (-score, (s, local_size, dist_farthest)))
    
    return queue, t


# Run the method
def run_method(df, types, G, rtree, seeds, params, start_time):
    
    logger = Result_Logger()

    # Initialize queue with seeds and start searching
    queue = queue_search(df, types, G, rtree, seeds, params, start_time, loggers=[logger])
 
    # Return top-1 region
    top_region = logger.results[0]
    
    # Timeline of updates: a dictionary of time instant (keys) where a new top-1 score (value) was found
    updates = logger.updates['time']

    return top_region.data['score'], top_region.data['region'], updates



# Bounding box of all coordinates in a data frame
def bbox(df):
    min_lon = df['lon'].min()
    min_lat = df['lat'].min()
    max_lon = df['lon'].max()
    max_lat = df['lat'].max()

    return min_lon, min_lat, max_lon, max_lat
