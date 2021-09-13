# METHODS USED IN ADAPTIVE GRID METHOD

import time
import heapq
import random
import pandas as pd
from scipy.stats import entropy
from itertools import product
from skopt import Space
from skopt import Optimizer
from skopt.space.space import Integer
from skopt.space.space import Categorical
import folium
from folium.plugins import HeatMap
from folium.map import Popup

from skopt.learning import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import ConstantKernel, Matern
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel)



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
    
    return score


# General Functions

def get_neighbors(G, region):
    return set([n for v in region for n in list(G[v]) if n not in region])


def extend_with_border(G, core):
    # get new neighbors
    border = set(get_neighbors(G, core))
    
    return core | border 


## extend to border, then compute score
def get_score_of_core(G, core, params, types):
    region = extend_with_border(G, core)
    return get_score_of_region(G, region, params, types)


def get_score_of_region(G, region, params, types):
    max_size = params['variables']['max_size']['current']
    if len(region) > max_size:
        return 0, None

    categories = [G.nodes[n]['cat'] for n in region]
    init = [categories.count(t) for t in types]
    score = compute_score(init, params)
    return score, init


def is_valid_region(G, region):
    core, border = get_core_border(G, region)
    # print(f'region {len(region)}: {region}')
    # print(f'core {len(core)}: {core}')
    # print(f'border {len(border)}: {core}')
    if (core | border) == region:
        return True
    else:
        return False


def is_core_connected(G, region):
    components = get_core_connected_components(G, region)
    return len(components) == 1
    # core, border = get_core_border(region)
    # return is_connected(core)

def get_core_connected_components(G, region):
    core, border = get_core_border(G, region)
    subgraph = G.subgraph(core)
    Gcc = sorted(nx.connected_components(subgraph), key=len, reverse=True)
    components = [ set( G.subgraph(comp).nodes()) for comp in Gcc ]
    
    return components


def get_core_border(G, region):
    region = set(region)
    ## identify core points
    core = set([])
    for point in region:
        if not ( set(list(G[point])) - region ):
            core.add(point)
    ## get border from core
    border = get_neighbors(G, core)
    
    return core, border



def get_nn(rtree, lat, lon):
    res = list(rtree.nearest([lon, lat], 1))
    return res[0]


## gets a list of coordinates, returns a list of coordinates for the nearest point in dataset
def snap_to_data(indices, points):
    df, rtree, G = indices
    snapped_points = []
    for point in points:
        point_id = get_nn(rtree, *point)
        snapped_point = (df.loc[point_id]['lat'], df.loc[point_id]['lon'])
        snapped_points.append(snapped_point)
    
    return snapped_points
        
        

def id_to_loc(df, point_id):
    return [df.loc[point_id]['lat'], df.loc[point_id]['lon']]



# Map and Figure Functions

# Bounding box of all coordinates in a data frame
def bbox(df):
    min_x = df['lat'].min()
    min_y = df['lon'].min()
    max_x = df['lat'].max()
    max_y = df['lon'].max()

    return min_x, min_y, max_x, max_y


# Filter locations in a data frame within a rectangle
def filterbbox(df, min_lon, min_lat, max_lon, max_lat):
    
    df = df.loc[df['lon'] >= min_lon]
    df = df.loc[df['lon'] <= max_lon]
    df = df.loc[df['lat'] >= min_lat]
    df = df.loc[df['lat'] <= max_lat]
    
    return df



def show_progressive(logger, G, params, types):
    result = logger.get_top1_result()
    # print(result)
    region = result.data['region']
    score, _ = get_score_of_region(G, region, params, types)
    print(f"result: score {score} with {len(region)} points")
    # print(f'top-1 region: {logger.get_top1_score()}')
    updates_df = pd.DataFrame.from_dict(logger.updates['time'], orient='index', columns=['score'])
    ax = updates_df.plot(y='score', logx=True)

# Logger
class PrioritizedObject:
    def __init__(self, data):
        self.data = data
   
    def __lt__(self, other):
        return self.data['score'] < other.data['score']
    
    def __str__(self):
        return f'{self.data}'


class Result_Logger(object):

    def __init__(self, topk=1, start_time=-1, w_history=False):
        self.topk = topk
        self.w_history = w_history
        self.reset(start_time)
    
    def reset(self, start_time):
        self.results = []
        self.iterations = 0
        if start_time == -1:
            self.start_time = time.time()
        else:
            self.start_time = start_time
        self.updates = {}
        self.updates['time'] = {}
        self.updates['iterations'] = {}
        self.history = []


    ## logs an entry specified by the dictionary data
    ## data *must* contain a key 'score', and can contain any other piece of information
    def log(self, **data):
        self.iterations += 1
        # if self.w_history:
        #     self.history.append(data)

        if len(self.results) < self.topk or data['score'] > heapq.nsmallest(1,self.results)[0].data['score']:
            ## new result
            heapq.heappush(self.results, PrioritizedObject(data))
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



## Search At Point


def greedy_search_best_at(G, seed, params, types, n_expansions=-1, loggers=[]):
    max_size = params['variables']['max_size']['current']
    if n_expansions == -1:
        n_expansions = max_size
    
    region = set([seed])

    region = extend_with_border(G, region)
    score, init = get_score_of_region(G, region, params, types)

    # local_logger.log(score=score, region=region, init=init, seed=seed, depth=0)
    for logger in loggers:
        logger.log(score=score, region=region, init=init, seed=seed, depth=0)

    size = 1
    depth = 0
    overall_best_score = score
    overall_best_region = region
    while size < max_size and depth < n_expansions:
        depth += 1

        core, border = get_core_border(G, region)
        if len(region) == 1: ## first point
            border = get_neighbors(G, region)

        neighbors = border
        best_score = 0
        best_neighbor = None
        for v in neighbors:
            new_nodes = set([n for n in list(G[v]) if n not in region])
            new_nodes.add(v)

            new_region = region | new_nodes ## union
            
            if len(new_region) > max_size:
                continue 

            new_score, new_init = get_score_of_region(G, new_region, params, types)
            for logger in loggers:
                logger.log(score=new_score, region=new_region, init=new_init, seed=seed, depth=depth)
            if new_score > best_score:
                best_score = new_score
                best_neighbor = v
        if best_neighbor:
            new_nodes = set([n for n in list(G[best_neighbor]) if n not in region])
            new_nodes.add(best_neighbor)
            region = region | new_nodes ## union
            size = len(region)
            score, init = get_score_of_region(G, region, params, types)
            assert(score == best_score)
            if score > overall_best_score:
                overall_best_score = score
                overall_best_region = region
            for logger in loggers:
                logger.log(score=new_score, region=new_region, init=new_init, seed=seed, depth=depth)
        else:
            break
    
    return overall_best_score, overall_best_region


def greedy_search_at(G, seed, params, types, n_expansions=-1, loggers=[], mode='best'):
    if n_expansions == -1:
        n_expansions = params['variables']['max_size']['current']
    if mode == 'best':
        return greedy_search_best_at(G, seed, params, types, n_expansions, loggers)
    elif mode == 'all':
        return greedy_search_all_at(G, seed, params, types, n_expansions, loggers)
    


def greedy_search_all_at(G, seed, params, types, n_expansions=-1, loggers=[]):
    max_size = params['variables']['max_size']['current']
    if n_expansions == -1:
        n_expansions = max_size

    region = set([seed])

    region = extend_with_border(G, region)
    score, init = get_score_of_region(G, region, params, types)

    best_score = score
    best_region = region

    size = 1
    depth = 0
    while size < max_size and depth < n_expansions:
        region = extend_with_border(G, region)
        score, init = get_score_of_region(G, region, params, types)

        if score > best_score:
            best_score = score
            best_region = region
        for logger in loggers:
                logger.log(score=score, region=region, init=init, seed=seed, depth=depth)

        size = len(region)
        depth += 1
    
    score = best_score
    region = best_region
    
    return score, region



# Search In Area

def search_in(indices, search_space, params, types, explore_mode='border', loggers=[]):
    df, rtree, G = indices
    left = search_space[1]
    bottom = search_space[0]
    right = search_space[3]
    top = search_space[2]
    points = [n for n in rtree.intersection((left, bottom, right, top))]
    seed = random.sample(points, 1)[0]
    score, _ = greedy_search_at(G, seed, params, types, loggers=loggers, mode=explore_mode)
    return score



def random_search(indices, n_seeds, topk, params, types, search_space=None, sample_method='uni_data', explore_mode='border', loggers=[]):
    df, rtree, G = indices

    ## set search space
    if search_space is None:
        xmin, ymin, xmax, ymax = bbox(df)
        the_search_space = (xmin, ymin, xmax, ymax)
    
    ## create seed points
    seeds = []
    if search_space is not None and sample_method == 'uni_data':
        left = the_search_space[1]
        bottom = the_search_space[0]
        right = the_search_space[3]
        top = the_search_space[2]
        points = [n for n in rtree.intersection((left, bottom, right, top))]
        seeds = random.sample(points, n_seeds)
    elif search_space is None and sample_method == 'uni_data':
        seeds = random.sample(list(G.nodes()), n_seeds)
    
    seeds = random.sample(list(G.nodes()), n_seeds)

    ## track the topk seeds found
    this_logger = Result_Logger(topk=topk)
    loggers.append(this_logger)


    ## search
    for seed in seeds:
        greedy_search_at(G, seed, params, types, loggers=loggers, mode=explore_mode)



# Grid class

class Grid:

    def __init__(self, rtree, df, gran, xmin, ymin, xmax, ymax):
        self.gran = gran ## 10x10 grid

        self.rtree = rtree
        self.df = df
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

        # square cell size (picking the max value in the two axes)
        self.cellSize = max((xmax - xmin)/gran, (ymax - ymin)/gran)
        
        self.build_grid_from_rtree()

    
    def build_grid_from_rtree(self):
        left = self.ymin
        bottom = self.xmin
        right = self.ymax
        top = self.xmax
        self.the_grid = dict()
        self.non_empty_cells = []
        for cell_x, cell_y in product(range(self.gran), range(self.gran)):
            left = self.ymin + cell_y * self.cellSize
            bottom = self.xmin + cell_x * self.cellSize
            right = self.ymin + (cell_y + 1) * self.cellSize
            top = self.xmin + (cell_x + 1) * self.cellSize

            cell_points = [n for n in self.rtree.intersection((left, bottom, right, top))]

            if cell_points:
                self.non_empty_cells.append( (cell_x, cell_y) )

            self.the_grid[(cell_x, cell_y)] = cell_points
    
    
    def build_grid_from_df(self):
        self.the_grid = dict()
        self.non_empty_cells = []
        for cell_x, cell_y in product(range(self.gran), range(self.gran)):
            self.the_grid[(cell_x, cell_y)] = []
        for idx, row in self.df.iterrows():
            cell = self.get_cell(row['lat'], row['lon'])
            self.the_grid[cell].append(idx)

        for cell_x, cell_y in product(range(self.gran), range(self.gran)):
            if self.the_grid[(cell_x, cell_y)]:
                self.non_empty_cells.append( (cell_x, cell_y) )


    def get_cell_coords(self, cell_x, cell_y):
        cell_xmin = self.xmin + cell_x * self.cellSize
        cell_ymin = self.ymin + cell_y * self.cellSize
        cell_xmax = self.xmin + (cell_x + 1) * self.cellSize
        cell_ymax = self.ymin + (cell_y + 1) * self.cellSize
        return cell_xmin, cell_ymin, cell_xmax, cell_ymax



    def get_random_seed_in_cell(self, cell_x, cell_y):
        points = self.the_grid[(cell_x, cell_y)]
        seed = None
        if points:
            seed = random.choice(points)
        return seed


    def get_cell(self, x, y):
        if x < self.xmin or x > self.xmax or y < self.ymin or y > self.ymax:
            return None
        cell_x = int( (x-self.xmin) // self.cellSize )
        cell_y = int( (y-self.ymin) // self.cellSize )
        return ( cell_x, cell_y )


class Grid_Search:

    def __init__(self, gran, indices, params, types, start_time, space_coords=None):
        self.df, self.rtree, self.G = indices
        self.params = params
        self.types = types
        self.start_time = start_time

        ## construct grid
        if space_coords is None: ## entire space
            self.xmin, self.ymin, self.xmax, self.ymax = bbox(self.df)
            space_coords = (self.xmin, self.ymin, self.xmax, self.ymax)
            
        self.grid = Grid(self.rtree, self.df, gran, *space_coords)
        

        ## find non empty cells
        self.non_empty_cells = self.grid.non_empty_cells
        self.n_non_empty_cells = len(self.non_empty_cells)

        ## map from cell-id to cell coords
        self.cell_id_to_cell = dict()
        for cell_id, cell in enumerate(self.non_empty_cells):
            self.cell_id_to_cell[cell_id] = cell

        ## initialize the history per cell; { cell: [ (seed, (coords), score) ] }
        self.grid_history = dict()
        for i in range(self.grid.gran):
            for j in range(self.grid.gran):
                self.grid_history[(i,j)] = []
    

    def search(self, n_samples, explore_mode, w_sample=True, loggers=[]):
        best_score = 0
        if w_sample: ## sample cells
            for i in range(n_samples):
                if (time.time() - self.start_time) > self.params['variables']['time_budget']['current']:
                    break
                cell_id = random.randint(0, self.n_non_empty_cells-1)
                cell = self.cell_id_to_cell[cell_id]
                
                while (seed := self.grid.get_random_seed_in_cell(*cell)) is None:
                    pass
                
                score, _ = greedy_search_at(self.G, seed, self.params, self.types, loggers=loggers, mode=explore_mode)
                if score > best_score:
                    best_score = score
                self.grid_history[cell].append( (seed, id_to_loc(self.df, seed), score) )
        else: 
            for cell in self.non_empty_cells[:n_samples]:
                if (time.time() - self.start_time) > self.params['variables']['time_budget']['current']:
                    break
                
                seed = self.grid.get_random_seed_in_cell(*cell)
                score, _ = greedy_search_at(self.G, seed, self.params, self.types, loggers=loggers, mode=explore_mode)
                if score > best_score:
                    best_score = score
                self.grid_history[cell].append( (seed, id_to_loc(self.df, seed), score) )
        return best_score


    def get_top_k(self, top_k):
        cell_max_scores = []
        for cell in self.grid_history.keys():
            if not self.grid_history[cell]:
                continue
            max_item = max(self.grid_history[cell], key=lambda tup: tup[2])
            cell_max_scores.append( (cell, max_item[0], max_item[2]) )
        
        cell_max_scores = sorted(cell_max_scores, key=lambda tup: tup[2], reverse=True)

        return cell_max_scores[:top_k]



class Grid_Search_Simple:

    def __init__(self, gran, indices, params, types, start_time, space_coords=None):
        self.df, self.rtree, self.G = indices
        self.params = params
        self.types = types
        self.start_time = start_time

        ## construct grid
        if space_coords is None: ## entire space
            self.xmin, self.ymin, self.xmax, self.ymax = bbox(self.df)
            space_coords = (self.xmin, self.ymin, self.xmax, self.ymax)
            
        self.grid = Grid(self.rtree, self.df, gran, *space_coords)
        

        ## find non empty cells
        self.active_cells = self.grid.non_empty_cells

        ## map from cell-id to cell coords
        self.cell_id_to_cell = dict()
        for cell_id, cell in enumerate(self.active_cells):
            self.cell_id_to_cell[cell_id] = cell

        ## initialize the history per cell; { cell: [ (seed, (coords), score) ] }
        self.grid_history = dict()
        for i in range(self.grid.gran):
            for j in range(self.grid.gran):
                self.grid_history[(i,j)] = []
    

    def set_active_cells(self, n_cells, explore_mode, loggers=[]):
        new_active_cells = []
        for cell in self.active_cells:
            if (time.time() - self.start_time) > self.params['variables']['time_budget']['current']:
                break
            seed = self.grid.get_random_seed_in_cell(*cell)
            score, _ = greedy_search_at(self.G, seed, self.params, self.types, loggers=loggers, mode=explore_mode)
            # print(cell, seed, score)
            self.grid_history[cell].append( (seed, id_to_loc(self.df, seed), score) )
        
        cell_max_scores = []
        for cell in self.active_cells:
            if not self.grid_history[cell]:
                continue
            max_item = max(self.grid_history[cell], key=lambda tup: tup[2])
            cell_max_scores.append( (cell, max_item[0], max_item[2]) ) # cell, seed, score
        cell_max_scores = sorted(cell_max_scores, key=lambda tup: tup[2], reverse=True)
        
        new_active_cells = [ tup[0] for tup in cell_max_scores[:n_cells] ]
        
        self.active_cells = new_active_cells

    
    def search_cell(self, cell, n_samples, explore_mode, loggers=[]):
        for i in range(n_samples):
            if (time.time() - self.start_time) > self.params['variables']['time_budget']['current']:
                break
            while (seed := self.grid.get_random_seed_in_cell(*cell)) is None:
                pass
            score, _ = greedy_search_at(self.G, seed, self.params, self.types, loggers=loggers, mode=explore_mode)
            self.grid_history[cell].append( (seed, id_to_loc(self.df, seed), score) )

    
    def pick_cell(self):
        return random.choice(self.active_cells)



class Meta_Grid_Search:

    def __init__(self, gran, indices, params, types, start_time, space_coords_s, w_bayes=False):
        self.df, self.rtree, self.G = indices
        self.params = params
        self.types = types
        self.start_time = start_time

        self.n_grids = len(space_coords_s)
        self.w_bayes = w_bayes

        ## Bayes Initialization
        if self.w_bayes:
            dimensions = [Categorical(range(self.n_grids))]
            kernel = 1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-1, 10.0),
                                nu=1.5)
            gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-10,
                                        normalize_y=True, noise="gaussian",
                                        n_restarts_optimizer=2)
            self.opt = Optimizer(dimensions, base_estimator=gpr, acq_optimizer="sampling", 
                                    n_initial_points = 10)


        self.grid_searches = []
        for space_coords in space_coords_s:
            grid_search = Grid_Search(gran, indices, params, types, start_time, space_coords=space_coords)
            self.grid_searches.append(grid_search)
    
    
    def search(self, n_samples, explore_mode, loggers=[]):
        if self.w_bayes:
            n_samples_per_probe = 10
            for i in range (int(n_samples / n_samples_per_probe)):
                if (time.time() - self.start_time) > self.params['variables']['time_budget']['current']:
                    break
                grid_id = self.opt.ask()[0] ## get a suggestion; NOTE: returns a list

                score = self.grid_searches[grid_id].search(n_samples_per_probe, explore_mode, loggers=loggers)
                self.opt.tell([grid_id], -score) ## learn from a suggestion; NOTE: requires a list as x
        
        else: ## no bayes
            for i in range(n_samples):
                if (time.time() - self.start_time) > self.params['variables']['time_budget']['current']:
                    break
                grid_id = random.randint(0, self.n_grids-1)
                self.grid_searches[grid_id].search(1, explore_mode, loggers=loggers)
    
    

    def get_top_1(self):
        grid_max_scores = []
        for grid_id in range(self.n_grids):
            if tup := self.grid_searches[grid_id].get_top_k(1)[0]:
                grid_max_score = tup[2]
                grid_max_scores.append(grid_max_score)
        return max(grid_max_scores)

            

def run_method(df, G, rtree, types, params, start_time):
    indices = (df, rtree, G)

    top_mode = params['grid']['top_mode']
    n_top_samples = params['grid']['n_top_samples']
    top_gran = params['grid']['top_gran']
    bot_gran = params['grid']['bot_gran']
    n_grids = params['grid']['n_grids']
    n_bot_samples = params['grid']['n_bot_samples']
    if 'w_bayes' in params['grid'].keys():
        w_bayes = params['grid']['w_bayes']
    else:
        w_bayes = False

    logger = Result_Logger(start_time=start_time)
    gs = Grid_Search(top_gran, indices, params, types, start_time)


    gs.search(n_top_samples, top_mode, w_sample=True, loggers=[logger])
    
    if n_grids != 0:
        space_coords_s = [ gs.grid.get_cell_coords(*tup[0]) for tup in gs.get_top_k(n_grids) ]

        # print('bottom')
        mgs = Meta_Grid_Search(bot_gran, indices, params, types, start_time, space_coords_s, w_bayes)

        mgs.search(n_bot_samples, 'best', loggers=[logger])

    result = logger.get_top1_result()

    return result.data['score'], result.data['region'], logger.updates['time'], logger



def run_method_simple(df, G, rtree, types, params, start_time):
    indices = (df, rtree, G)

    top_mode = params['grid']['top_mode']
    top_gran = params['grid']['gran']
    n_grids = params['grid']['n_cells']
    n_bot_samples = params['grid']['n_bot_samples']

    logger = Result_Logger(start_time=start_time)
    gs = Grid_Search_Simple(top_gran, indices, params, types, start_time)

    gs.set_active_cells(n_grids, explore_mode=top_mode, loggers=[logger])

    n_samples_per_cell = 10
    for i in range (int(n_bot_samples / n_samples_per_cell)):
        cell = gs.pick_cell()
        gs.search_cell(cell, n_samples_per_cell, explore_mode='best', loggers=[logger])
    
    result = logger.get_top1_result()

    return result.data['score'], result.data['region'], logger.updates['time'], logger
