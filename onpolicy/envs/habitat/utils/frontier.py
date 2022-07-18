import numpy as np
from queue import deque
import os
import sys
import torch

from onpolicy.utils.RRTStar.rrt_star import RRTStar
from onpolicy.utils.RRT.rrt import RRT, WMA_RRT
import matplotlib.pyplot as plt
import pickle
import skimage.morphology

def l2distance(a,b):
    return pow(pow(a[0]-b[0],2)+pow(a[1]-b[1],2),0.5)

def add_clear_disk(map, unexplored, r, loc):
    map = map.copy()
    unexplored = unexplored.copy()
    map[map==2] = 0
    map[unexplored == 1] = 3
    H, W = map.shape
    for x in range(H):
        for y in range(W):
            if l2distance((x,y), loc) <= r and map[x,y] == 3:
                map[x,y] = 0
                unexplored[x,y] = 0
    steps = [(0,1),(0,-1),(1,0),(-1,0)]
    for x in range(H):
        for y in range(W):
            if map[x,y] == 0:
                neighbors = [(x+dx, y+dy) for dx, dy in steps]
                if sum([(unexplored[u,v]==1) for u,v in neighbors])>0 and sum([map[u,v] == 1 for u,v in neighbors]) == 0: # neighbors are unexplored and no walls
                    map[x,y] = 2 # 2 for targets(frontiers)
    unexplored = (map == 3).astype(np.uint8)
    map[unexplored == 1] = 0
    return map, unexplored

def get_boundary(map, v):
    row = ((map==v).astype(np.int32).sum(1)>0).astype(np.int32).tolist()
    if 1 not in row:
        return map.shape[0], 0, map.shape[1], 0
    x1 = row.index(1)
    x2 = map.shape[0] - 1 - list(reversed(row)).index(1)

    col = ((map==v).astype(np.int32).sum(0)>0).astype(np.int32).tolist()
    y1 = col.index(1)
    y2 = map.shape[1] - 1 - list(reversed(col)).index(1)

    return x1, x2, y1, y2

def get_frontier(obstacle, explored, locations, cut_boundary=True):
    explored[obstacle == 1] = 1
    H, W = explored.shape
    steps = [(-1,0),(1,0),(0,-1),(0,1)]
    map = np.ones((H, W)).astype(np.int32) * 3 # 3 for unknown area 
    map[explored == 1] = 0
    map[obstacle == 1] = 1 # 1 for obstacles 
    num_agents = len(locations)
    # frontier
    lx, rx, ly, ry = 1e9, 0, 1e9, 0
    # boundary
    map[0, :] = 1
    map[H-1, :] = 1
    map[:, 0] = 1
    map[:, W-1] = 1
    que = deque([(1,1)])
    pad = 10
    
    def fix_boundary(lx, rx, ly, ry, H, W):    
        lx = max(lx, 0)
        rx = min(rx, H-1)
        ly = max(ly, 0)
        ry = min(ry, W-1)
        return lx, rx, ly, ry

    x1, x2, y1, y2 = get_boundary(map, 0)
    lx = min(lx, x1)
    rx = max(rx, x2)
    ly = min(ly, y1)
    ry = max(ry, y2)
    x1, x2, y1, y2 = get_boundary(explored, 1)
    lx = min(lx, x1 - pad)
    rx = max(rx, x2 + pad)
    ly = min(ly, y1 - pad)
    ry = max(ry, y2 + pad)
    
    lx, rx, ly, ry = fix_boundary(lx, rx, ly, ry, H, W)

    for agent_id in range(num_agents):
        lx = min(lx, locations[agent_id][0]-pad)
        ly = min(ly, locations[agent_id][1]-pad)
        rx = max(rx, locations[agent_id][0]+pad)
        ry = max(ry, locations[agent_id][1]+pad)

    for x in range(lx, rx+1):
        for y in range(ly, ry+1):
            if map[x,y] == 0:
                neighbors = [(x+dx, y+dy) for dx, dy in steps]
                if sum([(map[u,v] == 3) for u,v in neighbors])>0 and sum([map[u,v] == 1 for u,v in neighbors]) == 0: # neighbors are unexplored and no walls
                    map[x,y] = 2 # 2 for targets(frontiers)
    unexplored = (map == 3).astype(np.int8)
    map[map == 3] = 0 # set unkown area to obstacles

    x0, y0 = int((lx+rx)/2), int((ly+ry)/2)
    
    lx = min(lx, x0-40)
    rx = max(rx, x0+40)
    ly = min(ly, y0-40)
    ry = max(ry, y0+40)
    
    lx, rx, ly, ry = fix_boundary(lx, rx, ly, ry, H, W)

    unexplored[map == 1] = 0
    if not cut_boundary:
        lx = ly = 0
        rx, ry = H-1, W-1
    else:
        map[lx, :] = 1
        map[rx, :] = 1
        map[:, ly] = 1
        map[:, ry] = 1
    return map[lx:rx+1, ly:ry+1], (lx, ly), unexplored[lx:rx+1, ly:ry+1]

def get_frontier_cluster(frontiers, cluster_radius = 5.0, cluster_size = 5):
    if len(frontiers) == 0:
        return []
    num_frontier = len(frontiers)
    clusters = []
    # valid = [True for _ in range(num_frontier)]
    H = max([x for (x,y) in frontiers]) + 1
    W = max([y for (x,y) in frontiers]) + 1
    valid = np.zeros((H,W), dtype=np.uint8)
    for x,y in frontiers:
        valid[x,y] = 1
    cluster_radius = int(cluster_radius)
    for i in range(num_frontier):
        if valid[frontiers[i][0], frontiers[i][1]] == 1:
            neigh = []
            lx, rx, ly, ry = frontiers[i][0]-cluster_radius, frontiers[i][0]+cluster_radius, frontiers[i][1]-cluster_radius, frontiers[i][1]+cluster_radius
            lx = max(lx, 0)
            rx = min(rx, H-1)
            ly = max(ly, 0)
            ry = min(ry, W-1)
            for x in range(lx, rx+1):
                for y in range(ly, ry+1):
                    if valid[x,y] == 1:
                        valid[x,y] == 0
                        neigh.append((x,y))
            center = None
            min_r = 1e9
            for p in neigh:
                r = max([l2distance(p,q) for q in neigh])
                if r<min_r:
                    min_r = r
                    center = p
            if len(neigh) >= cluster_size:
                clusters.append({'center': center, 'weight': len(neigh)})
    return clusters

def nearest_frontier(map, unexplored, locations, steps, agent_id, clear_radius = 40, cluster_radius = 5, random_goal = False):
    map, unexplored = add_clear_disk(map, unexplored, clear_radius, locations[agent_id])
    H, W = map.shape
    que = deque([locations[agent_id]])
    vis = np.zeros((H, W), dtype=np.int8)
    dis = np.zeros((H, W), dtype=np.int32)
    vis[locations[agent_id][0], locations[agent_id][1]] = 1
    while len(que)>0:
        x, y = que.popleft()
        neighbors = [(x+dx, y+dy) for dx, dy in steps]
        for u,v in neighbors:
            if map[u,v] in [0,2] and vis[u,v] == 0:
                vis[u,v] = 1
                dis[u,v] = dis[x,y] + 1
                que.append((u,v))
    min_dis = 1e9
    min_x, min_y = None, None
    frontiers = []
    for x in range(H):
        for y in range(W):
            if map[x,y] == 2 and l2distance((x,y), locations[agent_id]) > clear_radius:
                #if min_dis > dis[x,y]:
                #    min_dis, min_x, min_y = dis[x,y], x, y
                frontiers.append((x,y))
    clusters = get_frontier_cluster(frontiers, cluster_radius=cluster_radius)
    for cluster in clusters:
        p = cluster['center']
        d = l2distance(p, locations[agent_id])
        if d > clear_radius:
            if min_dis > dis[p[0], p[1]]:
                min_x, min_y = p
                min_dis = dis[p[0], p[1]]
    if min_x == None:
        # no valid target
        if random_goal:
            x, y = np.random.randint(0, H), np.random.randint(0, W)
            while map[x,y] == 1:
                x, y = np.random.randint(0, H), np.random.randint(0, W)
            min_x, min_y = x, y
        else:
            min_x, min_y = locations[agent_id][0], locations[agent_id][1]
    return min_x, min_y

def circle_matrix(H, W, p, radius):
    mat = np.zeros((H, W), dtype = np.int32)
    for x in range(H):
        for y in range(W):
            if l2distance((x,y), p)<= radius:
                mat[x,y] = 1
    return mat

def bfs_distance(map, lx, ly, start, goals):
    if lx == None:
        return 1e6
    H, W = map.shape
    sx, sy = start[0], start[1]
    sx -= lx
    sy -= ly
    '''if np.array(goals).size == 2:
        goals = [goals]'''
    if goals is not None:
        goals = [(max(0,min(x - lx, H-1)), max(0, min(y - ly, W-1)) ) for x, y in goals]
        num_goals = len(goals)
    # print(gx, gy)
    if sx < 0 or sy < 0 or sx >= H or sy >= W:
        return 1e6

    dis = np.zeros((H, W), dtype = np.int32)
    dis[sx, sy] = 1
    steps = [(-1,0),(1,0),(0,-1),(0,1)]

    que = deque([(sx, sy)])
    while len(que)>0:
        x, y = que.popleft()
        neighbors = [(x+dx, y+dy) for dx, dy in steps]
        neighbors = [(x, y) for x,y in neighbors if x>=0 and x<H and y>=0 and y<W]
        for u,v in neighbors:
            if map[u,v] in [0,2] and dis[u,v] == 0:
                dis[u,v] = dis[x,y] + 1
                que.append((u,v))
    dis[dis == 0] = 1e6
    if goals is None:
        return dis
    for i, (gx, gy) in enumerate(goals):
        if map[gx,gy] == 1:
            # if the goal is obstacle, find nearest valid cell
            tar = (gx, gy)
            min_dis = 1e9
            for x in range(H):
                for y in range(W):
                    if map[x,y] in [0,2]:
                        tmp = l2distance((x,y), (gx, gy))
                        if tmp<min_dis:
                            min_dis = tmp
                            tar = (x,y)
            goals[i] = tar
    ret = [dis[gx, gy]-1 for gx, gy in goals]
    if num_goals == 1:
        return ret[0]
    return np.array(ret)

def max_utility_frontier(map, unexplored, locations,  clear_radius = 40, cluster_radius = 5, utility_radius = 50, pre_goals = None, goal_mask = None, random_goal = False):
    H, W = map.shape
    num_agents = len(locations)

    order = np.arange(num_agents)
    np.random.shuffle(order)
    unexplored = unexplored.copy().astype(np.int32)
    goals = np.zeros((num_agents, 2), dtype = np.int32)

    # masked agents
    if goal_mask == None:
        goal_mask = [False for _ in range(num_agents)]
    else:
        for agent_id in range(num_agents):
            if goal_mask[agent_id]:
                mat = circle_matrix(H, W, pre_goals[agent_id], utility_radius)
                unexplored[mat == 1] = 0

    # print("unexplored", unexplored.sum())
    o_map = map.copy()
    o_unexplored = unexplored.copy()
    for agent_id in order:
        if goal_mask[agent_id]:
            goals[agent_id] = pre_goals[agent_id]
            continue
        map, unexplored = add_clear_disk(map, unexplored, clear_radius, locations[agent_id])
        frontiers = []
        for x in range(H):
            for y in range(W):
                if map[x,y] == 2:
                    frontiers.append((x,y))
        clusters = get_frontier_cluster(frontiers, cluster_radius = cluster_radius)
        num_clusters = len(clusters)
        # compute utility
        max_utility = -1.0
        tar = None
        for it, cluster in enumerate(clusters):
            p = cluster['center']

            if max([l2distance(p, locations[i]) for i in range(num_agents)]) <= clear_radius:
                continue

            mat = circle_matrix(H, W, p, utility_radius)
            
            tmp = unexplored[mat == 1].sum()
            if tmp>max_utility:
                max_utility = tmp
                tar = p
        if tar == None:
            if random_goal:
                x, y = np.random.randint(0, H), np.random.randint(0, W)
                while map[x,y] == 1:
                    x, y = np.random.randint(0, H), np.random.randint(0, W)
                tar = (x,y)
            else:
                tar = (locations[agent_id][0], locations[agent_id][1])
        goals[agent_id] = np.array(tar)
        # print(locations[agent_id], tar, l2distance(locations[agent_id], tar), unexplored.sum(), max_utility)
        mat = circle_matrix(H, W, tar, utility_radius)
        unexplored = o_unexplored
        unexplored[mat == 1] = 0
    # re-allocate goals ?
    dist = np.zeros((num_agents, num_agents), dtype=np.int32)
    for i in range(num_agents):
        dist[i] = bfs_distance(map, 0, 0, locations[agent_id], goals)
    tar = np.arange(num_agents)
    for T in range(10):
        for i in range(num_agents):
            for j in range(num_agents):
                if i == j:
                    continue
                if (goal_mask[i] or l2distance(locations[i], goals[tar[j]]) > clear_radius) and (goal_mask[j] or l2distance(locations[j], goals[tar[i]]) > clear_radius) and dist[i, tar[j]] < dist[i, tar[i]] and dist[j,tar[i]] < dist[j,tar[j]]:
                    tmp = tar[i]
                    tar[i] = tar[j]
                    tar[j] = tmp
    ret = goals.copy()
    for i in range(num_agents):
        ret[i] = goals[tar[i]]
    return ret


def voronoi_based_planning(map, unexplored, locations, clear_radius = 40, cluster_radius = 5, utility_radius = 50, pre_goals = None, goal_mask = None, random_goal = False):
    H, W = map.shape
    num_agents = len(locations)

    o_unexplored = unexplored.copy()
    feasible = unexplored.copy()
    goals = np.zeros((num_agents, 2), dtype = np.int32)

    # masked agents
    if goal_mask == None:
        goal_mask = [False for _ in range(num_agents)]
    else:
        for agent_id in range(num_agents):
            if goal_mask[agent_id]:
                mat = circle_matrix(H, W, pre_goals[agent_id], utility_radius)
                feasible[mat == 1] = 0

    dist = np.zeros((num_agents, H, W), dtype=np.float32)
    o_map = map.copy()
    for i in range(num_agents):
        dist[i] = bfs_distance(map, 0, 0, locations[i], None)
    for agent_id in range(num_agents):
        if goal_mask[agent_id]:
            goals[agent_id] = pre_goals[agent_id]
            continue
        my_grids = np.ones_like(unexplored)
        for j in range(num_agents):
            if agent_id != j:
                my_grids[dist[agent_id] >= dist[j]] = 0.

        frontiers = []
        for x in range(H):
            for y in range(W):
                if map[x,y] == 2 and my_grids[x, y] == 1:
                    frontiers.append((x,y))
        clusters = get_frontier_cluster(frontiers, cluster_radius = cluster_radius)
        num_clusters = len(clusters)
        if num_clusters == 0:
            tar = None
        else:
            # compute utility
            max_utility = -1e9
            tar = None
            centers = [cluster['center'] for cluster in clusters]
            distance = np.array([dist[i][x, y] for x, y in centers]) # bfs_distance(map, 0, 0, locations[agent_id], centers)
            distance = distance / max(distance)
            for it, p in enumerate(centers):

                if l2distance(p, locations[agent_id]) <= clear_radius:
                    continue

                mat = circle_matrix(H, W, p, utility_radius)
                
                tmp = feasible[mat == 1].sum() / (utility_radius ** 2) / np.pi
                tmp = tmp - distance[it] * 1.
                if tmp > max_utility:
                    max_utility = tmp
                    tar = p
        if tar == None:
            if random_goal:
                x, y = np.random.randint(0, H), np.random.randint(0, W)
                while map[x,y] == 1:
                    x, y = np.random.randint(0, H), np.random.randint(0, W)
                tar = (x,y)
            else:
                tar = (locations[agent_id][0], locations[agent_id][1])
        goals[agent_id] = np.array(tar)
        # print(locations[agent_id], tar, l2distance(locations[agent_id], tar), unexplored.sum(), max_utility)
        mat = circle_matrix(H, W, tar, utility_radius)
        feasible = o_unexplored
        feasible[mat == 1.] = 0.

        '''import imageio
        rgb = np.zeros((H, W, 3), dtype=np.uint8)
        rgb[:, :, 2][my_grids.cpu().numpy() == 1.] += 100
        rgb[:, :, 1][feasible == 1.] += 100
        mat = circle_matrix(H, W, tar, 3)
        rgb[:, :, 0][mat == 1.] += 100

        mat = circle_matrix(H, W, locations[agent_id], 3)
        for c in range(3):
            rgb[:, :, c][mat == 1.] += 100
        imageio.imwrite(f"gjx/debug_{agent_id}.png", rgb)
        rgb = np.zeros((H, W, 3), dtype=np.uint8)
        rgb[:, :, 2][my_grids.cpu().numpy() == 1.] += 100
        imageio.imwrite(f"gjx/grids_{agent_id}.png", rgb)'''

    dist = np.zeros((num_agents, num_agents), dtype=np.int32)
    for i in range(num_agents):
        dist[i] = bfs_distance(map, 0, 0, locations[agent_id], goals)
    tar = np.arange(num_agents)
    for T in range(10):
        for i in range(num_agents):
            for j in range(num_agents):
                if i == j:
                    continue
                if (not goal_mask[i] and l2distance(locations[i], goals[tar[j]]) > clear_radius) and (not goal_mask[j] and l2distance(locations[j], goals[tar[i]]) > clear_radius) and dist[i, tar[j]] < dist[i, tar[i]] and dist[j,tar[i]] < dist[j,tar[j]]:
                    tmp = tar[i]
                    tar[i] = tar[j]
                    tar[j] = tmp
    ret = goals.copy()
    for i in range(num_agents):
        ret[i] = goals[tar[i]]

    return ret

def find_rectangle_obstacles(map):
    map = map.copy().astype(np.int32)
    map[map == 2] = 0
    H, W = map.shape
    obstacles = []
    covered = np.zeros((H, W), dtype = np.int32)
    pad = 0.01
    for x in range(H):
        for y in range(W):
            if map[x,y] == 1 and covered[x,y] == 0:
                x1 = x
                x2 = x
                while x2 < H-1 and map[x2 + 1, y] == 1:
                    x2 = x2 + 1
                y1 = y
                y2 = y
                while y2 < W-1 and map[x1 : x2+1, y2 + 1].sum() == x2-x1+1:
                    y2 = y2 + 1
                covered[x1 : x2 + 1, y1 : y2 + 1] = 1
                obstacles.append((x1-pad, y1-pad, x2 + 1 + pad, y2 + 1 + pad))    
    return obstacles

def rrt_global_plan(map, unexplored, locations, agent_id, clear_radius = 40, cluster_radius = 5, step = 0, utility_radius = 50, random_goal = False, sections = None, get_farthest = False, return_rrt=False, return_success = False, return_targets = False, rrt_iterations=2000):
    map, unexplored = add_clear_disk(map, unexplored, clear_radius, locations[agent_id])
    H, W = map.shape
    map = map.astype(np.int32)
    loc = (locations[agent_id][0], locations[agent_id][1])
    map[loc[0] - 2: loc[0] + 3, loc[1] - 2 : loc[1] + 3] = 0

    # greedily assemble obstacles into rectangles to reduce the number of obstacles
    obstacles = find_rectangle_obstacles(map)
    
    # print("num of obstacles", len(obstacles))

    rrt = RRT(start=(loc[0] + 0.5, loc[1] + 0.5),
        goals=[],
        rand_area=((0, H), (0, W)),
        obstacle_list=obstacles,
        expand_dis=5.0 ,
                 goal_sample_rate=-1,
                 max_iter=rrt_iterations)
    rrt.set_obs(map, unexplored)
    mat = circle_matrix(H, W, loc, clear_radius)
    rrt_map = unexplored.copy().astype(np.int32)
    rrt_map[mat == 1] = 0
    all_targets, success = rrt.select_frontiers(rrt_map, num_targets = 100, get_farthest=get_farthest, sections = sections)

    if sections is None:
        all_targets = [all_targets]
    
    goals = []

    for targets in all_targets:
        clusters = get_frontier_cluster(targets, cluster_radius = cluster_radius, cluster_size = 1)

        if len(clusters) == 0:
            if random_goal:
                x, y = np.random.randint(0, H), np.random.randint(0, W)
                while map[x,y] == 1:
                    x, y = np.random.randint(0, H), np.random.randint(0, W)
                goal = (x,y)
            else:
                goal = (loc[0], loc[1])
            goals.append(goal)
            continue

        for cluster in clusters:
            center = cluster['center']
            # navigation cost
            nav_cost = l2distance(center, loc)
            # information gain
            mat = circle_matrix(H, W, center, utility_radius)
            area = mat.sum()
            info_gain = rrt_map[mat == 1].sum()
            info_gain /= area
            cluster['info_gain'] = info_gain
            cluster['nav_cost'] = nav_cost
        D = max([cluster['nav_cost'] for cluster in clusters]) + 0.01
        goal = None
        mx = -1e9
        for cluster in clusters:
            cluster['nav_cost'] /= D
            cluster['utility'] = cluster['info_gain'] - 1.0 * cluster['nav_cost']
            if mx < cluster['utility']:
                mx = cluster['utility']
                goal = cluster['center']
        
        #if rrt_iterations>0:
        #    print("num of targets", len(targets))
        #    print("num of clusters", len(clusters))
        #    print("Max dist", D - 0.1)
        goals.append(goal)
    
    if sections is None:
        ret = [goals[0], ]
    else:
        ret = [goals, ]
    if return_targets:
        ret += [all_targets]
    if return_rrt:
        ret += [rrt]
    if return_success:
        ret += [success]
    return ret if len(ret)>1 else ret[0]

def dilate(a):
    map = (a == 1)
    H, W = map.shape
    ret = map.copy()
    ret[1:] = ret[1:] + map[:H-1]
    ret[:H-1] = ret[:H-1] + map[1:]
    ret[:, 1:] = ret[:, 1:] + map[:, :W-1]
    ret[:, :W-1] = ret[:, :W-1] + map[:, 1:]
    return ret.astype(np.int32)

def get_closest_frontier(map, start, goal):
    start = [int(start[0]), int(start[1])]
    goal = (int(goal[0]), int(goal[1]))

    obstacle = np.rint(map[0])
    explored = np.rint(map[1])

    if explored[goal[0], goal[1]] == 1:
        return goal

    H, W = explored.shape

    start[0] = max(0, min(start[0], H-1))
    start[1] = max(0, min(start[1], W-1))

    row = np.array([(i-start[0])**2 for i in range(H)]).repeat(W).reshape(H, W)
    col = np.array([(i-start[1])**2 for i in range(W)]).repeat(H).reshape(W, H).transpose()

    dist_map = row+col

    explored[dist_map <=50] = 1
    explored[explored > 1] = 1

    unexplored = (explored == 0).astype(np.int32)
    tmp = dilate(unexplored) * (1-obstacle)
    for _ in range(3):
        unexplored = tmp
        tmp = dilate(unexplored) * (1-obstacle)
    frontier_map = (tmp - unexplored) * (1 - obstacle)

    row = np.array([(i-goal[0])**2 for i in range(H)]).repeat(W).reshape(H, W)
    col = np.array([(i-goal[1])**2 for i in range(W)]).repeat(H).reshape(W, H).transpose()

    dist_map = row+col

    dist = dist_map * frontier_map + (1-frontier_map) * 1e9

    (index_a, index_b) = np.unravel_index(np.argmin(dist, axis=None), dist.shape)

    return (index_a, index_b)
