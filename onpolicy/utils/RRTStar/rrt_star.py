"""
Path planning Sample Code with RRT*
author: Atsushi Sakai(@Atsushi_twi)
"""

import math
import os
import sys
import random

import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../RRT/")

try:
    from rrt import RRT
except ImportError:
    raise
from onpolicy.utils.RRT.rrt import RRT

show_animation = False
show_path = True

def cmp_to_key(mycmp):
    'Convert a cmp= function into a key= function'
    class K(object):
        def __init__(self, obj, *args):
            self.obj = obj
        def __lt__(self, other):
            return mycmp(self.obj, other.obj) < 0
        def __gt__(self, other):
            return mycmp(self.obj, other.obj) > 0
        def __eq__(self, other):
            return mycmp(self.obj, other.obj) == 0
        def __le__(self, other):
            return mycmp(self.obj, other.obj) <= 0
        def __ge__(self, other):
            return mycmp(self.obj, other.obj) >= 0
        def __ne__(self, other):
            return mycmp(self.obj, other.obj) != 0
    return K

class RRTStar(RRT):
    """
    Class for RRT Star planning
    """

    class Node(RRT.Node):
        def __init__(self, x, y):
            super().__init__(x, y)
            self.cost = 0.0

    def __init__(self,
                 start,
                 goals,
                 obstacle_list,
                 rand_area,
                 expand_dis=30.0,
                 path_resolution=1.0,
                 goal_sample_rate=20,
                 max_iter=300,
                 connect_circle_dist=50.0,
                 search_until_max_iter=False):
        """
        Setting Parameter
        start:Start Position [x,y]
        goal:Goal Position [x,y]
        obstacleList:obstacle Positions [[x,y,size],...]
        randArea:Random Sampling Area [min,max]
        """
        super().__init__(start, goals, obstacle_list, rand_area, expand_dis,
                         path_resolution, goal_sample_rate, max_iter)
        self.connect_circle_dist = connect_circle_dist
        self.search_until_max_iter = search_until_max_iter

    def planning(self, animation=True, smooth_path = False, return_path = True):
        """
        rrt star path planning
        animation: flag for animation on or off .
        """

        self.node_list = [self.start]
        i = 0
        while True:
            i = i + 1
            if i > self.max_iter:
                # could not find node
                print("reached max iteration")
                min_dist = 1e9
                ret_node = None
                def cmp(x, y):
                    nx = self.node_list[x]
                    ny = self.node_list[y]
                    if nx.x < ny.x:
                        return -1
                    if nx.x > ny.x:
                        return 1
                    if nx.y < ny.y:
                        return -1
                    if nx.y > ny.y:
                        return 1
                    return 0
                idx = list(range(len(self.node_list)))
                idx = sorted(idx, key = cmp_to_key(cmp))
                '''for id in idx:
                    node = self.node_list[id]
                    print("x : %.3f, y: %.3f"%(node.x, node.y))'''
                for it, node in enumerate(self.node_list):
                    tar = self.get_nearest_goal(node.x, node.y)
                    dist = math.hypot(node.x - tar.x, node.y - tar.y)
                    # print("x : %.3f, y: %.3f, dist : %.3f"%(node.x, node.y, dist))
                    if min_dist > dist:
                        min_dist = dist
                        ret_node = it
                path = self.generate_final_course(ret_node, ignore_goal = True)
                if smooth_path:
                    path = self.smooth_path(path)
                return path if return_path else path[-1]
                
            # print("Iter:", i, ", number of nodes:", len(self.node_list))
            rnd = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd)
            new_node = self.steer(self.node_list[nearest_ind], rnd,
                                  self.expand_dis)
            near_node = self.node_list[nearest_ind]
            new_node.cost = near_node.cost + \
                math.hypot(new_node.x-near_node.x,
                           new_node.y-near_node.y)

            if self.check_collision(new_node, self.obstacle_list):
                near_inds = self.find_near_nodes(new_node)
                node_with_updated_parent = self.choose_parent(
                    new_node, near_inds)
                if node_with_updated_parent:
                    self.rewire(node_with_updated_parent, near_inds)
                    self.push_node(node_with_updated_parent)
                else:
                    self.push_node(new_node)

            if animation:
                self.draw_graph(rnd)

            if ((not self.search_until_max_iter)
                    and new_node):  # if reaches goal
                last_index = self.search_best_goal_node()
                if last_index is not None:
                    print("Iterations : %d"%i)
                    path = self.generate_final_course(last_index)
                    if smooth_path:
                        path = self.smooth_path(path)
                    return path if return_path else path[-1]
        
        # print("reached max iteration")

        last_index = self.search_best_goal_node()
        if last_index is not None:
            return self.generate_final_course(last_index)

        return None

    def choose_parent(self, new_node, near_inds):
        """
        Computes the cheapest point to new_node contained in the list
        near_inds and set such a node as the parent of new_node.
            Arguments:
            --------
                new_node, Node
                    randomly generated node with a path from its neared point
                    There are not coalitions between this node and th tree.
                near_inds: list
                    Indices of indices of the nodes what are near to new_node
            Returns.
            ------
                Node, a copy of new_node
        """
        if not near_inds:
            return None

        # search nearest cost in near_inds
        costs = []
        for i in near_inds:
            near_node = self.node_list[i]
            t_node = self.steer(near_node, new_node)
            if t_node and self.check_collision(t_node, self.obstacle_list):
                costs.append(self.calc_new_cost(near_node, new_node))
            else:
                costs.append(float("inf"))  # the cost of collision node
        min_cost = min(costs)

        if min_cost == float("inf"):
            print("There is no good path.(min_cost is inf)")
            return None

        min_ind = near_inds[costs.index(min_cost)]
        new_node = self.steer(self.node_list[min_ind], new_node)
        new_node.cost = min_cost

        return new_node

    def search_best_goal_node(self):
        dist_to_goal_list = [
            self.calc_dist_to_goal(n.x, n.y) for n in self.node_list
        ]
        goal_inds = [
            dist_to_goal_list.index(i) for i in dist_to_goal_list
            if i <= self.expand_dis
        ]

        safe_goal_inds = []
        for goal_ind in goal_inds:
            t_node = self.steer(self.node_list[goal_ind], self.get_nearest_goal(self.node_list[goal_ind].x, self.node_list[goal_ind].y))
            if self.check_collision(t_node, self.obstacle_list):
                safe_goal_inds.append(goal_ind)

        if not safe_goal_inds:
            return None

        min_cost = min([self.node_list[i].cost for i in safe_goal_inds])
        for i in safe_goal_inds:
            if self.node_list[i].cost == min_cost:
                return i

        return None

    def find_near_nodes(self, new_node):
        """
        1) defines a ball centered on new_node
        2) Returns all nodes of the three that are inside this ball
            Arguments:
            ---------
                new_node: Node
                    new randomly generated node, without collisions between
                    its nearest node
            Returns:
            -------
                list
                    List with the indices of the nodes inside the ball of
                    radius r
        """
        nnode = len(self.node_list) + 1
        r = self.connect_circle_dist * math.sqrt((math.log(nnode) / nnode))
        # if expand_dist exists, search vertices in a range no more than
        # expand_dist
        if hasattr(self, 'expand_dis'):
            r = min(r, self.expand_dis)
        dist_list = [(node.x - new_node.x)**2 + (node.y - new_node.y)**2
                     for node in self.node_list]
        near_inds = [dist_list.index(i) for i in dist_list if i <= r**2]
        return near_inds

    def rewire(self, new_node, near_inds):
        """
            For each node in near_inds, this will check if it is cheaper to
            arrive to them from new_node.
            In such a case, this will re-assign the parent of the nodes in
            near_inds to new_node.
            Parameters:
            ----------
                new_node, Node
                    Node randomly added which can be joined to the tree
                near_inds, list of uints
                    A list of indices of the self.new_node which contains
                    nodes within a circle of a given radius.
            Remark: parent is designated in choose_parent.
        """
        for i in near_inds:
            near_node = self.node_list[i]
            edge_node = self.steer(new_node, near_node)
            if not edge_node:
                continue
            edge_node.cost = self.calc_new_cost(new_node, near_node)

            no_collision = self.check_collision(edge_node, self.obstacle_list)
            improved_cost = near_node.cost > edge_node.cost

            if no_collision and improved_cost:
                near_node.x = edge_node.x
                near_node.y = edge_node.y
                near_node.cost = edge_node.cost
                near_node.path_x = edge_node.path_x
                near_node.path_y = edge_node.path_y
                near_node.parent = edge_node.parent
                self.propagate_cost_to_leaves(new_node)

    def calc_new_cost(self, from_node, to_node):
        d, _ = self.calc_distance_and_angle(from_node, to_node)
        return from_node.cost + d

    def propagate_cost_to_leaves(self, parent_node):

        for node in self.node_list:
            if node.parent == parent_node:
                node.cost = self.calc_new_cost(parent_node, node)
                self.propagate_cost_to_leaves(node)
    
    def smooth_path(self, path):
        upt = True
        while upt and len(path) >= 3:
            upt = False
            n = len(path)
            idx = []
            for i in range(n-2):
                if self.check_collision(self.Node(path[i+2][0], path[i+2][1]), self.obstacle_list, last_node = self.Node(path[i][0], path[i][1])):
                    idx.append(i+1)
            if len(idx) > 0:
                w = random.choice(idx)
                upt = True
                path = [x for i, x in enumerate(path) if i != w]
        return path

def main():
    print("Start " + __file__)

    # ====Search Path with RRT====
    obstacle_list = [
        (5, 5, 6, 6),
        (3, 6, 7, 8),
        (3, 8, 5, 9),
        (3, 5, 13, 6),
        (7, 5, 8, 10),
        (9, 5, 10, 6),
        (1, 6, 3, 12),
        (10, 7, 11, 14),
        (3, 10, 5, 14),
        (7, 2, 8, 6),
        (9, 3, 16, 4),
        (5, 12, 10, 14)
    ]  # [x,y,size(radius)]

    # Set Initial parameters
    rrt_star = RRTStar(
        start=[0, 0],
        goals=[[6.1, 10.1], [9,7]],
        rand_area=[[-2, 14], [0, 14]],
        obstacle_list=obstacle_list,
        expand_dis=0.5 ,
                 goal_sample_rate=20,
                 max_iter=100000,
                 connect_circle_dist=10.0)
    path = rrt_star.planning(animation=show_animation, smooth_path = True)

    if path is None:
        print("Cannot find path")
    else:
        print("found path!!")

        # Draw final path
    if show_path:
        rrt_star.draw_graph()
        if type(path)!= type(None):
            plt.plot([x for (x, y) in path], [y for (x, y) in path], 'r--')
        plt.grid(True)
        plt.savefig('/home/gaojiaxuan/onpolicy/onpolicy/scripts/gjx_tmp/rrt_star.png')


if __name__ == '__main__':
    main()