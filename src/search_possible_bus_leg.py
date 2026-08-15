# -*- coding: utf-8 -*-
"""
  Author: Author
  Email : liusuri@mail.dlut.edu.cn
   Time : 2024/12/25 16:06
Function: 1) Search acceptable bus lets for all trips.
          2) This .py file is mainly for the response search algorithm
          3) the parameters parallel and n_core in the function find_hub_leg_all are to be specified depending on the usage
"""
import time

import gurobipy as gp
import numpy as np
from gurobipy import GRB
from read_data import Modeldata
import queue
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import pandas as pd
from matplotlib import pyplot as plt
import pickle
import networkx as nx


def find_hub_leg_indep_fun(idx_list, file_name, n_iter_max):
    """
    find potential bus leg routines for trip r under scenario s
    this is an independent function without relying on a class
    :return:
    """
    class Data():
        def __init__(self, s, r, file_name):
            current_directory = os.path.dirname(os.path.abspath(__file__))
            upper_2_dir = os.path.dirname(current_directory)
            output_folder = os.path.join(upper_2_dir, 'output', '{}_data_info'.format(
                os.path.basename(file_name)[:-5]))  # create a folder to store the information
            with open(os.path.join(output_folder, 'Hub_pairs.pkl'), 'rb') as f:
                self.Hub_pairs = pickle.load(f)
            with open(os.path.join(output_folder, 'Hub.pkl'), 'rb') as f:
                self.Hub = pickle.load(f)

            with open(os.path.join(output_folder, 'Node_pairs_for_mdl_sp_{}.pkl'.format(r)), 'rb') as f:
                self.Node_pairs_for_mdl_sp = pickle.load(f)
            with open(os.path.join(output_folder, 'Trip_origin_{}.pkl'.format(r)), 'rb') as f:
                self.Trip_origin = pickle.load(f)
            with open(os.path.join(output_folder, 'Trip_destination_{}.pkl'.format(r)), 'rb') as f:
                self.Trip_destination = pickle.load(f)
            with open(os.path.join(output_folder, 'Node_of_trip_{}.pkl'.format(r)), 'rb') as f:
                self.Node_of_trip = pickle.load(f)

            with open(os.path.join(output_folder, 'tao_follower_{}.pkl'.format(s)), 'rb') as f:
                self.tao_follower = pickle.load(f)
            with open(os.path.join(output_folder, 'gamma_follower_{}.pkl'.format(s)), 'rb') as f:
                self.gamma_follower = pickle.load(f)
            with open(os.path.join(output_folder, 'tao_leader_{}.pkl'.format(s)), 'rb') as f:
                self.tao_leader = pickle.load(f)
            with open(os.path.join(output_folder, 'gamma_leader_{}.pkl'.format(s)), 'rb') as f:
                self.gamma_leader = pickle.load(f)

    class RoutineFinderInner():
        def __init__(self, s, r):
            self.s = s
            self.r = r
            self.forbiden_leg = []  # [(h,l), ...]

        def solve_model_dijkstra(self, data):
            """
            solve the follower problem as a shortest path problem using Dijkstra algorithm
            :return:
            """
            self.data = data
            G = nx.DiGraph()
            for (h,l) in self.data.Hub_pairs:
                if (h,l) not in self.forbiden_leg:
                    G.add_edge(h, l, weight=self.data.tao_follower[h,l,self.s])
            for (i,j) in self.data.Node_pairs_for_mdl_sp[self.r]:
                G.add_edge(i, j, weight=self.data.gamma_follower[i, j, self.s])

            path = nx.dijkstra_path(G, source=self.data.Trip_origin[self.r], target=self.data.Trip_destination[self.r], weight='weight')

            used_bus_leg = []
            leader_cost = 0
            follower_cost = 0
            for idx in range(len(path)-1):
                ori = path[idx]
                des = path[idx + 1]
                if ori in self.data.Hub and des in self.data.Hub:
                    # use shuttle
                    used_bus_leg.append((ori, des))
                    leader_cost += self.data.tao_leader[ori, des, self.s]
                    follower_cost += self.data.tao_follower[ori, des, self.s]
                else:
                    leader_cost += self.data.gamma_leader[ori, des, self.s]
                    follower_cost += self.data.gamma_follower[ori, des, self.s]
            used_bus_leg.sort()

            result = {
                'used_leg': used_bus_leg,
                'obj_follower': follower_cost,
                'obj_leader': leader_cost
            }
            return result




        def solve_model(self, data):
            self.data = data
            s = self.s
            r = self.r
            with gp.Env() as env, gp.Model('SubShortestPathWithStrongDualModel', env=env) as model:
                # add variables
                x = model.addVars(
                    gp.tuplelist((h, l) for (h, l) in self.data.Hub_pairs),
                    vtype=GRB.CONTINUOUS, name='x')
                y = model.addVars(
                    gp.tuplelist((i, j) for (i, j) in self.data.Node_pairs_for_mdl_sp[r]),
                    vtype=GRB.CONTINUOUS, name='y')
                z = model.addVars(
                    self.data.Hub_pairs, vtype=GRB.BINARY)
                for (h,l) in self.forbiden_leg:
                    z[h,l].setAttr('ub', 0)

                # add constraints
                i = self.data.Trip_origin[r]
                model.addConstr((gp.quicksum(
                    x[i, h] - x[h, i] for h in self.data.Hub if
                    (i, h) in self.data.Hub_pairs) + gp.quicksum(
                    y[i, j] for j in self.data.Node_of_trip[r] if
                    (i, j) in self.data.Node_pairs_for_mdl_sp[r]) - gp.quicksum(
                    y[j, i] for j in self.data.Node_of_trip[r] if (j, i) in self.data.Node_pairs_for_mdl_sp[r]) == 1),
                                name='c_flb_or')
                i = self.data.Trip_destination[r]
                model.addConstr((gp.quicksum(
                    x[i, h] - x[h, i] for h in self.data.Hub if
                    (i, h) in self.data.Hub_pairs) + gp.quicksum(
                    y[i, j] for j in self.data.Node_of_trip[r] if
                    (i, j) in self.data.Node_pairs_for_mdl_sp[r]) - gp.quicksum(
                    y[j, i] for j in self.data.Node_of_trip[r] if (j, i) in self.data.Node_pairs_for_mdl_sp[r]) == -1),
                                name='c_flb_de')
                model.addConstrs((gp.quicksum(
                    x[i, h] - x[h, i] for h in self.data.Hub if
                    (i, h) in self.data.Hub_pairs) + gp.quicksum(
                    y[i, j] for j in self.data.Node_of_trip[r] if
                    (i, j) in self.data.Node_pairs_for_mdl_sp[r]) - gp.quicksum(
                    y[j, i] for j in self.data.Node_of_trip[r] if (j, i) in self.data.Node_pairs_for_mdl_sp[r]) == 0 for i
                                  in
                                  self.data.Node_of_trip[r] if
                                  i not in [self.data.Trip_origin[r],
                                            self.data.Trip_destination[r]]),
                                 name='c_flb_normal')
                model.addConstrs(
                    (x[h, l] <= z[h, l] for (h, l) in self.data.Hub_pairs),
                    name='conn_enable')

                model.addConstrs((x[h, l] <= 1 for (h, l) in self.data.Hub_pairs), name='x_ub')
                model.addConstrs((y[i, j] <= 1 for (i, j) in self.data.Node_pairs_for_mdl_sp[r]), name='y_ub')

                # set objective - follower
                obj_follower = gp.quicksum(
                    self.data.tao_follower[h, l, s] * x[h, l] for (h, l) in self.data.Hub_pairs)
                obj_follower += gp.quicksum(
                    self.data.gamma_follower[i, j, s] * y[i, j] for (i, j) in self.data.Node_pairs_for_mdl_sp[r])
                model.setObjective(obj_follower)

                # calculate objective -leader
                obj_leader = gp.quicksum(
                    self.data.tao_leader[h, l, s] * x[h, l] for (h, l) in self.data.Hub_pairs)
                obj_leader += gp.quicksum(
                    self.data.gamma_leader[i, j, s] * y[i, j] for (i, j) in self.data.Node_pairs_for_mdl_sp[r])

                # model.setParam('TimeLimit', 360)  # 10 minutes
                model.setParam('LogToConsole', 0)
                model.optimize()

                opened_bus_leg = [(h,l) for (h,l) in self.data.Hub_pairs if x[h,l].X > 0.9]
                opened_bus_leg.sort()
                result = {
                    'used_leg': opened_bus_leg,
                    'obj_follower': obj_follower.getValue(),
                    'obj_leader': obj_leader.getValue()
                }

                return result

    # define empty sets
    full_sorted_bus_leg_sol = {}
    full_sorted_leader_obj = {}
    full_original_leader_obj = {}
    full_not_completely_explored = {}
    full_sorted_follower_obj = {}
    full_search_time = {}
    full_search_iter = {}

    for (s,r) in idx_list:
        start_time = time.time()
        data = Data(s, r, file_name)
        bus_leg_sol = []  # all possible bus leg routines[[(1,2), (2,3)], [(3,4)], ...]
        sol_id = 0
        leader_obj = {}  # the leader obj associated with each bus leg selection
        follower_obj = {}  # the follower obj associated with each bus leg selection
        node_to_explore = queue.Queue()  # tree search records
        node_to_explore.put(RoutineFinderInner(s,r))  # tree search records
        not_completely_explored = False
        n_iters = 0
        while node_to_explore.qsize() > 0:
            node = node_to_explore.get()
            result = node.solve_model_dijkstra(data=data)

            if len(result['used_leg']) > 0:
                if result['used_leg'] not in bus_leg_sol:
                    bus_leg_sol.append(result['used_leg'])  # record used bus leg
                    leader_obj[sol_id] = result['obj_leader']  # record associated bus leg
                    follower_obj[sol_id] = result['obj_follower']  # record associated bus leg
                    sol_id += 1
                for (h,l) in result['used_leg']:
                    node_new = RoutineFinderInner(s, r)
                    node_new.forbiden_leg = node.forbiden_leg.copy()
                    node_new.forbiden_leg.append((h,l))
                    node_to_explore.put(node_new)
            n_iters += 1
            if n_iters > n_iter_max or node_to_explore.qsize() + n_iters > n_iter_max:
                not_completely_explored = True
                break



        node_new = RoutineFinderInner(s, r)
        node_new.forbiden_leg = data.Hub_pairs
        result = node_new.solve_model_dijkstra(data=data)
        original_leader_obj = result['obj_leader']  # the leader obj when all hubs are closed

        sorted_follower_obj = dict(sorted(follower_obj.items(), key=lambda item: item[1], reverse=False))  # sort by obj value, asscending
        sorted_leader_obj = {key: leader_obj[key] for key in sorted_follower_obj}
        sorted_leader_obj = {i: value for i, value in enumerate(sorted_leader_obj.values())}
        sorted_bus_leg_sol = [bus_leg_sol[key] for key in sorted_follower_obj]
        search_time = round(time.time() - start_time, 5)

        full_sorted_bus_leg_sol.update({(s,r): sorted_bus_leg_sol.copy()})
        full_sorted_leader_obj.update({(s,r): sorted_leader_obj.copy()})
        full_original_leader_obj.update({(s,r): original_leader_obj})
        full_not_completely_explored.update({(s,r): not_completely_explored})
        full_sorted_follower_obj.update({(s,r): sorted_follower_obj.copy()})
        full_search_time.update({(s,r): search_time})
        full_search_iter.update({(s,r): n_iters})

    return full_sorted_bus_leg_sol, full_sorted_leader_obj, full_original_leader_obj, full_not_completely_explored, full_sorted_follower_obj, full_search_time, full_search_iter










class PotentialHubFinder():
    """
    Find potential hub legs for all trips under all scenarios
    """
    def __init__(self, data:Modeldata, n_iter_max=100):
        self.data = data
        self.n_iter_max = n_iter_max

    def find_hub_leg_all(self, parallel=False, n_core=8):
        """
        find potential hub legs for all trips under all scenarios
        @:param parallel: False runs response search sequentially with ``self.data``;
                         True distributes response-search tasks among worker processes
        @:param n_core: number of response-search worker processes (default: 8).
                        Parallel workers load compact scenario/trip data from the
                        temporary .pkl files written by ``Modeldata.output_to_disk``.
                        This is separate from ``IterateComb.parallel_method``.
        :return:
        """
        current_directory = os.path.dirname(os.path.abspath(__file__))
        upper_2_dir = os.path.dirname(current_directory)
        data_file = os.path.basename(self.data.file_name)[:-5]
        output_path = os.path.join(upper_2_dir, 'output')

        if not parallel:
            prefered_hubs = {}
            prefered_leader_obj = {}
            original_leader_obj = {}
            not_completely_explored = {}
            prefered_follower_obj = {}
            search_time_rec = {}
            search_iter_rec = {}
            for s,r in self.data.Sce_Trip:
                bus_leg_sol, leader_obj, original_leader_obj_tmp, not_completely_explored_tmp, prefered_follower_obj_tmp, search_time, search_iter = find_hub_leg_indep_fun([(s, r)], file_name=self.data.file_name, n_iter_max=self.n_iter_max)
                prefered_hubs[s,r] = bus_leg_sol[s,r].copy()
                prefered_leader_obj[s,r] = leader_obj[s,r].copy()
                original_leader_obj[s,r] = original_leader_obj_tmp[s,r]
                not_completely_explored[s,r] = not_completely_explored_tmp[s,r]
                prefered_follower_obj[s,r] = prefered_follower_obj_tmp[s,r].copy()
                search_time_rec[s,r] = search_time[s,r]
                search_iter_rec[s,r] = search_iter[s,r]


        else:

            prefered_hubs = {}
            prefered_leader_obj = {}
            original_leader_obj = {}
            not_completely_explored = {}
            prefered_follower_obj = {}
            search_time_rec = {}
            search_iter_rec = {}


            idx_cores = [[] for _ in range(n_core)]
            g_idx = 0
            for r in self.data.Trips:
                for s in self.data.Scenarios:
                    idx_cores[g_idx].append((s,r))
                    g_idx += 1
                    g_idx = g_idx%n_core

            t_parallel = time.time()
            pool = mp.Pool(processes=n_core)
            result = pool.starmap(find_hub_leg_indep_fun, [(prob_idx,self.data.file_name, self.n_iter_max) for prob_idx in idx_cores])
            pool.close()
            pool.join()
            t_parallel = time.time() - t_parallel
            for i in range(n_core):
                prefered_hubs.update(result[i][0])
                prefered_leader_obj.update(result[i][1])
                original_leader_obj.update(result[i][2])
                not_completely_explored.update(result[i][3])
                prefered_follower_obj.update(result[i][4])
                search_time_rec.update(result[i][5])
                search_iter_rec.update(result[i][6])

        f_name = os.path.join(output_path, data_file + '_prefered_hubs.json')
        prefered_hubs_tmp = {str(key): value for key,value in prefered_hubs.items()}
        with open(f_name, "w") as f:
            json.dump(prefered_hubs_tmp, f, indent=4)

        f_name = os.path.join(output_path, data_file + '_prefered_leader_obj.json')
        prefered_leader_obj_tmp = {str(key): value for key,value in prefered_leader_obj.items()}
        with open(f_name, "w") as f:
            json.dump(prefered_leader_obj_tmp, f, indent=4)

        f_name = os.path.join(output_path, data_file + '_prefered_follower_obj.json')
        prefered_follower_obj_tmp = {str(key): value for key,value in prefered_follower_obj.items()}
        with open(f_name, "w") as f:
            json.dump(prefered_follower_obj_tmp, f, indent=4)

        f_name = os.path.join(output_path, data_file + '_prefered_original_leader_obj.json')
        original_leader_obj_tmp = {str(key): value for key,value in original_leader_obj.items()}
        with open(f_name, "w") as f:
            json.dump(original_leader_obj_tmp, f, indent=4)

        f_name = os.path.join(output_path, data_file + '_not_complete_explored.json')
        not_completely_explored_tmp = {str(key): value for key,value in not_completely_explored.items() if value}
        with open(f_name, "w") as f:
            json.dump(not_completely_explored_tmp, f, indent=4)

        f_name = os.path.join(output_path, data_file + '_search_time.json')
        search_time_tmp = {str(key): value for key,value in search_time_rec.items() if value}
        with open(f_name, "w") as f:
            json.dump(search_time_tmp, f, indent=4)

        f_name = os.path.join(output_path, data_file + '_search_iter.json')
        search_iter_tmp = {str(key): value for key,value in search_iter_rec.items() if value}
        with open(f_name, "w") as f:
            json.dump(search_iter_tmp, f, indent=4)

        return prefered_hubs, prefered_leader_obj, original_leader_obj, not_completely_explored, search_time_rec, search_iter_rec





