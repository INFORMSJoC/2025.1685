# -*- coding: utf-8 -*-
"""
  Author: Author
  Email : liusuri@mail.dlut.edu.cn
   Time : 2024/11/1 17:23
Function: read data, further restrict self.Arc and self.node_pairs
"""

import pandas as pd
import networkx as nx
from matplotlib import pyplot as plt
import numpy as np
from geopy.distance import geodesic
import os
import shutil
import pickle


class Modeldata:
    """
    read the input file
    """

    def __str__(self):
        return 'this is the input data'

    __repr__ = __str__

    def __init__(self, file_name, arc_elimination=False, Delta_type='MST', Delta_value=None, leader_theta_given=None):
        """
        Model data reading
        :param file_name: input file name
        :param arc_elimination: whether eliminate cut and remain only sparse ones
        :param Delta_type: bigM: all arcs are available; halfBigM: bigM/2; MST: minimum spanning tree, rational routes
        :param keep_all: Whether keep information of all arcs
        """
        self.file_name = file_name
        self.leader_theta_given = leader_theta_given
        self.arc_elimination = arc_elimination
        self.Delta_type = Delta_type
        self.bigM = 1e6  # to represent the distance of not directly connected nodes
        self.Delta_value = Delta_value
        self._read_excel()
        if self.leader_theta_given is not None:
            self.file_name = self.file_name[:-5] + '{}.xlsx'.format(self.leader_theta_given)

    def _read_excel(self):
        """
        read Excel file and get input data
        :return:
        """
        print('begin to load excel...')
        raw_data = pd.read_excel(self.file_name, sheet_name=None)  # read file
        print('generate sets...')
        self.Node_all = list(raw_data['Node']['node_id'])  # set of Node
        tmp = raw_data['Node'].set_index('node_id', drop=True)
        self.node_lat = tmp['node_lat'].to_dict()
        self.node_lon = tmp['node_lon'].to_dict()
        self.node_income_level = tmp['node_income_level'].to_dict()
        self.Hub = list(raw_data['Hub']['hub_id'])  # set of Hub
        self.All_trips = list(raw_data['Trip']['trip_id'])  # set of all trips

        self.Trip_origin = {}  # origin of trip r
        self.Trip_destination = {}  # destination of trip r
        self.Trip_p_amount = {}  # the number of passengers taking trip r

        # read trip information
        print('read trip info...')
        trip_df = raw_data['Trip']
        self.Trip_p_amount = dict(zip(trip_df['trip_id'], trip_df['amount']))
        self.Trip_origin = dict(zip(trip_df['trip_id'], trip_df['origin']))
        self.Trip_destination = dict(zip(trip_df['trip_id'], trip_df['destination']))

        # Apply the scenario-specific demand scale.
        trip_scale = raw_data['Trip_scale']
        list_trips = raw_data['Trip']['trip_id'].tolist()
        trip_scale.index = list_trips
        list_scenarios = list(raw_data['SampleSet']['Sample_id'])
        trip_scale.columns = list_scenarios
        dict_scale = trip_scale.stack().to_dict()  # value: r, s

        self.Trip_p_amount = {(s, r): max(1, np.round(self.Trip_p_amount[r] * dict_scale[r, s]))/50 for s in list_scenarios for r
                              in list_trips}

        # Retain only hubs and nodes that are endpoints of modeled trips.
        self.Node = list(set(list(self.Trip_origin.values()) + list(self.Trip_destination.values()) + self.Hub))
        self.Node.sort()

        # use in model
        self.Trips = self.All_trips.copy()

        # directly connected arc
        print('define connected arcs...')
        if not self.arc_elimination:
            self.Arc = {(i, j) for i in self.Node for j in self.Node if i != j}
        else:
            self.Arc = {(self.Trip_origin[r], h) for r in self.Trips for h in self.Hub}
            self.Arc = self.Arc | {(h, self.Trip_destination[r]) for r in self.Trips for h in self.Hub}
            self.Arc = self.Arc | {(self.Trip_origin[r], self.Trip_destination[r]) for r in self.Trips}
            self.Arc = self.Arc | {(h1, h2) for h1 in self.Hub for h2 in self.Hub if h1 != h2}
            self.Arc = [(i, j) for (i, j) in self.Arc if i != j]

        self.Node_pairs = self.Arc.copy()
        # at least i or j is not a Hub
        self.Hub_pairs = {(h, l) for h in self.Hub for l in self.Hub if (h, l) in self.Arc}

        self.Node_pair_for_model = {(i, j) for (i, j) in self.Arc if (i, j) not in self.Hub_pairs}

        self.Node_pairs_for_mdl_sp = {r: {(self.Trip_origin[r], self.Trip_destination[r])} |
                {(self.Trip_origin[r], h) for h in self.Hub} |
                {(h, self.Trip_destination[r]) for h in self.Hub} for r in self.Trips}  # simplified node pairs
        self.Node_of_trip = {r: [self.Trip_origin[r], self.Trip_destination[r]]+self.Hub for r in self.Trips}

        print('calculate distance dict...')
        self.Dist = {
            (i, j): geodesic((self.node_lat[i], self.node_lon[i]), (self.node_lat[j], self.node_lon[j])).miles for
            (i, j) in self.Arc}
        # Treat distances below 0.01 mile as zero.
        self.Dist = {key: (0 if value < 0.01 else value) for key, value in self.Dist.items()}
        print('calculate max distance...')
        self.max_distance = max([self.Dist[i, j] for (i, j) in self.Arc])

        # set of samples
        print('set of samples...')
        df_samples = raw_data['SampleSet'].copy()
        self.Scenarios = list(df_samples['Sample_id'])
        self.Scenarios_prob = {i: float(df_samples[df_samples['Sample_id'] == i]['Probability']) for i in
                               self.Scenarios}

        self.Sce_Trip = [(s, r) for s in self.Scenarios for r in self.Trips]
        self.Sce_Trip_unexplored = self.Sce_Trip.copy()
        self.Sce_Trip_explored = []

        # read speed
        df_speed = raw_data['Speed'].copy()
        df_speed['index'] = df_speed[['scenario', 'node']].apply(tuple, axis=1)
        df_speed['index'] = df_speed['index'].apply(lambda x: str(x))
        df_speed.set_index('index', inplace=True, drop=True)

        self.speed = {(i, j, s): df_speed.loc['({}, {})'.format(s, i), j] for s in self.Scenarios for (i, j) in
                      self.Arc}

        # read speed_scale
        df_speed_scale = raw_data['speed_scale'].copy()
        df_speed_scale.set_index('scenario', inplace=True, drop=True)
        self.speed_scale = df_speed_scale['scale'].to_dict()

        print('generate travel time')
        self.Travel_time = {(i, j, s): self.Dist[i, j] / self.speed[i, j, s] * 3600 for (i, j) in self.Arc for s in
                            self.Scenarios}

        # other parameters
        print('read other parameters...')
        param_info = raw_data['Parameters']  # the sheet "Parameters"
        param_info.set_index('Param', inplace=True, drop=True)
        self.theta_leader_high_income = param_info.loc['theta_leader', 'Value']
        self.theta_leader_middle_income = self.theta_leader_high_income
        self.theta_leader_low_income = self.theta_leader_high_income
        if self.leader_theta_given is not None:
            self.theta_leader_high_income = self.leader_theta_given
            self.theta_leader_middle_income = self.leader_theta_given
            self.theta_leader_low_income = self.leader_theta_given
        self.theta_follower_high_income = param_info.loc['theta_follower_high_income', 'Value']
        self.theta_follower_middle_income = param_info.loc['theta_follower_middle_income', 'Value']
        self.theta_follower_low_income = param_info.loc['theta_follower_low_income', 'Value']
        self.b_cost_bus = param_info.loc['b', 'Value']
        self.n_buses = param_info.loc['n', 'Value']
        self.g_cost_shuttle = param_info.loc['g', 'Value']
        self.S_wait_bus = param_info.loc['S', 'Value']  # unit: "second"
        if self.Delta_value is not None:
            self.Delta = {'High': self.Delta_value, 'Low': self.Delta_value, 'Middle': self.Delta_value}
        else:
            self.Delta = self.cal_Delta()  # The condition is seperated by the symbol "<"

        self.income_type = ['High', 'Middle', 'Low']

        # beta
        self.Beta_hl = {(h, l): (1 - (self.theta_leader_high_income if self.node_income_level[
                                                                           h] == 'High' else self.theta_follower_low_income)) * self.b_cost_bus * self.n_buses *
                                self.Dist[h, l]/50 for h in self.Hub for l in self.Hub if (h, l) in self.Arc}

        # tao
        self.tao_leader = {(h, l, s): (self.Travel_time[h, l, s] + self.S_wait_bus) * (
            self.theta_leader_high_income if
            self.node_income_level[h] == 'High' else self.theta_leader_low_income) for h
                           in self.Hub for l in self.Hub if (h, l) in self.Arc for s in self.Scenarios}

        self.tao_follower = {(h, l, s): (self.theta_follower_high_income if self.node_income_level[
                                                                                h] == 'High' else self.theta_follower_low_income) * (
                                                self.Travel_time[h, l, s] + self.S_wait_bus) for h in
                             self.Hub for l in self.Hub if (h, l) in self.Arc for s in self.Scenarios}

        # gamma

        # self.gamma_leader = {(i, j, s): (1 - (self.theta_leader_high_income if self.node_income_level[
        #                                 j] == 'High' else self.theta_leader_low_income)) * self.g_cost_shuttle / 2 * self.Dist[i,j] + (
        #                                 self.theta_leader_high_income if self.node_income_level[
        #                                                         j] == 'High' else self.theta_leader_low_income) *
        #                                 self.Travel_time[i, j, s] if
        # self.Dist[i, j] < self.Delta else gamma_penalty for i in self.Node for j in
        #                      self.Node if (i, j) in self.Arc for s in self.Scenarios}

        Delta_use = self.Delta

        # calculate the value of gamma at leader and follower
        theta_use = {'High': self.theta_leader_high_income, 'Middle': self.theta_leader_middle_income,
                     'Low': self.theta_leader_low_income}
        self.gamma_leader = {
            (i, j, s): (1 - theta_use[self.node_income_level[i]]) * self.g_cost_shuttle * self.Dist[i, j] +
                       theta_use[self.node_income_level[i]] * self.Travel_time[i, j, s] * self.speed_scale[s] for (i, j)
            in self.Arc for s in
            self.Scenarios}

        theta_use = {'High': self.theta_follower_high_income, 'Middle': self.theta_follower_middle_income,
                     'Low': self.theta_follower_low_income}
        self.gamma_follower = {
            (i, j, s): (1 - theta_use[self.node_income_level[i]]) * self.g_cost_shuttle * self.Dist[i, j] +
                       theta_use[self.node_income_level[i]] * self.Travel_time[i, j, s] * self.speed_scale[s] for (i, j)
            in self.Arc for s in
            self.Scenarios}
        # penalize on the long-distance arcs
        if max(self.gamma_leader.values()) > 20 * np.mean(list(self.gamma_leader.values())):
            gamma_penalty = max(self.gamma_leader.values())
        else:
            gamma_penalty = min(max(self.gamma_leader.values()) * 100, 5e5)

        self.gamma_leader = {(i, j, s): self.gamma_leader[i, j, s] if self.Dist[i, j] < Delta_use[
            self.node_income_level[i]] else gamma_penalty for (i, j) in self.Arc for s in self.Scenarios}

        if max(self.gamma_follower.values()) > 20 * np.mean(list(self.gamma_follower.values())):
            gamma_penalty = max(self.gamma_follower.values())
        else:
            gamma_penalty = min(max(self.gamma_follower.values())*100, 5e5)

        self.gamma_follower = {(i, j, s): self.gamma_follower[i, j, s] if self.Dist[i, j] < Delta_use[
            self.node_income_level[i]] else gamma_penalty for (i, j) in self.Arc for s in self.Scenarios}



    def obtain_average_value(self):
        """
        obtain the average value of tao & gamma (leader and follower)
        :return:
        """
        self.tao_leader = {(h, l, self.Scenarios[0]): np.sum(
            [self.tao_leader[h, l, s] * self.Scenarios_prob[s] for s in self.Scenarios]) for
                           (h, l) in self.Hub_pairs if (h, l) in self.Arc}
        self.tao_follower = {(h, l, self.Scenarios[0]): np.sum(
            [self.tao_follower[h, l, s] * self.Scenarios_prob[s] for s in self.Scenarios]) for
                             (h, l) in self.Hub_pairs if (h, l) in self.Arc}

        self.gamma_leader = {(i, j, self.Scenarios[0]): np.sum(
            [self.gamma_leader[i, j, s] * self.Scenarios_prob[s] for s in self.Scenarios]) for
                             (i, j) in self.Node_pair_for_model if (i, j) in self.Arc}
        self.gamma_follower = {(i, j, self.Scenarios[0]): np.sum(
            [self.gamma_follower[i, j, s] * self.Scenarios_prob[s] for s in self.Scenarios]) for
            (i, j) in self.Node_pair_for_model if (i, j) in self.Arc}
        self.Trip_p_amount = {
            (self.Scenarios[0], r): np.sum(self.Trip_p_amount[s, r] * self.Scenarios_prob[s] for s in self.Scenarios)
            for r in self.Trips}

        self.Scenarios = [self.Scenarios[0]]
        self.Scenarios_prob = {self.Scenarios[0]: 1}
        self.Sce_Trip = [(s, r) for s in self.Scenarios for r in self.Trips]

    def plot_graph(self, graph=None, directed=False):
        """
        :param graph:
        :param directed: is a directed graph inputted, default: False
        :return:
        """
        if not graph:
            graph = self.Graph
        pos = nx.spring_layout(graph)  # 用 FR算法排列节点
        if not directed:
            nx.draw(graph, pos, with_labels=True, alpha=0.5)
            labels = nx.get_edge_attributes(graph, 'weight')
            nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels)
        else:
            nx.draw(graph, pos, with_labels=True, alpha=0.5, connectionstyle='arc3, rad = 0.1')
            labels = dict([((u, v,), d['weight'])
                           for u, v, d in graph.edges(data=True)])
            nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels, label_pos=0.65)
            nx.draw_networkx_edge_labels(graph, pos)

        plt.show()

    def cal_Delta(self):
        # The MST threshold ensures each node has at least one eligible connection.
        self.Graph = nx.Graph()
        self.Graph.add_weighted_edges_from([i, j, self.Dist[i, j]] for (i, j) in self.Arc)

        if self.Delta_type == 'bigM':
            Delta = self.bigM * 0.99  # to exclude self loops whose distance is set as bigM
            Delta = {'High': Delta, 'Low': Delta, 'Middle': Delta}
        elif self.Delta_type == 'halfBigM':
            Delta = self.bigM * 0.5
            Delta = {'High': Delta, 'Low': Delta, 'Middle': Delta}
        elif self.Delta_type == 'MST':
            min_span_tree = nx.minimum_spanning_tree(self.Graph)

            list_edges = list(min_span_tree.edges())
            Delta = max([self.Dist[i, j] if (i, j) in self.Arc else self.Dist[j, i] for (i, j) in list_edges]) * 1.02
            Delta = {'High': Delta, 'Low': Delta, 'Middle': Delta}

        elif self.Delta_type == 'intersection':
            Delta_high_income = 2 * self.theta_follower_high_income * self.S_wait_bus / (
                        1 - self.theta_follower_high_income) / self.g_cost_shuttle
            Delta_middle_income = 2 * self.theta_follower_middle_income * self.S_wait_bus / (
                        1 - self.theta_follower_middle_income) / self.g_cost_shuttle
            Delta_low_income = 2 * self.theta_follower_low_income * self.S_wait_bus / (
                        1 - self.theta_follower_low_income) / self.g_cost_shuttle
            Delta = {'High': Delta_high_income * 1.02, 'Low': Delta_low_income * 1.02,
                     'Middle': Delta_middle_income * 1.02}
        else:
            raise 'unknown Delta type: {}'.format(self.Delta_type)

        return Delta

    def output_to_disk(self):
        """Write compact temporary data for parallel response-search workers.

        Each worker loads only the scenario- and trip-specific information it needs
        from these .pkl files. This avoids passing the full ``Modeldata`` object to
        every worker. These files are not used by ``IterateComb.parallel_method``
        when solving the later C&CG follower subproblems.
        """
        current_directory = os.path.dirname(os.path.abspath(__file__))
        upper_2_dir = os.path.dirname(current_directory)
        output_folder = os.path.join(upper_2_dir, 'output', '{}_data_info'.format(os.path.basename(self.file_name)[:-5]))
        if os.path.exists(output_folder):
            for filename in os.listdir(output_folder):
                file_path = os.path.join(output_folder, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f"Failed to delete {file_path}. Reason: {e}")
        else:
            os.makedirs(output_folder)

        with open(os.path.join(output_folder, 'Hub_pairs.pkl'), 'wb') as f:
            pickle.dump(self.Hub_pairs, f)
        with open(os.path.join(output_folder, 'Hub.pkl'), 'wb') as f:
            pickle.dump(self.Hub, f)

        for r in self.Trips:
            Node_pairs_for_mdl_sp_r = {r: self.Node_pairs_for_mdl_sp[r]}
            with open(os.path.join(output_folder, 'Node_pairs_for_mdl_sp_{}.pkl'.format(r)), 'wb') as f:
                pickle.dump(Node_pairs_for_mdl_sp_r, f)
            Trip_origin_r = {r: self.Trip_origin[r]}
            with open(os.path.join(output_folder, 'Trip_origin_{}.pkl'.format(r)), 'wb') as f:
                pickle.dump(Trip_origin_r, f)
            Trip_destination_r = {r: self.Trip_destination[r]}
            with open(os.path.join(output_folder, 'Trip_destination_{}.pkl'.format(r)), 'wb') as f:
                pickle.dump(Trip_destination_r, f)
            Node_of_trip_r = {r: self.Node_of_trip[r]}
            with open(os.path.join(output_folder, 'Node_of_trip_{}.pkl'.format(r)), 'wb') as f:
                pickle.dump(Node_of_trip_r, f)

        for s in self.Scenarios:
            tao_follower_s = {(h,l,s): self.tao_follower[h,l,s] for
                           (h, l) in self.Hub_pairs if (h, l) in self.Arc}
            with open(os.path.join(output_folder, 'tao_follower_{}.pkl'.format(s)), 'wb') as f:
                pickle.dump(tao_follower_s, f)
            gamma_follower_s = {(i,j,s): self.gamma_follower[i,j,s] for (i, j) in self.Node_pair_for_model if (i, j) in self.Arc}
            with open(os.path.join(output_folder, 'gamma_follower_{}.pkl'.format(s)), 'wb') as f:
                pickle.dump(gamma_follower_s, f)
            tao_leader_s = {(h,l,s): self.tao_leader[h,l,s] for
                           (h, l) in self.Hub_pairs if (h, l) in self.Arc}
            with open(os.path.join(output_folder, 'tao_leader_{}.pkl'.format(s)), 'wb') as f:
                pickle.dump(tao_leader_s, f)
            gamma_leader_s = {(i,j,s): self.gamma_leader[i,j,s] for (i, j) in self.Node_pair_for_model if (i, j) in self.Arc}
            with open(os.path.join(output_folder, 'gamma_leader_{}.pkl'.format(s)), 'wb') as f:
                pickle.dump(gamma_leader_s, f)


    def clear_output_disk(self):
        """
        clear the information output to the disk
        :return:
        """
        current_directory = os.path.dirname(os.path.abspath(__file__))
        upper_2_dir = os.path.dirname(current_directory)
        output_folder = os.path.join(upper_2_dir, 'output', '{}_data_info'.format(
            os.path.basename(self.file_name)[:-5]))

        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)




    def plot_graph_lat_lon(self):
        # plot graph using lat & lon
        for n in self.Node:
            plt.scatter(self.node_lat[n], self.node_lon[n], linewidth=0.5, color='blue', marker='o')
        for h in self.Hub:
            plt.scatter(self.node_lat[h], self.node_lon[h], linewidth=0.5, color='red', marker='o')

        # draw trips
        for r in self.Trips:
            ori = self.Trip_origin[r]
            des = self.Trip_destination[r]
            plt.plot([self.node_lat[ori], self.node_lat[des]], [self.node_lon[ori], self.node_lon[des]], linewidth=1.2,
                     color='black', alpha=0.5, linestyle='--')

        plt.show()





if __name__ == '__main__':
    file_name = '../data/node725-hub10-tripNfull-scenario1-passenger2586-time2025-06-02-14-33-42-theta_0.0001.xlsx'
    modeldata = Modeldata(file_name, arc_elimination=True, Delta_type='MST', Delta_value=5)
    modeldata.obtain_average_value()
    modeldata.output_to_disk()
    modeldata.plot_graph_lat_lon()

