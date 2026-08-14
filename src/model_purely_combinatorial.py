# -*- coding: utf-8 -*-
"""
  Author: Author
  Email : liusuri@mail.dlut.edu.cn
   Time : 2024/12/26 16:30
Function: 1) This code redefines the cutting plane algorithm in model_parallel.py.
          2) Combinatorial cuts are added for a follower when the response search algorithm explores all its responses.
          3) the class IterateComb is used for evaluating the efficiency of the response search and cutting plane algorithms in the manuscript
"""

import gurobipy as gp
from gurobipy import GRB
from search_possible_bus_leg import PotentialHubFinder
from read_data import Modeldata
import os
import re
import json
import pandas as pd
import time
from model_parallel import VarValue, SubModel, SubShortestPathWithStrongDualModel
import argparse
from multiprocessing.pool import ThreadPool
import multiprocessing as mp
import numpy as np
from matplotlib import pyplot as plt
import networkx as nx


ALLOW_CORE = 16


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--CCG_timelimit", type=int, help="time limit of CCG algo", default=3600*12)
    parser.add_argument("--MP_timelimit", type=int, help="time limit of MP", default=720)  # spend less time on MP. The i-ccg algo will solve MP repeatedly when required.
    parser.add_argument("--use_i_ccg", type=int, help="whether use i-CCG algo", default=True)
    parser.add_argument("--MP_gap_scale", type=int, help="MIPGap = MIP * scale", default=0.15)
    parser.add_argument("--inexact_gap_threshold", type=int, help="inexact_gap_threshold", default=0.05)
    parser.add_argument("--stop_gap", type=int, help="stop gap CCG", default=1e-5)
    parser.add_argument("--MP_ini_gap", type=int, help="initial gap of MP", default=0.1)

    parser.add_argument("--Single_Level_timelimit", type=int, help="timelimit of single level algo rithm", default=3600)

    return parser



class CombinatorialCutBilevelModel():
    """dr and fr are identical, one layer model"""
    def __init__(self, data: Modeldata):
        """
        initial function
        :param data: modeldata
        :param use_KKT: use KKT or not
        :param use_sos1: use SOS1 constraints or not
        :param add_strong_dual: whether including strong duality when using SOS1, valid only when use_sos1=True
        """
        self.data = data
        hub_finder = PotentialHubFinder(data=data)
        self.prefered_hubs, self.prefered_leader_obj, self.original_leader_obj, self.unexplred_probles = hub_finder.find_hub_leg_all()  # find all potential hubs
        self.n_leg_trip = {(s,r): len(self.prefered_leader_obj[s,r]) for s,r in self.data.Sce_Trip}  # number of potential legs per subproblem (s,r)
        self.s_r_l = [(s, r, l) for s, r in data.Sce_Trip for l in range(self.n_leg_trip[s, r]) if
                      self.n_leg_trip[s, r] > 0]


        current_directory = os.path.dirname(os.path.abspath(__file__))
        upper_2_dir = os.path.dirname(current_directory)
        data_file = os.path.basename(data.file_name)[:-5]
        log_file_prefix = os.path.join(upper_2_dir, 'output')
        log_file_prefix = os.path.join(log_file_prefix, data_file)
        self.env = gp.Env(log_file_prefix + '_OneLayer_comb_bilevel')
        self.model = gp.Model('OneLayerCombBilevelModel', self.env)
        self.var = self.Variable(model=self.model, data=data, set_s_r_l=self.s_r_l)
        self.cons = self.Constraint()
        self.add_constraints()
        self.set_objective()


    class Variable():
        """class of variables"""
        def __init__(self, model, data: Modeldata, set_s_r_l):
            # vars in the master problem
            self.z = model.addVars(data.Hub_pairs, vtype=GRB.BINARY, name='z')
            self.aux_omega = model.addVars(data.Sce_Trip, vtype=GRB.CONTINUOUS, name='aux_omega')
            self.delta = model.addVars(set_s_r_l,
                                       vtype=GRB.BINARY, name='delta')  # whether the leg is satisfied
            self.beta = model.addVars(set_s_r_l,
                                      vtype=GRB.BINARY, name='beta')  # whether the leg is selected

    class Constraint():
        """class of constraint"""

        def __init__(self):
            self.bigM = 1e6


    def add_constraints(self):
        self.cons.leader_flow_balance = self.model.addConstrs((gp.quicksum(
            self.var.z[h, l] for l in self.data.Hub if (h, l) in self.data.Hub_pairs) == gp.quicksum(
            self.var.z[l, h] for l in self.data.Hub if (l, h) in self.data.Hub_pairs) for h in
                                                               self.data.Hub), name='leader_flow_balance')

        self.cons.delta_1 = self.model.addConstrs(
            (self.var.delta[s, r, leg] >= gp.quicksum(self.var.z[h, l] for h, l in self.prefered_hubs[s, r][leg]) -
                self.n_leg_trip[s, r] + 1 for s, r in self.data.Sce_Trip for leg in range(self.n_leg_trip[s, r])), name='delta_1'
        )
        self.cons.delta_2 = self.model.addConstrs(
            (self.var.delta[s, r, leg] <= gp.quicksum(self.var.z[h, l] for h, l in self.prefered_hubs[s, r][leg])/
             self.n_leg_trip[s, r] for s, r in self.data.Sce_Trip for leg in range(self.n_leg_trip[s, r])), name='delta_2'
        )
        self.cons.beta_1 = self.model.addConstrs(
            (gp.quicksum(self.var.beta[s, r, leg] for leg in range(self.n_leg_trip[s, r])) <= 1 for s, r in
             self.data.Sce_Trip), name='beta_1'
        )
        self.cons.beta_2 = self.model.addConstrs(
            (self.var.beta[s,r,leg] <= self.var.delta[s,r,leg] for s, r in self.data.Sce_Trip
             for leg in range(self.n_leg_trip[s, r])), name='beta_2'
        )
        # TODO: the cons.beta_3 is incorrect, but this class is not used so didn't revise. Check the other class for reference
        self.cons.beta_3 = self.model.addConstrs(
            (self.var.beta[s, r, leg] >= self.var.beta[s, r, leg+1] for s, r in self.data.Sce_Trip
             for leg in range(self.n_leg_trip[s, r]-1)), name='beta_3'
        )
        self.cons.beta_lb = self.model.addConstrs(
            (gp.quicksum(self.var.beta[s, r, leg] for leg in range(self.n_leg_trip[s, r])) >= gp.quicksum(
                self.var.delta[s, r, leg] for leg in range(self.n_leg_trip[s, r])) / self.n_leg_trip[s, r] for s, r in
             self.data.Sce_Trip if self.n_leg_trip[s, r] > 0), name='beta_lb'
        )
        for s,r in self.data.Sce_Trip:
            if self.n_leg_trip[s,r] == 0:
                self.var.aux_omega[s,r].setAttr('lb', self.original_leader_obj[s,r])
                self.var.aux_omega[s,r].setAttr('ub', self.original_leader_obj[s,r])
            else:
                potential_M = list(self.prefered_leader_obj[s, r].values()) + [self.original_leader_obj[s, r]]
                bigM = max(potential_M)
                self.model.addConstrs(
                    (self.var.aux_omega[s, r] >= self.prefered_leader_obj[s, r][leg] - bigM * (
                                1 - self.var.beta[s, r, leg]) for leg in range(self.n_leg_trip[s, r])),
                    name='omega_lb_{}_{}'.format(s, r)
                )
                self.model.addConstr(
                    (self.var.aux_omega[s, r] >= self.original_leader_obj[s, r] - bigM * (
                        gp.quicksum(self.var.beta[s, r, leg] for leg in range(self.n_leg_trip[s, r])))),
                    name='omega_lb_no_bus_opened_{}_{}'.format(s, r)
                )


    def set_objective(self):
        obj = gp.quicksum(self.data.Beta_hl[h, l] * self.var.z[h, l] for (h, l) in self.data.Hub_pairs)
        obj += gp.quicksum(self.data.Scenarios_prob[s]*self.data.Trip_p_amount[s,r]*self.var.aux_omega[s,r] for s,r in self.data.Sce_Trip)
        self.model.setObjective(obj, sense=GRB.MINIMIZE)

    def solve_model(self):
        self.model.setParam('TimeLimit', 3600*8)  # 10 minutes
        # self.model.setParam('FeasibilityTo', 1e-1)
        # self.model.setParam('Presolve', 0)
        # self.model.setParam('Heuristics', 0)
        self.model.setParam('LogToConsole', 0)
        self.model.optimize()

    def present_solution(self):
        for s,r in self.data.Sce_Trip:
            print(s,r,self.var.aux_omega[s,r].X)

    def update_result(self, result):
        result = {
            'file': [],
            'ub': [],
            'lb': [],
            'MIPGap': [],
            'solve_time': []
        }
        result['file'].append(re.findall(r'\d+', self.data.file_name)[:4])
        result['ub'].append(self.model.getObjective().getValue())
        result['lb'].append(self.model.ObjBound)
        result['MIPGap'].append(self.model.MIPGap)
        result['solve_time'].append(self.model.Runtime)
        return result


    def write_solution(self):
        """
        write solution: for subproblem (s,r), which beta is activated
        :return:
        """
        if self.model.SolCount == 0:
            return
        result = {(s,r): l for s,r,l in self.s_r_l if self.var.beta[s,r,l].X > 0.1}
        result = {key: value for key, value in result.items()}
        current_directory = os.path.dirname(os.path.abspath(__file__))
        upper_2_dir = os.path.dirname(current_directory)
        data_file = os.path.basename(data.file_name)[:-5]
        output_file_prefix = os.path.join(upper_2_dir, 'output', data_file+'combinatorial_bilevel_decision.json')
        with open(output_file_prefix, 'w') as json_file:
            json.dump(result, json_file, indent=4)



class RoutineFinder():
    """
    find the shortest path given a trip r under scenario s
    """
    def __init__(self, data: Modeldata, param_z):
        self.data = data
        self.param_z = param_z

    def solve_model(self, s, r):
        with gp.Env() as env, gp.Model('SubShortestPathWithStrongDualModel', env=env) as model:
            # add variables
            x = model.addVars(
                gp.tuplelist((h, l) for (h, l) in self.data.Hub_pairs),
                vtype=GRB.CONTINUOUS, name='x')
            y = model.addVars(
                gp.tuplelist((i, j) for (i, j) in self.data.Node_pairs_for_mdl_sp[r]),
                vtype=GRB.CONTINUOUS, name='y')

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
                (x[h, l] <= self.param_z[h, l] for (h, l) in self.data.Hub_pairs),
                name='conn_enable')

            model.addConstrs((x[h, l] <= 1 for (h, l) in self.data.Hub_pairs), name='x_ub')
            model.addConstrs((y[i, j] <= 1 for (i, j) in self.data.Node_pairs_for_mdl_sp[r]), name='y_ub')

            # set objective - follower
            obj_follower = gp.quicksum(
                self.data.tao_follower[h, l, s] * x[h, l] for (h, l) in self.data.Hub_pairs)
            obj_follower += gp.quicksum(
                self.data.gamma_follower[i, j, s] * y[i, j] for (i, j) in self.data.Node_pairs_for_mdl_sp[r])
            model.setObjective(obj_follower, sense=GRB.MINIMIZE)


            # model.setParam('TimeLimit', 360)  # 10 minutes
            model.setParam('LogToConsole', 0)
            model.optimize()

            x_sol = {(h,l): x[h,l].X for (h,l) in self.data.Hub_pairs if x[h,l].X > 0.1}
            y_sol = {(i,j): y[i,j].X for (i,j) in self.data.Node_pairs_for_mdl_sp[r] if y[i,j].X > 0.1}

            result = {
                'x_sol': x_sol,
                'y_sol': y_sol
            }

            return result

class MasterModel():
    def __init__(self, data: Modeldata, agg_cut: bool, solution_appro: bool = False, agg_cut_part=False, on_linux=False,
                 use_mdl_copy=False, copy_mdl=None, n_leg_trip=None, s_r_l=None, prefered_hubs=None,
                                  prefered_leader_obj=None,
                                  original_leader_obj=None):
        """
        Class of the master model
        :param data: modeldata
        :param agg_cut: whether using the aggregated cut
        :param linux: whether the code runs on Linux Server
        """
        parser = get_parser()
        self.args = parser.parse_args()
        self.original_MIP_gap = self.args.MP_ini_gap  # for i-CCG
        self.MIP_gap = self.args.MP_ini_gap  # initial MIP Gap, if use i-ccg, also solve MP to optimal at first iterations
        self.MP_timelimit = self.args.MP_timelimit  # initial MP time limit
        self.solved_to_optimal = False
        self.obj_lb_is_0 = True  # the initial lower bound of obj

        self.data = data
        self.agg_cut = agg_cut
        self.solution_appro = solution_appro
        self.agg_cut_part = agg_cut_part
        self.on_linux = on_linux
        self.n_leg_trip = n_leg_trip
        self.s_r_l = s_r_l
        self.prefered_hubs = prefered_hubs
        self.prefered_leader_obj = prefered_leader_obj
        self.original_leader_obj = original_leader_obj

        current_directory = os.path.dirname(os.path.abspath(__file__))
        upper_2_dir = os.path.dirname(current_directory)
        data_file = os.path.basename(data.file_name)[:-5]
        # log_file_prefix = upper_2_dir + '/output/' + data_file
        log_file_prefix = os.path.join(upper_2_dir, 'output')
        log_file_prefix = os.path.join(log_file_prefix, data_file)

        self.console_output_file = os.path.join(upper_2_dir, 'output', 'console_info_{}.txt'.format(data_file))

        if not use_mdl_copy:
            self.env = gp.Env(log_file_prefix + '_Master')
            self.model = gp.Model('MasterModel', self.env)
            if ':' not in os.path.abspath(__file__):
                current_directory = os.path.dirname(os.path.abspath(__file__))
                upper_2_dir = os.path.dirname(current_directory)  # resembles '..'
                # model_to_disk_file = os.path.join(upper_2_dir, 'model2disk')  # write the model to disk instead of memory
                # self.model.setParam('NodefileDir', model_to_disk_file)
                # self.model.setParam('NodefileStart', 0.45)  # 当使用内存达到45%时开始写入磁盘
            self.var = self.Variable(model=self.model, data=data, agg_cut=agg_cut, solution_appro=self.solution_appro, agg_cut_part=agg_cut_part, s_r_l=self.s_r_l)
            self.cons = self.Constraint()
            self.add_constraints()
            self.add_constr_explored_sp()
        else:
            self.model = copy_mdl.copy()
            self.var = self.VariableFramework(model=self.model, data=data)
            self.cons = self.Constraint()


        self.set_objective()
        self.set_params()


    class VariableFramework():
        """Master variables, the model is copied from somewhere"""
        def __init__(self, model, data):
            self.z = {}
            self.gamma = {}
            self.gamma_exist = True
            self.gamma_agg = None
            self.gamma_agg_part = None
            self.gamma_RHS = gp.LinExpr()
            self.gamma_RHS_part = None
            self.n_cut_per_class = 0
            self.n_gamma = 0
            # vars to be added during iterations
            self.theta1 = {}
            self.theta2 = {}
            self.theta3 = {}
            self.theta4 = {}
            self.delta = {}
            self.added_var_idx = []
            self.n_iters = 0  # number of iterations (theta1-4 will be added one time in each iteration)

            for (h,l) in data.Hub_pairs:
                self.z[h,l] = model.getVarByName(f'z[{h},{l}]')
            for (s,r) in data.Sce_Trip:
                self.gamma[s,r] = model.getVarByName(f'zeta[{s},{r}]')

    class Variable():
        """class of variables"""

        def __init__(self, model, data: Modeldata, agg_cut: bool, solution_appro: bool, agg_cut_part=False, s_r_l=None):
            # vars in the master problem
            self.z = model.addVars(data.Hub_pairs, vtype=GRB.BINARY, name='z')
            for h,l in data.Hub_pairs:
                self.z[h,l].setAttr('BranchPriority', 1)
            if agg_cut:
                if agg_cut_part:
                    self.n_cut_per_class = 5  # assume 10 cuts (of all scenarios) are aggregated
                    self.n_gamma = int(len(data.Trips)/self.n_cut_per_class) + 1
                    self.gamma_agg_part = model.addVars(range(self.n_gamma), vtype=GRB.CONTINUOUS, name='gamma_agg_part')
                    self.gamma_RHS_part = {i: gp.LinExpr() for i in range(self.n_gamma)}
                    self.gamma_exist = False
                    self.use_part_agg = True  # the single agg doesn't exist, but use partly aggerated cut
                else:
                    self.gamma_agg = model.addVar(vtype=GRB.CONTINUOUS, name='gamma')
                    self.gamma_RHS = gp.LinExpr()
                    self.gamma_exist = False
                    self.use_part_agg = False
            else:
                self.gamma = model.addVars(gp.tuplelist((s, r) for s in data.Scenarios for r in data.Trips),
                                       vtype=GRB.CONTINUOUS, name='gamma')
                self.gamma_exist = True
            # vars to be added during iterations
            self.theta1 = {}
            self.theta2 = {}
            self.theta3 = {}
            self.theta4 = {}
            self.delta = {}
            self.added_var_idx = []
            self.n_iters = 0  # number of iterations (theta1-4 will be added one time in each iteration)

            # add variables for explored subproblems
            self.delta_comb = model.addVars(s_r_l,
                                       vtype=GRB.BINARY, name='delta_comb')  # whether the leg is satisfied
            self.beta_comb = model.addVars(s_r_l,
                                      vtype=GRB.BINARY, name='beta_comb')  # whether the leg is selected


            # if solution_appro == 'primal':
            #     self.theta1 = model.addVars(
            #         gp.tuplelist((s, r, i) for s in data.Scenarios for r in data.Trips for i in data.Node),
            #         vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name='theta1')
            #     self.theta2 = model.addVars(
            #         gp.tuplelist((s, r, h, l) for s in data.Scenarios for r in data.Trips for (h, l) in data.Hub_pairs),
            #         vtype=GRB.CONTINUOUS, name='theta2')
            #     self.theta3 = model.addVars(
            #         gp.tuplelist((s, r, h, l) for s in data.Scenarios for r in data.Trips for (h, l) in data.Hub_pairs),
            #         vtype=GRB.CONTINUOUS, name='theta3')
            #     self.theta4 = model.addVars(gp.tuplelist(
            #         (s, r, i, j) for s in data.Scenarios for r in data.Trips for (i, j) in data.Node_pairs_for_mdl_sp[r]),
            #                                 vtype=GRB.CONTINUOUS, name='theta4')
            #     self.delta = model.addVars(
            #         gp.tuplelist((s, r, h, l) for s in data.Scenarios for r in data.Trips for (h, l) in data.Hub_pairs),
            #         vtype=GRB.CONTINUOUS, name='delta')


    class Constraint():
        """class of constraint"""
        def __init__(self):
            # constraints to be added
            self.gamma_lb = {}
            self.gamma_part_lb = {}
            self.theta_cons_1 = {}
            self.theta_cons_2 = {}
            self.delta_lin_1 = {}
            self.delta_lin_2 = {}
            self.delta_lin_3 = {}

    def add_constraints(self):
        with open(self.console_output_file, 'a') as file:
            print('------------add constraints in the initial MP', file=file)
            print('\t\tadd flow balance constraints...', file=file)
        self.cons.flow_balance = self.model.addConstrs((gp.quicksum(
            self.var.z[h, l] for l in self.data.Hub if (h, l) in self.data.Hub_pairs) == gp.quicksum(
            self.var.z[l, h] for l in self.data.Hub if (l, h) in self.data.Hub_pairs) for h in
                                                        self.data.Hub), name='flow_balance')
        if self.solution_appro == 'primal':
            with open(self.console_output_file, 'a') as file:
                print('\t\tadd primal special constraints...', file=file)
            # self.cons.theta_cons_1 = self.model.addConstrs((-self.var.theta1[s, r, h] + self.var.theta1[s, r, l] -
            #                                                 self.var.theta2[s, r, h, l] - self.var.theta3[s, r, h, l] <=
            #                                                 self.data.tao_follower[h, l, s] for s in self.data.Scenarios
            #                                                 for r in self.data.Trips for (h, l) in
            #                                                 self.data.Hub_pairs), name='theta_cons_1')
            # self.cons.theta_cons_2 = self.model.addConstrs((-self.var.theta1[s, r, i] + self.var.theta1[s, r, j] -
            #                                                 self.var.theta4[s, r, i, j] <= self.data.gamma_follower[
            #                                                     i, j, s] for s in self.data.Scenarios for r in
            #                                                 self.data.Trips for (i, j) in self.data.Node_pairs_for_mdl_sp[r]),
            #                                                name='theta_cons_2')
            #
            # self.cons.delta_lin_1 = self.model.addConstrs(
            #     (self.var.delta[s, r, h, l] <= self.var.z[h, l] * self.data.bigM for s in self.data.Scenarios for r in
            #      self.data.Trips for (h, l) in self.data.Hub_pairs), name='delta_lin_1')
            # self.cons.delta_lin_2 = self.model.addConstrs(
            #     (self.var.delta[s, r, h, l] <= self.var.theta2[s, r, h, l] for s in self.data.Scenarios for r in
            #      self.data.Trips for (h, l) in self.data.Hub_pairs), name='delta_lin_2')
            # self.cons.delta_lin_3 = self.model.addConstrs(
            #     (self.var.delta[s, r, h, l] >= self.var.theta2[s, r, h, l] - self.data.bigM * (1 - self.var.z[h, l]) for
            #      s in self.data.Scenarios for r in self.data.Trips for (h, l) in self.data.Hub_pairs),
            #     name='delta_lin_3')

    def add_constr_explored_sp(self):
        """
        add constraints for explored nodes
        :return:
        """

        self.cons.delta_1 = self.model.addConstrs(
            (self.var.delta_comb[s, r, leg] >= gp.quicksum(self.var.z[h, l] for h, l in self.prefered_hubs[s, r][leg]) -
                len(self.prefered_hubs[s, r][leg]) + 1 for s, r in self.data.Sce_Trip_explored for leg in range(self.n_leg_trip[s, r])), name='delta_1'
        )
        self.cons.delta_2 = self.model.addConstrs(
            (self.var.delta_comb[s, r, leg] <= self.var.z[h, l] for s, r in self.data.Sce_Trip_explored for leg in
             range(self.n_leg_trip[s, r]) for h, l in self.prefered_hubs[s, r][leg]), name='delta_2'
        )
        self.cons.beta_1 = self.model.addConstrs(
            (gp.quicksum(self.var.beta_comb[s, r, leg] for leg in range(self.n_leg_trip[s, r])) <= 1 for s, r in
             self.data.Sce_Trip_explored), name='beta_1'
        )
        self.cons.beta_2 = self.model.addConstrs(
            (self.var.beta_comb[s,r,leg] <= self.var.delta_comb[s,r,leg] for s, r in self.data.Sce_Trip_explored
             for leg in range(self.n_leg_trip[s, r])), name='beta_2'
        )
        self.cons.beta_lb = self.model.addConstrs(
            (gp.quicksum(self.var.beta_comb[s, r, leg] for leg in range(self.n_leg_trip[s, r])) >= gp.quicksum(
                self.var.delta_comb[s, r, leg] for leg in range(self.n_leg_trip[s, r])) / self.n_leg_trip[s, r] for s, r in
             self.data.Sce_Trip_explored if self.n_leg_trip[s, r] > 0), name='beta_lb'
        )
        for s,r in self.data.Sce_Trip_explored:
            if self.n_leg_trip[s, r] >= 2:
                # if there is only 1 bus leg routine, this constraint is not required
                for leg_i in range(self.n_leg_trip[s, r]-1):
                    for leg_j in range(leg_i, self.n_leg_trip[s, r]):
                        self.model.addConstr(
                            self.var.delta_comb[s, r, leg_i] - self.var.beta_comb[s, r, leg_i] <= 1 - self.var.beta_comb[s, r, leg_j],
                            name='delta_beta_comb_{}_{}_{}_{}'.format(s, r, leg_i, leg_j)
                        )

        for s,r in self.data.Sce_Trip_explored:
            if self.n_leg_trip[s,r] == 0:
                self.var.gamma[s,r].setAttr('lb', self.original_leader_obj[s,r])
                self.var.gamma[s,r].setAttr('ub', self.original_leader_obj[s,r])
            else:
                potential_M = list(self.prefered_leader_obj[s, r].values()) + [self.original_leader_obj[s, r]]
                bigM = max(potential_M)
                self.model.addConstrs(
                    (self.var.gamma[s, r] >= self.prefered_leader_obj[s, r][leg] - bigM * (
                                1 - self.var.beta_comb[s, r, leg]) for leg in range(self.n_leg_trip[s, r])),
                    name='omega_lb_{}_{}'.format(s, r)
                )
                self.model.addConstr(
                    (self.var.gamma[s, r] >= self.original_leader_obj[s, r] - bigM * (
                        gp.quicksum(self.var.beta_comb[s, r, leg] for leg in range(self.n_leg_trip[s, r])))),
                    name='omega_lb_no_bus_opened_{}_{}'.format(s, r)
                )

    def set_objective(self):
        with open(self.console_output_file, 'a') as file:
            print('\t\tset objective for MP...', file=file)
        if self.agg_cut:
            if self.agg_cut_part:
                self.obj = gp.quicksum(
                    self.data.Beta_hl[h, l] * self.var.z[h, l] for (h, l) in self.data.Hub_pairs) + gp.quicksum(self.var.gamma_agg_part)
            else:
                self.obj = gp.quicksum(
                    self.data.Beta_hl[h, l] * self.var.z[h, l] for (h, l) in self.data.Hub_pairs) + self.var.gamma_agg
        else:
            self.obj = gp.quicksum(self.data.Beta_hl[h, l] * self.var.z[h, l] for (h, l) in self.data.Hub_pairs) + gp.quicksum(
                self.data.Scenarios_prob[s] * gp.quicksum(
                    self.data.Trip_p_amount[s,r] * self.var.gamma[s, r] for r in self.data.Trips) for s in
                self.data.Scenarios)
        self.model.setObjective(self.obj)

    def restrict_bound_obj(self, lb, ub):
        """
        assign an UB and a LB for obj
        :return:
        """
        self.model.addConstr(self.obj <= ub, name='obj_ub')
        self.model.addConstr(self.obj >= lb, name='obj_lb')


    def add_var_and_cons_ccg_dual(self, varValue, s, r):
        """
        add column and constraints given s and r
        :param varValue: to obtain the value of vars pi and t
        :param s: scenario
        :param r: trip
        :param algo_type: 'dual': CCG based on dual sub models; 'primal': CCG based on primal sub models
        :return: no return
        """
        theta1_name = 'theta1_{}_{}_{}'.format(s, r, self.var.n_iters)
        self.var.theta1[theta1_name] = self.model.addVars(self.data.Node_of_trip[r], vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY,
                                                          name=theta1_name)
        theta2_name = 'theta2_{}_{}_{}'.format(s, r, self.var.n_iters)
        self.var.theta2[theta2_name] = self.model.addVars(self.data.Hub_pairs, vtype=GRB.CONTINUOUS, name=theta2_name)
        theta3_name = 'theta3_{}_{}_{}'.format(s, r, self.var.n_iters)
        self.var.theta3[theta3_name] = self.model.addVars(self.data.Hub_pairs, vtype=GRB.CONTINUOUS, name=theta3_name)
        theta4_name = 'theta4_{}_{}_{}'.format(s, r, self.var.n_iters)
        self.var.theta4[theta4_name] = self.model.addVars(self.data.Node_pairs_for_mdl_sp[r], vtype=GRB.CONTINUOUS, name=theta4_name)
        delta_name = 'delta_{}_{}_{}'.format(s, r, self.var.n_iters)
        self.var.delta[delta_name] = self.model.addVars(self.data.Hub_pairs, vtype=GRB.CONTINUOUS, name=delta_name)

        if self.agg_cut:
            if self.agg_cut_part:
                raise "undeveloped: partly aggregated cut in add_var_and_cons_ccg_dual"
            else:
                self.var.gamma_RHS += self.data.Scenarios_prob[s] * self.data.Trip_p_amount[s,r] * (
                            -varValue.pi1[s, r, self.data.Trip_origin[r]] + varValue.pi1[s, r,
                    self.data.Trip_destination[r]] - gp.quicksum(
                        self.var.z[h, l] * varValue.pi2[s, r, h, l] for (h, l) in self.data.Hub_pairs) - gp.quicksum(
                        varValue.pi3[s, r, h, l] for (h, l) in self.data.Hub_pairs) - gp.quicksum(
                        varValue.pi4[s, r, i, j] for (i, j) in self.data.Node_pairs_for_mdl_sp[r]) + self.var.theta1[theta1_name][
                                self.data.Trip_origin[r]] - self.var.theta1[theta1_name][
                                self.data.Trip_destination[r]] + gp.quicksum(
                        self.var.delta[delta_name]) + gp.quicksum(
                        self.var.theta3[theta3_name]) + gp.quicksum(self.var.theta4[theta4_name]))
        else:
            self.cons.gamma_lb['gamma_lb_{}_{}_{}'.format(s, r, self.var.n_iters)] = self.model.addConstr(
                self.var.gamma[s, r] >= -varValue.pi1[s, r, self.data.Trip_origin[r]] + varValue.pi1[s, r,
                self.data.Trip_destination[r]] - gp.quicksum(
                    self.var.z[h, l] * varValue.pi2[s, r, h, l] for (h, l) in self.data.Hub_pairs) - gp.quicksum(
                    varValue.pi3[s, r, h, l] for (h, l) in self.data.Hub_pairs) - gp.quicksum(
                    varValue.pi4[s, r, i, j] for (i, j) in self.data.Node_pairs_for_mdl_sp[r]) + self.var.theta1[theta1_name][
                    self.data.Trip_origin[r]] - self.var.theta1[theta1_name][
                    self.data.Trip_destination[r]] + gp.quicksum(
                    self.var.delta[delta_name]) + gp.quicksum(
                    self.var.theta3[theta3_name]) + gp.quicksum(self.var.theta4[theta4_name]),
                name='gamma_lb_{}_{}_{}'.format(s,r,self.var.n_iters))

        self.cons.theta_cons_1['theta_cons_1_{}_{}_{}'.format(s,r,self.var.n_iters)] = self.model.addConstrs((
            -self.var.theta1[theta1_name][h] + self.var.theta1[theta1_name][l] - self.var.theta2[theta2_name][h, l] -
            self.var.theta3[theta3_name][h, l] <= self.data.tao_follower[h, l, s] * varValue.t[s, r] for (h, l) in
            self.data.Hub_pairs), name='theta_cons_1_{}_{}_{}'.format(s,r,self.var.n_iters))
        self.cons.theta_cons_2['theta_cons_2_{}_{}_{}'.format(s,r,self.var.n_iters)] = self.model.addConstrs((
            -self.var.theta1[theta1_name][i] + self.var.theta1[theta1_name][j] - self.var.theta4[theta4_name][i, j] <=
            self.data.gamma_follower[i, j, s] * varValue.t[s, r] for (i, j) in self.data.Node_pairs_for_mdl_sp[r]),
            name='theta_cons_2_{}_{}_{}'.format(s,r,self.var.n_iters))
        self.cons.delta_lin_1['delta_lin_1_{}_{}_{}'.format(s,r,self.var.n_iters)] = self.model.addConstrs((
            self.var.delta[delta_name][h, l] <= self.var.z[h, l] * self.data.bigM for (h, l) in self.data.Hub_pairs),
            name='delta_lin_1_{}_{}_{}'.format(s,r,self.var.n_iters))

        self.cons.delta_lin_2['delta_lin_2_{}_{}_{}'.format(s,r,self.var.n_iters)] = self.model.addConstrs((
            self.var.delta[delta_name][h, l] <= self.var.theta2[theta2_name][h, l] for (h, l) in self.data.Hub_pairs),
            name='delta_lin_2_{}_{}_{}'.format(s,r,self.var.n_iters))

        self.cons.delta_lin_3['delta_lin_3_{}_{}_{}'.format(s, r, self.var.n_iters)] = self.model.addConstrs((
            self.var.delta[delta_name][h, l] >= self.var.theta2[theta2_name][h, l] - self.data.bigM * (
                    1 - self.var.z[h, l]) for (h, l) in self.data.Hub_pairs),
            name='delta_lin_3_{}_{}_{}'.format(s, r, self.var.n_iters))

    def add_combinatorial_cut(self, varValue, s, r):
        """
        add combinatorial cut
        :param varValue: to obtain the value of vars pi and t
        :param s: scenario
        :param r: trip
        :return: no return
        """

        cut_name = 'combinatorial_{}_{}_{}'.format(s, r, self.var.n_iters)
        z_idx_zero = [(h,l) for (h,l) in self.data.Hub_pairs if varValue.z.get((h,l), 0) < 0.5]
        z_idx_one = [(h,l) for (h,l) in self.data.Hub_pairs if varValue.z.get((h,l), 0) >= 0.5]
        self.model.addConstr(
            (self.var.gamma[s, r] >= varValue.latest_sub_obj[s, r] - self.data.bigM * (
                gp.quicksum(self.var.z[h, l] for (h, l) in z_idx_zero)) - self.data.bigM * gp.quicksum(
                1 - self.var.z[h, l] for (h, l) in z_idx_one)), name=cut_name)

    def add_theta_vars_callback(self):
        """
        add theta vars. They will be used in callback functions
        :return:
        """
        for (s,r) in self.data.Sce_Trip:
            self.var.theta1.update(
                self.model.addVars(gp.tuplelist((s, r, i) for i in self.data.Node), vtype=GRB.CONTINUOUS,
                                   lb=-GRB.INFINITY, name='theta1[{},{}]'.format(s, r))
            )
            self.var.theta2.update(self.model.addVars(
                gp.tuplelist((s, r, h, l) for (h, l) in self.data.Hub_pairs), vtype=GRB.CONTINUOUS,
                name='theta2[{},{}]'.format(s, r)))
            self.var.theta3.update(
                self.model.addVars(gp.tuplelist((s, r, h, l) for (h, l) in self.data.Hub_pairs), vtype=GRB.CONTINUOUS,
                                   name='theta3[{},{}]'.format(s, r)))
            self.var.theta4.update(
                self.model.addVars(gp.tuplelist((s, r, i, j) for (i, j) in self.data.Node_pairs_for_mdl_sp[r]),
                                   vtype=GRB.CONTINUOUS,
                                   name='theta4[{},{}]'.format(s, r)))
            self.var.delta.update(
                self.model.addVars(gp.tuplelist((s, r, h, l) for (h, l) in self.data.Hub_pairs), vtype=GRB.CONTINUOUS,
                                   name='delta[{},{}]'.format(s, r)))
            self.var.added_var_idx.append((s,r))
            # add associated constraints
            self.cons.theta_cons_1.update(self.model.addConstrs((-self.var.theta1[s, r, h] + self.var.theta1[s, r, l] -
                                                            self.var.theta2[s, r, h, l] - self.var.theta3[s, r, h, l] <=
                                                            self.data.tao_follower[h, l, s] for (h, l) in
                                                            self.data.Hub_pairs), name=f'theta_cons_1[{s},{r}]'))
            self.cons.theta_cons_2.update(self.model.addConstrs((-self.var.theta1[s, r, i] + self.var.theta1[s, r, j] -
                                                            self.var.theta4[s, r, i, j] <= self.data.gamma_follower[
                                                                i, j, s] for (i, j) in self.data.Node_pairs_for_mdl_sp[r]),
                                                           name=f'theta_cons_2[{s},{r}]'))

            self.cons.delta_lin_1.update(self.model.addConstrs(
                (self.var.delta[s, r, h, l] <= self.var.z[h, l] * self.data.bigM for (h, l) in self.data.Hub_pairs),
                name=f'delta_lin_1[{s},{r}]'))
            self.cons.delta_lin_2.update(self.model.addConstrs(
                (self.var.delta[s, r, h, l] <= self.var.theta2[s, r, h, l] for (h, l) in self.data.Hub_pairs),
                name=f'delta_lin_2[{s},{r}]'))
            self.cons.delta_lin_3.update(self.model.addConstrs(
                (self.var.delta[s, r, h, l] >= self.var.theta2[s, r, h, l] - self.data.bigM * (1 - self.var.z[h, l]) for
                 (h, l) in self.data.Hub_pairs),
                name=f'delta_lin_3[{s},{r}]'))

    def add_var_and_cons_ccg_primal(self, varValue, s, r):
        # add auxilary variables
        if (s,r) not in self.var.added_var_idx:
            self.var.theta1.update(
                self.model.addVars(gp.tuplelist((s, r, i) for i in self.data.Node), vtype=GRB.CONTINUOUS,
                                   lb=-GRB.INFINITY, name='theta1[{},{}]'.format(s, r))
            )
            self.var.theta2.update(self.model.addVars(
                gp.tuplelist((s, r, h, l) for (h, l) in self.data.Hub_pairs), vtype=GRB.CONTINUOUS,
                name='theta2[{},{}]'.format(s, r)))
            self.var.theta3.update(
                self.model.addVars(gp.tuplelist((s, r, h, l) for (h, l) in self.data.Hub_pairs), vtype=GRB.CONTINUOUS,
                                   name='theta3[{},{}]'.format(s, r)))
            self.var.theta4.update(
                self.model.addVars(gp.tuplelist((s, r, i, j) for (i, j) in self.data.Node_pairs_for_mdl_sp[r]),
                                   vtype=GRB.CONTINUOUS,
                                   name='theta4[{},{}]'.format(s, r)))
            self.var.delta.update(
                self.model.addVars(gp.tuplelist((s, r, h, l) for (h, l) in self.data.Hub_pairs), vtype=GRB.CONTINUOUS,
                                   name='delta[{},{}]'.format(s, r)))
            self.var.added_var_idx.append((s,r))
            # add associated constraints
            self.cons.theta_cons_1.update(self.model.addConstrs((-self.var.theta1[s, r, h] + self.var.theta1[s, r, l] -
                                                            self.var.theta2[s, r, h, l] - self.var.theta3[s, r, h, l] <=
                                                            self.data.tao_follower[h, l, s] for (h, l) in
                                                            self.data.Hub_pairs), name=f'theta_cons_1[{s},{r}]'))
            self.cons.theta_cons_2.update(self.model.addConstrs((-self.var.theta1[s, r, i] + self.var.theta1[s, r, j] -
                                                            self.var.theta4[s, r, i, j] <= self.data.gamma_follower[
                                                                i, j, s] for (i, j) in self.data.Node_pairs_for_mdl_sp[r]),
                                                           name=f'theta_cons_2[{s},{r}]'))

            self.cons.delta_lin_1.update(self.model.addConstrs(
                (self.var.delta[s, r, h, l] <= self.var.z[h, l] * self.data.bigM for (h, l) in self.data.Hub_pairs),
                name=f'delta_lin_1[{s},{r}]'))
            self.cons.delta_lin_2.update(self.model.addConstrs(
                (self.var.delta[s, r, h, l] <= self.var.theta2[s, r, h, l] for (h, l) in self.data.Hub_pairs),
                name=f'delta_lin_2[{s},{r}]'))
            self.cons.delta_lin_3.update(self.model.addConstrs(
                (self.var.delta[s, r, h, l] >= self.var.theta2[s, r, h, l] - self.data.bigM * (1 - self.var.z[h, l]) for
                 (h, l) in self.data.Hub_pairs),
                name=f'delta_lin_3[{s},{r}]'))


        if self.agg_cut:
            if self.agg_cut_part:
                trip_id = self.data.Trips.index(r)
                self.var.gamma_RHS_part[int(trip_id/self.var.n_cut_per_class)] += self.data.Scenarios_prob[s] * self.data.Trip_p_amount[s,r] * (
                        -varValue.pi1[s, r, self.data.Trip_origin[r]] + varValue.pi1[s, r,
                self.data.Trip_destination[r]] - gp.quicksum(
                    self.var.z[h, l] * varValue.pi2[s, r, h, l] for (h, l) in self.data.Hub_pairs) - gp.quicksum(
                    varValue.pi3[s, r, h, l] for (h, l) in self.data.Hub_pairs) - gp.quicksum(
                    varValue.pi4[s, r, i, j] for (i, j) in self.data.Node_pairs_for_mdl_sp[r]) + varValue.t[s, r] * (
                                self.var.theta1[s, r, self.data.Trip_origin[r]] -
                                self.var.theta1[s, r, self.data.Trip_destination[r]] + gp.quicksum(
                            self.var.delta[s, r, h, l] for (h, l) in self.data.Hub_pairs) + gp.quicksum(
                            self.var.theta3[s, r, h, l] for (h, l) in self.data.Hub_pairs) + gp.quicksum(
                            self.var.theta4[s, r, i, j] for (i, j) in self.data.Node_pairs_for_mdl_sp[r])))
            else:
                self.var.gamma_RHS += self.data.Scenarios_prob[s] * self.data.Trip_p_amount[s,r] * (
                        -varValue.pi1[s, r, self.data.Trip_origin[r]] + varValue.pi1[s, r,
                self.data.Trip_destination[r]] - gp.quicksum(
                    self.var.z[h, l] * varValue.pi2[s, r, h, l] for (h, l) in self.data.Hub_pairs) - gp.quicksum(
                    varValue.pi3[s, r, h, l] for (h, l) in self.data.Hub_pairs) - gp.quicksum(
                    varValue.pi4[s, r, i, j] for (i, j) in self.data.Node_pairs_for_mdl_sp[r]) + varValue.t[s, r] * (
                                self.var.theta1[s, r, self.data.Trip_origin[r]] -
                                self.var.theta1[s, r, self.data.Trip_destination[r]] + gp.quicksum(
                            self.var.delta[s, r, h, l] for (h, l) in self.data.Hub_pairs) + gp.quicksum(
                            self.var.theta3[s, r, h, l] for (h, l) in self.data.Hub_pairs) + gp.quicksum(
                            self.var.theta4[s, r, i, j] for (i, j) in self.data.Node_pairs_for_mdl_sp[r])))

        else:
            self.cons.gamma_lb['gamma_lb_{}_{}_{}'.format(s, r, self.var.n_iters)] = self.model.addConstr(
                self.var.gamma[s, r] >= -varValue.pi1[s, r, self.data.Trip_origin[r]] + varValue.pi1[s, r,
                self.data.Trip_destination[r]] - gp.quicksum(
                    self.var.z[h, l] * varValue.pi2[s, r, h, l] for (h, l) in self.data.Hub_pairs) - gp.quicksum(
                    varValue.pi3[s, r, h, l] for (h, l) in self.data.Hub_pairs) - gp.quicksum(
                    varValue.pi4[s, r, i, j] for (i, j) in self.data.Node_pairs_for_mdl_sp[r]) + varValue.t[s, r] * (
                        self.var.theta1[s, r, self.data.Trip_origin[r]] -
                        self.var.theta1[s, r, self.data.Trip_destination[r]] + gp.quicksum(
                    self.var.delta[s, r, h, l] for (h, l) in self.data.Hub_pairs) + gp.quicksum(
                    self.var.theta3[s, r, h, l] for (h, l) in self.data.Hub_pairs) + gp.quicksum(
                    self.var.theta4[s, r, i, j] for (i, j) in self.data.Node_pairs_for_mdl_sp[r])),
                name='gamma_lb_{}_{}_{}'.format(s, r, self.var.n_iters))

    def update_var_index(self):
        if self.agg_cut:
            if self.agg_cut_part:
                for i in range(self.var.n_gamma):
                    self.cons.gamma_part_lb['gamma_lb_id{}_iter{}'.format(i, self.var.n_iters)] = self.model.addConstr(
                        self.var.gamma_agg_part[i] >= self.var.gamma_RHS_part[i],
                        name='gamma_lb_id{}_iter{}'.format(i, self.var.n_iters)
                    )
                    self.var.gamma_RHS_part[i] = gp.LinExpr()

            else:
                self.cons.gamma_lb['gamma_lb_{}'.format(self.var.n_iters)] = self.model.addConstr(
                    self.var.gamma_agg >= self.var.gamma_RHS, name='gamma_lb_{}'.format(self.var.n_iters))
                self.var.gamma_RHS = gp.LinExpr()

        self.var.n_iters += 1

    def fix_z(self, given_z: dict):
        """
        fix z
        :param given_z: a dictionary, but the keys are str
        :return:
        """
        for key in self.var.z:
            if str(key) in given_z.keys():
                self.var.z[key].setAttr('lb', 1)
                self.var.z[key].setAttr('ub', 1)
            else:
                self.var.z[key].setAttr('lb', 0)
                self.var.z[key].setAttr('ub', 0)

    def free_z(self):
        for key in self.var.z:
            self.var.z[key].setAttr('lb', 0)
            self.var.z[key].setAttr('ub', 1)

    def solve_model(self, use_i_ccg=False, remaining_time=1e10, supposed_time_limit=1e11):
        if remaining_time < supposed_time_limit:
            new_timelimit = max(int(remaining_time), 1)
            self.model.setParam('TimeLimit', new_timelimit)
            with open(self.console_output_file, 'a') as file:
                print('Confirm that MP timilimit is set to {:.4f}'.format(new_timelimit), file=file)
        self.model.optimize()
        t_optimize = time.time()
        while self.model.solCount <= 0 and self.model.Status != 3:
            if t_optimize > supposed_time_limit:
                with open(self.console_output_file, 'a') as file:
                    print('\n\n!!!!!!!!!!!!!!no solution for MP is found!!!!!!!!!!!!!!\n\n', file=file)
                    self.solved_to_optimal = False
                return
            tmp = time.time()
            with open(self.console_output_file, 'a') as file:
                print('No feasible MP solution is found, continue to optimize', file=file)
            self.model.optimize()
            t_optimize += time.time() - tmp

        if self.model.Status == 3:
            with open(self.console_output_file, 'a') as file:
                print('\n\n!!!!!!!!!!!!!!MP infeasible!!!!!!!!!!!!!!\n\n', file=file)
            self.model.setParam('TimeLimit', 60)
            self.model.computeIIS()
            current_directory = os.path.dirname(os.path.abspath(__file__))
            upper_2_dir = os.path.dirname(current_directory)
            data_file = os.path.basename(self.data.file_name)[:-5]
            # log_file_prefix = upper_2_dir + '/output/' + data_file
            log_file_prefix = os.path.join(upper_2_dir, 'output')
            log_file_prefix = os.path.join(log_file_prefix, data_file)
            ilp_file_path = log_file_prefix+'_iis.ilp'
            self.model.write(ilp_file_path)


        if abs(self.model.MIPGap) < 5e-5:
            # if Status code is 2, that means gap is in the permitted MIPGap rather really optimal!!
            self.solved_to_optimal = True
        else:
            self.solved_to_optimal = False
        with open(self.console_output_file, 'a') as file:
            print('Master model is solved, gap = {}'.format(self.model.MIPGap), file=file)
        # self.present_solutions()

    def update_obj_lb(self, obj_lb=0):
        """
        update the lower bound of obj
        :return:
        """
        c = self.model.getConstrByName('obj_lb')
        if c:
            self.model.remove(c)
        self.model.addConstr(self.obj >= obj_lb, name='obj_lb')

    def assign_MIP_gap_constant(self, gap):
        with open(self.console_output_file, 'a') as file:
            print("\t\t set param: MIP_gap =", gap, file=file)
        self.model.setParam('MIPGap', gap)
    def update_MIP_gap(self, scale, to_original=False):
        self.MIP_gap = self.MIP_gap * scale
        if not to_original:
            with open(self.console_output_file, 'a') as file:
                print("\t\t set param: MIP_gap =", self.MIP_gap, file=file)
            self.model.setParam('MIPGap', self.MIP_gap)
        if to_original:
            self.model.setParam('MIPGap', self.original_MIP_gap)
            with open(self.console_output_file, 'a') as file:
                print("\t\t set param MIP_gap to initial value=", self.original_MIP_gap, file=file)

    # def update_MP_timelimit(self, dt):
    #     self.MP_timelimit += dt
    #     self.model.setParam('TimeLimit', self.MP_timelimit)

    def set_params(self):
        self.model.setParam('TimeLimit', self.MP_timelimit)
        # self.model.setParam('FeasibilityTo', 1e-1)
        # self.model.setParam('Presolve', 0)
        self.model.setParam('LogToConsole', 0)
        self.model.setParam('Heuristics', 0.1)  # default is 0.05
        # self.model.setParam('PrePasses', 3)  # limit the time on presolve
        # self.model.setParam('Cuts', 2)  # aggressive cut generation
        # self.model.setParam('MIPFocus', 1)  # finding feasible solutions quickly
        self.model.setParam('MIPGap', self.MIP_gap)  # MIP Gap
        # self.model.setParam('Method', 0)  # primal simplex for the initial root relaxation, otherwise barrier will be chosed, which is numerically sensitive

    def set_params_additional_exact_ccg(self, ccg_time_limit):
        """
        set parameters for MP when using exact CCG
        :return:
        """
        self.model.setParam('TimeLimit', ccg_time_limit)
        self.MP_timelimit = ccg_time_limit
        self.model.setParam('MIPGap', 0)  # MIP Gap

    def update_master_timelimit_1_h(self):
        self.model.setParam('TimeLimit', 3600)
        self.model.setParam('MIPGap', 1e-5)

    def present_solutions(self):
        print('print the solutions of master problem: z')
        for key in self.var.z:
            if self.var.z[key].X > 1e-4:
                print(key, self.var.z[key].X)



class IterateComb:
    """
    Benders iterations
    - Use combinatorial cuts for data.Sce_trip_explored
    - Use CCG cuts for data.Sce_trip_unexplored
    """

    def __init__(self, data: Modeldata, agg_cut: bool=False, use_lp_basis=False, parallel_method='normal',
                 solution_appro='dual', primal_based_dual_to_ini=False, agg_cut_part=False, solve_MP_use_Benders=False,
                 use_mdl_copy=False, copy_mdl=None, revise_dual_value=False, n_tree_iter_max=100):
        """
        :param data:
        :param agg_cut:
        :param use_lp_basis: use lp basis as warm start
        :param parallel_method: 'normal': solve in sequence; 'thread': threading; 'process': multiprocessing
        :param solution_appro: 'dual': approach1, based on dual formulation; 'primal': approach 2, based on primal formulation
        :param primal_based_dual_to_ini: use primal approach, but also solve dual at the initail scenario
        """
        current_directory = os.path.dirname(os.path.abspath(__file__))
        upper_2_dir = os.path.dirname(current_directory)
        self.console_output_file = os.path.join(upper_2_dir, 'output', 'console_info_{}.txt'.format(os.path.basename(data.file_name)[:-5]))
        with open(self.console_output_file, 'a') as file:
            print('initiate data...', file=file)
        self.data = data
        self.data.output_to_disk()  # output the information to the disk
        self.n_tree_iter_max = n_tree_iter_max
        self.time_preprocess_tree_search = time.time()
        self.find_hub_legs()
        self.time_preprocess_tree_search = time.time() - self.time_preprocess_tree_search
        self.data.clear_output_disk()  # clear the temporary information in the disk
        self.n_leg_trip = {(s, r): len(self.prefered_leader_obj[s, r]) for s, r in
                           self.data.Sce_Trip}  # number of potential legs per subproblem (s,r)
        self.s_r_l = [(s, r, l) for s, r in data.Sce_Trip for l in range(self.n_leg_trip[s, r]) if
                      self.n_leg_trip[s, r] > 0]  # set (s,r,l) where l is the number of potential bus routines in problem s,r
        self.agg_cut = agg_cut
        self.use_lp_basis = use_lp_basis
        self.parallel_method = parallel_method
        self.solve_MP_use_Benders = solve_MP_use_Benders
        self.record = {}  # store the solution information

        if self.parallel_method not in ['normal', 'thread', 'process', 'process_1_core']:
            with open(self.console_output_file, 'a') as file:
                print('Unknown parallel method: {}, please confirm. Using normal instead'.format(self.parallel_method), file=file)
            self.parallel_method = 'normal'

        self.solution_appro = solution_appro
        if self.solution_appro not in ['primal', 'dual']:
            raise 'unknown parameter solution_appro in function start_Benders_iteration()'
        self.primal_based_dual_to_ini = primal_based_dual_to_ini
        self.revise_dual_value = revise_dual_value

        with open(self.console_output_file, 'a') as file:
            if ':' in os.path.abspath(__file__):
                self.run_in_linux_server = False
                print('The code is running on a Windows System', file=file)
            else:
                self.run_in_linux_server = True
                print('The code is running on a Linux System', file=file)
        with open(self.console_output_file, 'a') as file:
            print('create MP model...', file=file)
        tmp = time.time()
        if not self.solve_MP_use_Benders:
            self.MP = MasterModel(data=self.data, agg_cut=agg_cut, solution_appro=self.solution_appro,
                                  agg_cut_part=agg_cut_part, on_linux=self.run_in_linux_server,
                                  use_mdl_copy=use_mdl_copy, copy_mdl=copy_mdl, n_leg_trip=self.n_leg_trip,
                                  s_r_l=self.s_r_l, prefered_hubs=self.prefered_hubs,
                                  prefered_leader_obj=self.prefered_leader_obj,
                                  original_leader_obj=self.original_leader_obj)
        else:
            with open(self.console_output_file, 'a') as file:
                print('\t\tSolve MP with Benders is not correct. Exit.', file=file)
            exit()
        self.time_MP_creation = time.time() - tmp  # time cost including adding vars and constrs

        with open(self.console_output_file, 'a') as file:
            print('\t\tdeclare empty lists and dicts...', file=file)
        tmp_s = time.time()
        self.s_c_cut_stat = {(s,r):0 for (s,r) in self.data.Sce_Trip}
        self.dict_SP = {}  # for approach 1
        self.dict_SP_dual = {}  # for approach 2
        self.dict_SP_primal = {}  # for approach 2
        self.dict_originalSP = {}
        self.dict_CCGSP = {}
        self.create_submodels()
        self.time_SP_creation = time.time() - tmp_s  # time cost including adding vars and constrs
        self.time_mdl_creation = time.time() - tmp  # time for MP and SP model creation
        self.varValue = VarValue()
        self.LB_record = []  # record of lower bound
        self.UB_record = []  # record of upper bound
        self.time_record = []
        self.time_start = time.time()  # start time
        self.time_record_solve_MP = []
        self.time_record_solve_SP = []
        self.SP_VBasis = {}  # VBasis of subproblems, to accelerate the parallel computing
        self.SP_CBasis = {}
        self.VBasis_info = {}  # VBasis information
        self.time_retrieve_MP_solution = []
        self.time_retrieve_SP_solution = []
        self.time_update_MP = []
        self.time_update_SP = []
        self.list_n_cut_in_iter = []

    def find_hub_legs(self):
        """
        (s,r) in self.data.Sce_Trip_unexplored need to be handeled by CCG
        (s,r) in self.data.Sce_Trip_explred need combinatorial cuts
        :return:
        """
        hub_finder = PotentialHubFinder(data=self.data, n_iter_max=self.n_tree_iter_max)
        self.prefered_hubs, self.prefered_leader_obj, self.original_leader_obj, self.unexplred_probles, self.search_time_rec, self.search_iter_rec = hub_finder.find_hub_leg_all()  # find all potential hubs
        self.unexplred_probles = {key: value for key,value in self.unexplred_probles.items() if value}
        self.data.Sce_Trip_unexplored = list(set(self.unexplred_probles.keys()))
        self.data.Sce_Trip_unexplored.sort()
        self.data.Sce_Trip_explored = list(set(self.data.Sce_Trip) - set(self.data.Sce_Trip_unexplored))
        self.data.Sce_Trip_explored.sort()

        # #TODO: test, output the trips that used buses. Delete later 20250422
        #
        # trip_id = [r for (s,r) in self.prefered_hubs if len(self.prefered_hubs[s,r]) > 0]
        # trip_o = [self.data.Trip_origin[r] for r in trip_id]
        # trip_d = [self.data.Trip_destination[r] for r in trip_id]
        # amount = [self.data.Trip_p_amount[1,r] for r in trip_id]
        # hub_routes = [len(self.prefered_hubs[1,r]) for r in trip_id]
        # not_fully_explored = [self.unexplred_probles.get(r,False) for r in trip_id]
        # trip_info = {
        #     'trip_id': trip_id,
        #     'trip_o': trip_o,
        #     'trip_d': trip_d,
        #     'amount': amount,
        #     'hub_routes': hub_routes,
        #     'not_fully_explored':not_fully_explored,
        # }
        # df = pd.DataFrame.from_dict(trip_info)
        #
        # current_directory = os.path.dirname(os.path.abspath(__file__))
        # upper_2_dir = os.path.dirname(current_directory)
        # excel_file = os.path.join(upper_2_dir, 'output', 'find_bus_users_{}.xlsx'.format(os.path.basename(self.data.file_name)[:-5]))
        # df.to_excel(excel_file, index=False)




    def create_submodels(self):
        if self.solution_appro == 'dual':
            if self.parallel_method in ['normal', 'thread']:
                # created model in advance only in normal and thread parallel
                for s in self.data.Scenarios:
                    for r in self.data.Trips:
                        self.dict_SP[s, r] = SubModel(data=self.data, s=s, r=r)
                        self.dict_SP[s, r].declare_model()
            elif self.parallel_method in ['process', 'process_1_core']:
                self.SP_template = SubModel(data=self.data, s=self.data.Scenarios[0], r=self.data.Trips[0])
        elif self.solution_appro == 'primal':
            if self.parallel_method in ['normal', 'thread']:
                # created model in advance only in normal and thread parallel
                for s in self.data.Scenarios:
                    for r in self.data.Trips:
                        self.dict_SP_primal[s, r] = SubShortestPathWithStrongDualModel(data=self.data, s=s, r=r)
                        self.dict_SP_primal[s, r].declare_model()
                for r in self.data.Trips:
                    # only create one dual-based model
                    self.dict_SP_dual[self.data.Scenarios[0], r] = SubModel(data=self.data, s=self.data.Scenarios[0], r=r)
                    self.dict_SP_dual[self.data.Scenarios[0], r].declare_model()

            elif self.parallel_method in ['process', 'process_1_core']:
                self.SP_template_dual = SubModel(data=self.data, s=self.data.Scenarios[0], r=self.data.Trips[0])
                self.SP_template_primal = SubShortestPathWithStrongDualModel(data=self.data, s=self.data.Scenarios[0],
                                                                             r=self.data.Trips[0])

    def solve_subproblems_primal(self, n_iters, use_dual_also:bool=True, need_x = False, revise_dual_value=False):
        """solve subproblems: approach 2 based on primal formulation
        :param use_dual_also: use dual info to warm start
        """
        if self.parallel_method == 'normal':
            time_update_SP = 0
            time_record_solve_SP = 0
            time_retrieve_SP_solution = 0
            # in the first iteration, solve without warmstart
            for s,r in self.data.Sce_Trip_unexplored:
                # for r in self.data.Trips:
                if not use_dual_also:  # solve only primal models
                    tmp = time.time()
                    self.dict_SP_primal[s,r].update_objective_cons(param_z=self.varValue.z)
                    time_update_SP += time.time() - tmp

                    tmp = time.time()
                    if self.use_lp_basis:
                        self.dict_SP_primal[s, r].write_lp_basis_info(True, self.SP_VBasis.get(r, None),
                                                                      self.SP_CBasis.get(r, None))
                    else:
                        self.dict_SP_primal[s, r].write_lp_basis_info()
                    time_update_SP += time.time() - tmp
                    tmp = time.time()
                    self.dict_SP_primal[s, r].model.optimize()
                    time_record_solve_SP += time.time() - tmp
                    tmp = time.time()
                    self.dict_SP_primal[s, r].change_obj()  # change obj to leader params
                    time_update_SP += time.time() - tmp
                    tmp = time.time()
                    self.dict_SP_primal[s, r].model.optimize()
                    time_record_solve_SP += time.time() - tmp
                    tmp = time.time()
                    result_primal = self.dict_SP_primal[s, r].obtain_solution_info(self.use_lp_basis)
                    time_retrieve_SP_solution += time.time() - tmp

                    tmp = time.time()
                    self.varValue.update_Sub_var_primal(s, r,
                                                        x = result_primal['x'],
                                                        y = result_primal['y'],
                                                        pi1=result_primal['pi1'],
                                                        pi2=result_primal['pi2'],
                                                        pi3=result_primal['pi3'],
                                                        pi4=result_primal['pi4'],
                                                        t=result_primal['t'],
                                                        obj=result_primal['obj'])#,
                                                        # VBasis=result_primal['VBasis'],
                                                        # CBasis=result_primal['CBasis']
                    time_retrieve_SP_solution += time.time() - tmp


                else:
                    with open(self.console_output_file, 'a') as f:
                        print('This function is not verified yet: Solve Subproblems using dual info.', file=f)



            self.time_update_SP.append(time_update_SP)
            self.time_record_solve_SP.append(time_record_solve_SP)
            self.time_retrieve_SP_solution.append(time_retrieve_SP_solution)

        elif self.parallel_method == 'process':
            time_record_solve_SP = 0
            time_retrieve_SP_solution = 0
            tmp = time.time()
            if (not use_dual_also) or (use_dual_also and n_iters == 1):
            # use only primal model
                # solve scenario 1 and get warm start info
                pool = mp.Pool(processes=ALLOW_CORE)
                if not revise_dual_value:
                    if self.use_lp_basis:
                        # if use lp basis as warm start
                        result = pool.starmap(self.SP_template_primal.create_and_solve_by_with,
                                              [(self.data.Scenarios[0], r, self.varValue.z, True, self.SP_VBasis.get(r, None),
                                                self.SP_CBasis.get(r, None), True) for r in
                                               self.data.Trips if (self.data.Scenarios[0],r) in self.data.Sce_Trip_unexplored])
                    else:
                        result = pool.starmap(self.SP_template_primal.create_and_solve_by_with,
                                              [(self.data.Scenarios[0], r, self.varValue.z, False, None, None, True) for r in
                                               self.data.Trips if (self.data.Scenarios[0],r) in self.data.Sce_Trip_unexplored])
                else:
                    """Revise the value of dual variables"""
                    # solve primal to get smaller dual var
                    result = pool.starmap(self.SP_template_primal.create_and_solve_by_with_revise_dual,
                                          [(self.data.Scenarios[0], r, self.varValue.z) for r in
                                           self.data.Trips if (self.data.Scenarios[0],r) in self.data.Sce_Trip_unexplored])

                pool.close()
                pool.join()
                time_record_solve_SP += time.time() - tmp

                tmp = time.time()
                result_dict = {list(item.keys())[0]: item[list(item.keys())[0]] for item in result}
                for r in self.data.Trips:
                    if (self.data.Scenarios[0], r) in self.data.Sce_Trip_unexplored:
                        item = result_dict[self.data.Scenarios[0], r]
                        if self.use_lp_basis:
                            VBasis_info = {
                                'pi1': [item['pi1'][k] for k in self.data.Node_of_trip[r]],
                                'pi2': [item['pi2'][h, l] for (h, l) in self.data.Hub_pairs],
                                'pi3': [item['pi3'][h, l] for (h, l) in self.data.Hub_pairs],
                                'pi4': [item['pi4'][i, j] for (i, j) in self.data.Node_pairs_for_mdl_sp[r]],
                                't': item['t']
                            }
                            self.VBasis_info[r] = VBasis_info.copy()  # VBasis for dual model
                            self.SP_VBasis[r] = item['VBasis']  # V and C Basis for primal model
                            self.SP_CBasis[r] = item['CBasis']

                        self.varValue.update_Sub_var_primal(self.data.Scenarios[0], r,
                                                                x=item['x'],
                                                                y=item['y'],
                                                                pi1=item['pi1'],
                                                                pi2=item['pi2'],
                                                                pi3=item['pi3'],
                                                                pi4=item['pi4'],
                                                                t=item['t'],
                                                                obj=item['obj'])#,
                                                                # VBasis=item['VBasis'],
                                                                # CBasis=item['CBasis'])
                time_retrieve_SP_solution += time.time() - tmp

                # then solve remaining scenarios
                tmp = time.time()
                pool = mp.Pool(processes=ALLOW_CORE)
                if not revise_dual_value:
                    if self.use_lp_basis:
                        # if use lp basis as warm start
                        result = pool.starmap(self.SP_template_primal.create_and_solve_by_with,
                                              [(s, r, self.varValue.z, True,
                                                self.SP_VBasis[r],
                                                self.SP_CBasis[r]) for s in self.data.Scenarios[1:] for r in
                                               self.data.Trips if (s,r) in self.data.Sce_Trip_unexplored])
                    else:
                        result = pool.starmap(self.SP_template_primal.create_and_solve_by_with,
                                              [(s, r, self.varValue.z) for s in self.data.Scenarios[1:] for r in
                                               self.data.Trips if (s,r) in self.data.Sce_Trip_unexplored])
                else:
                    """Revise the value of dual variables"""
                    result = pool.starmap(self.SP_template_primal.create_and_solve_by_with_revise_dual,
                                          [(s, r, self.varValue.z) for s in self.data.Scenarios[1:] for r in
                                           self.data.Trips if (s,r) in self.data.Sce_Trip_unexplored])
                pool.close()
                pool.join()
                time_record_solve_SP += time.time() - tmp


                tmp = time.time()
                result_dict = {list(item.keys())[0]: item[list(item.keys())[0]] for item in result}
                for r in self.data.Trips:
                    for s in self.data.Scenarios[1:]:
                        if (s, r) in self.data.Sce_Trip_unexplored:
                            item = result_dict[s, r]
                            self.varValue.update_Sub_var_primal(s, r,
                                                                x=item['x'],
                                                                y=item['y'],
                                                                pi1=item['pi1'],
                                                                pi2=item['pi2'],
                                                                pi3=item['pi3'],
                                                                pi4=item['pi4'],
                                                                t=item['t'],
                                                                obj=item['obj'])  # ,
                                                                # VBasis=item['VBasis'],
                                                                # CBasis=item['CBasis'])

                time_retrieve_SP_solution += time.time() - tmp

            else:
                # solve scenario[0] first
                tmp = time.time()
                pool = mp.Pool(processes=ALLOW_CORE)
                if self.use_lp_basis:
                    # if use lp basis as warm start
                    result = pool.starmap(self.SP_template_dual.create_and_solve_by_with,
                                          [(self.data.Scenarios[0], r, self.varValue.z, True, None, None,
                                            self.VBasis_info[r]) for r in self.data.Trips if (self.data.Scenarios[0],r) in self.data.Sce_Trip_unexplored])
                else:
                    result = pool.starmap(self.SP_template_dual.create_and_solve_by_with,
                                          [(self.data.Scenarios[0], r, self.varValue.z) for r in
                                           self.data.Trips if (self.data.Scenarios[0],r) in self.data.Sce_Trip_unexplored])
                pool.close()
                pool.join()
                time_record_solve_SP += time.time() - tmp

                tmp = time.time()
                result_dict = {list(item.keys())[0]: item[list(item.keys())[0]] for item in result}
                for r in self.data.Trips:
                    if (self.data.Scenarios[0], r) in self.data.Sce_Trip_unexplored:
                        item = result_dict[self.data.Scenarios[0], r]
                        self.varValue.update_Sub_var_primal(self.data.Scenarios[0], r,
                                                                x=item['x'],
                                                                y=item['y'],
                                                                pi1=item['pi1'],
                                                                pi2=item['pi2'],
                                                                pi3=item['pi3'],
                                                                pi4=item['pi4'],
                                                                t=item['t'],
                                                                obj=item['obj'])#,
                                                                # VBasis=None,
                                                                # CBasis=None)
                        if self.use_lp_basis:
                            self.SP_VBasis[r] = list(item['x'].values()) + list(item['y'].values())
                time_retrieve_SP_solution += time.time() - tmp

                # then solve remaining scenarios
                tmp = time.time()
                pool = ThreadPool(processes=ALLOW_CORE)
                if self.use_lp_basis:
                    # if use lp basis as warm start
                    result = pool.starmap(self.SP_template_primal.create_and_solve_by_with,
                                          [(s, r, self.varValue.z, True, self.SP_VBasis[r],
                                            None) for s in self.data.Scenarios[1:] for r in
                                           self.data.Trips if (s,r) in self.data.Sce_Trip_unexplored])
                else:
                    result = pool.starmap(self.SP_template_primal.create_and_solve_by_with,
                                          [(s, r, self.varValue.z) for s in self.data.Scenarios[1:] for r in
                                           self.data.Trips if (s,r) in self.data.Sce_Trip_unexplored])
                pool.close()
                pool.join()
                time_record_solve_SP += time.time() - tmp

                tmp = time.time()
                result_dict = {list(item.keys())[0]: item[list(item.keys())[0]] for item in result}
                for r in self.data.Trips:
                    for s in self.data.Scenarios[1:]:
                        if (s, r) in self.data.Sce_Trip_unexplored:
                            item = result_dict[s, r]
                            self.varValue.update_Sub_var_primal(s, r,
                                                                    x=item['x'],
                                                                    y=item['y'],
                                                                    pi1=item['pi1'],
                                                                    pi2=item['pi2'],
                                                                    pi3=item['pi3'],
                                                                    pi4=item['pi4'],
                                                                    t=item['t'],
                                                                    obj=item['obj'])#,
                                                                    # VBasis=item['VBasis'],
                                                                    # CBasis=item['CBasis'])
                time_retrieve_SP_solution += time.time() - tmp

            self.time_record_solve_SP.append(time_record_solve_SP)
            self.time_retrieve_SP_solution.append(time_retrieve_SP_solution)
            self.time_update_SP.append(0)  # no such steps

        elif self.parallel_method == 'process_1_core':
            time_record_solve_SP = 0
            time_retrieve_SP_solution = 0
            tmp = time.time()
            if (not use_dual_also) or (use_dual_also and n_iters == 1):
                # use only primal model
                # solve scenario 1 and get warm start info
                result = []
                s = self.data.Scenarios[0]
                if not revise_dual_value:
                    for r in self.data.Trips:
                        if (self.data.Scenarios[0], r) in self.data.Sce_Trip_unexplored:
                            res = self.SP_template_primal.create_and_solve_by_with(s, r, self.varValue.z,False, None, None, True)
                            result.append(res)
                else:
                    """Revise the value of dual variables"""
                    # solve primal to get smaller dual var
                    for r in self.data.Trips:
                        if (self.data.Scenarios[0], r) in self.data.Sce_Trip_unexplored:
                            res = self.SP_template_primal.create_and_solve_by_with_revise_dual(s, r, self.varValue.z,)
                            result.append(res)

                tmp = time.time()
                result_dict = {list(item.keys())[0]: item[list(item.keys())[0]] for item in result}
                for r in self.data.Trips:
                    if (self.data.Scenarios[0], r) in self.data.Sce_Trip_unexplored:
                        item = result_dict[self.data.Scenarios[0], r]
                        if self.use_lp_basis:
                            VBasis_info = {
                                'pi1': [item['pi1'][k] for k in self.data.Node_of_trip[r]],
                                'pi2': [item['pi2'][h, l] for (h, l) in self.data.Hub_pairs],
                                'pi3': [item['pi3'][h, l] for (h, l) in self.data.Hub_pairs],
                                'pi4': [item['pi4'][i, j] for (i, j) in self.data.Node_pairs_for_mdl_sp[r]],
                                't': item['t']
                            }
                            self.VBasis_info[r] = VBasis_info.copy()  # VBasis for dual model
                            self.SP_VBasis[r] = item['VBasis']  # V and C Basis for primal model
                            self.SP_CBasis[r] = item['CBasis']

                        self.varValue.update_Sub_var_primal(self.data.Scenarios[0], r,
                                                            x=item['x'],
                                                            y=item['y'],
                                                            pi1=item['pi1'],
                                                            pi2=item['pi2'],
                                                            pi3=item['pi3'],
                                                            pi4=item['pi4'],
                                                            t=item['t'],
                                                            obj=item['obj'])  # ,
                        time_record_solve_SP += item['sol_time']
                time_retrieve_SP_solution += time.time() - tmp

                # then solve remaining scenarios
                result = []
                tmp = time.time()
                if not revise_dual_value:
                    for s in self.data.Scenarios[1:]:
                        for r in self.data.Trips:
                            if (s,r) in self.data.Sce_Trip_unexplored:
                                res = self.SP_template_primal.create_and_solve_by_with(s, r, self.varValue.z)
                                result.append(res)

                else:
                    """Revise the value of dual variables"""
                    for s in self.data.Scenarios[1:]:
                        for r in self.data.Trips:
                            if (s,r) in self.data.Sce_Trip_unexplored:
                                res = self.SP_template_primal.create_and_solve_by_with_revise_dual(s, r, self.varValue.z)
                                result.append(res)



                tmp = time.time()
                result_dict = {list(item.keys())[0]: item[list(item.keys())[0]] for item in result}
                for r in self.data.Trips:
                    for s in self.data.Scenarios[1:]:
                        if (s, r) in self.data.Sce_Trip_unexplored:
                            item = result_dict[s, r]
                            self.varValue.update_Sub_var_primal(s, r,
                                                                x=item['x'],
                                                                y=item['y'],
                                                                pi1=item['pi1'],
                                                                pi2=item['pi2'],
                                                                pi3=item['pi3'],
                                                                pi4=item['pi4'],
                                                                t=item['t'],
                                                                obj=item['obj'])  # ,
                            # VBasis=item['VBasis'],
                            # CBasis=item['CBasis'])
                            time_record_solve_SP += item['sol_time']

                time_retrieve_SP_solution += time.time() - tmp

                self.time_record_solve_SP.append(time_record_solve_SP)
                self.time_retrieve_SP_solution.append(time_retrieve_SP_solution)
                self.time_update_SP.append(0)  # no such steps
            else:
                raise 'process_1_core, but use dual also. Not developed.'
        else:
            raise 'unknown parallel method in function solve_subproblems_primal()'

    def solve_subproblems_dual(self):
        # solve the sub problem
        time_update_SP = 0
        time_record_solve_SP = 0
        time_retrieve_SP_solution = 0
        if self.parallel_method == 'normal':
            for s in self.data.Scenarios:
                for r in self.data.Trips:
                    tmp = time.time()
                    self.dict_SP[s, r].update_objective_cons(param_z=self.varValue.z)
                    time_update_SP += time.time() - tmp

                    tmp = time.time()
                    if self.use_lp_basis:
                        self.dict_SP[s, r].write_lp_basis_info(True, self.SP_VBasis.get(r, None), self.SP_CBasis.get(r, None))
                    else:
                        self.dict_SP[s, r].write_lp_basis_info()
                    time_record_solve_SP += time.time() - tmp

                    tmp = time.time()
                    self.dict_SP[s,r].model.optimize()
                    time_record_solve_SP += time.time() - tmp

                    tmp = time.time()
                    result = self.dict_SP[s,r].obtain_solution_info(self.use_lp_basis)
                    time_retrieve_SP_solution += time.time() - tmp

                    tmp = time.time()
                    self.varValue.update_Sub_var(s=s, r=r, pi1=result['pi1'], pi2=result['pi2'], pi3=result['pi3'],
                                                 pi4=result['pi4'], t=result['t'], obj=result['obj'])  # update sub vars for the model under scenario r and trip t
                    time_retrieve_SP_solution += time.time() - tmp

            self.time_update_SP.append(time_update_SP)
            self.time_record_solve_SP.append(time_record_solve_SP)
            self.time_retrieve_SP_solution.append(time_retrieve_SP_solution)

        elif self.parallel_method == 'thread':
            time_update_SP = 0
            time_record_solve_SP = 0
            time_retrieve_SP_solution = 0
            # solve scenario 1
            # update
            tmp = time.time()
            pool = ThreadPool(processes=ALLOW_CORE)
            if self.use_lp_basis:
                pool.starmap(self.aux_solve_parallel_update,
                                      [(self.data.Scenarios[0], r, True, self.SP_VBasis.get(r, None),
                                        self.SP_CBasis.get(r, None), None) for r in self.data.Trips])
            else:
                pool.starmap(self.aux_solve_parallel_update,
                                      [(self.data.Scenarios[0], r) for r in self.data.Trips])
            pool.close()
            pool.join()
            time_update_SP += time.time() - tmp

            # solve
            tmp = time.time()
            pool = ThreadPool(processes=ALLOW_CORE)
            pool.starmap(self.aux_solve_parallel_optimize,
                         [(self.data.Scenarios[0], r) for r in self.data.Trips])
            time_record_solve_SP += time.time() - tmp

            # obtain info
            tmp = time.time()
            pool = ThreadPool(processes=ALLOW_CORE)
            result = pool.starmap(self.aux_solve_parallel_retrieval,
                         [(self.data.Scenarios[0], r, self.use_lp_basis) for r in self.data.Trips])
            time_retrieve_SP_solution += time.time() - tmp

            tmp = time.time()
            result_dict = {list(item.keys())[0]: item[list(item.keys())[0]] for item in result}
            for r in self.data.Trips:
                item = result_dict[self.data.Scenarios[0],r]
                self.varValue.update_Sub_var(s=self.data.Scenarios[0], r=r, pi1=item['pi1'], pi2=item['pi2'], pi3=item['pi3'],
                                             pi4=item['pi4'], t=item['t'], obj=item['obj'])
                if self.use_lp_basis:
                    self.SP_VBasis[r] = item['VBasis']
                    self.SP_CBasis[r] = item['CBasis']
            time_retrieve_SP_solution += time.time() - tmp

            # solve remaining scenarios
            # update
            tmp = time.time()
            pool = ThreadPool(processes=ALLOW_CORE)
            if self.use_lp_basis:
                pool.starmap(self.aux_solve_parallel_update,
                             [(s, r, True, self.SP_VBasis.get(r, None),
                               self.SP_CBasis.get(r, None), None) for s in self.data.Scenarios[1:] for r in self.data.Trips])
            else:
                pool.starmap(self.aux_solve_parallel_update,
                             [(s, r) for s in self.data.Scenarios[1:] for r in self.data.Trips])
            pool.close()
            pool.join()
            time_update_SP += time.time() - tmp

            # solve
            tmp = time.time()
            pool = ThreadPool(processes=ALLOW_CORE)
            pool.starmap(self.aux_solve_parallel_optimize,
                         [(s, r) for s in self.data.Scenarios[1:] for r in self.data.Trips])
            time_record_solve_SP += time.time() - tmp

            # obtain info
            tmp = time.time()
            pool = ThreadPool(processes=ALLOW_CORE)
            result = pool.starmap(self.aux_solve_parallel_retrieval,
                                  [(s, r, self.use_lp_basis) for s in self.data.Scenarios[1:] for r in self.data.Trips])
            time_retrieve_SP_solution += time.time() - tmp

            tmp = time.time()
            result_dict = {list(item.keys())[0]: item[list(item.keys())[0]] for item in result}
            for s in self.data.Scenarios[1:]:
                for r in self.data.Trips:
                    item = result_dict[s, r]
                    self.varValue.update_Sub_var(s=s, r=r, pi1=item['pi1'], pi2=item['pi2'], pi3=item['pi3'],
                                                 pi4=item['pi4'], t=item['t'], obj=item['obj'])
            time_retrieve_SP_solution += time.time() - tmp

            self.time_update_SP.append(time_update_SP)
            self.time_record_solve_SP.append(time_record_solve_SP)
            self.time_retrieve_SP_solution.append(time_retrieve_SP_solution)


        elif self.parallel_method == 'process':
            # solve scenario 1 and get warm start information
            time_record_solve_SP = 0
            time_retrieve_SP_solution = 0
            tmp = time.time()
            pool = mp.Pool(processes=ALLOW_CORE)
            with open(self.console_output_file, 'a') as file:
                print('\t\tstart to solve sp in parallel - the first scenario', file=file)
            if self.use_lp_basis:
                # if use lp basis as warm start
                result = pool.starmap(self.SP_template.create_and_solve_by_with,
                                      [(self.data.Scenarios[0], r, self.varValue.z, True, self.SP_VBasis.get(r, None),
                                        self.SP_CBasis.get(r, None)) for r in
                                       self.data.Trips if (self.data.Scenarios[0], r) in self.data.Sce_Trip_unexplored])
            else:
                result = pool.starmap(self.SP_template.create_and_solve_by_with,
                                      [(self.data.Scenarios[0], r, self.varValue.z) for r in
                                       self.data.Trips if (self.data.Scenarios[0], r) in self.data.Sce_Trip_unexplored])
            pool.close()
            pool.join()
            time_record_solve_SP += time.time() - tmp

            with open(self.console_output_file, 'a') as file:
                print('\t\tstart to arrange results from sp', file=file)
            tmp = time.time()
            result_dict = {list(item.keys())[0]: item[list(item.keys())[0]] for item in result}

            with open(self.console_output_file, 'a') as file:
                print('\t\trecord solutions', file=file)
            for r in self.data.Trips:
                if (self.data.Scenarios[0], r) in self.data.Sce_Trip_unexplored:
                    item = result_dict[self.data.Scenarios[0],r]
                    self.varValue.update_Sub_var(s=self.data.Scenarios[0], r=r, pi1=item['pi1'], pi2=item['pi2'], pi3=item['pi3'],
                                                 pi4=item['pi4'], t=item['t'], obj=item['obj'])
                    if self.use_lp_basis:
                        self.SP_VBasis[r] = item['VBasis']
                        self.SP_CBasis[r] = item['CBasis']
            time_retrieve_SP_solution += time.time() - tmp

            with open(self.console_output_file, 'a') as file:
                print('\t\tstart to solve sp in parallem - scenario >= 2', file=file)
            # then solve remaining scenarios
            tmp = time.time()
            pool = mp.Pool(processes=ALLOW_CORE)
            if self.use_lp_basis:
                # if use lp basis as warm start
                result = pool.starmap(self.SP_template.create_and_solve_by_with,
                                      [(s, r, self.varValue.z, True, self.SP_VBasis[r],
                                        self.SP_CBasis[r]) for s in self.data.Scenarios[1:] for r in
                                       self.data.Trips if (s,r) in self.data.Sce_Trip_unexplored])
            else:
                result = pool.starmap(self.SP_template.create_and_solve_by_with,
                                      [(s, r, self.varValue.z) for s in self.data.Scenarios[1:] for r in
                                       self.data.Trips if (s,r) in self.data.Sce_Trip_unexplored])
            pool.close()
            pool.join()
            time_record_solve_SP += time.time() - tmp

            with open(self.console_output_file, 'a') as file:
                print('\t\tarrange results - scenario>=2', file=file)
            tmp = time.time()
            result_dict = {list(item.keys())[0]: item[list(item.keys())[0]] for item in result}
            for s in self.data.Scenarios[1:]:
                for r in self.data.Trips:
                    if (s, r) in self.data.Sce_Trip_unexplored:
                        item = result_dict[s, r]
                        self.varValue.update_Sub_var(s=s, r=r, pi1=item['pi1'], pi2=item['pi2'], pi3=item['pi3'],
                                                     pi4=item['pi4'], t=item['t'], obj=item['obj'])

            time_retrieve_SP_solution += time.time() - tmp


            self.time_record_solve_SP.append(time_record_solve_SP)
            self.time_retrieve_SP_solution.append(time_retrieve_SP_solution)
            self.time_update_SP.append(0)  # no such steps



    def aux_solve_parallel_primal_update(self, s, r, use_lp_basis=False, VBasis=None, CBasis=None):
        self.dict_SP_primal[s, r].update_objective_cons(param_z=self.varValue.z)
        self.dict_SP_primal[s, r].write_lp_basis_info(use_lp_basis=use_lp_basis, VBasis=VBasis, CBasis=CBasis)

    def aux_solve_parallel_primal_retrieval(self, s, r, use_lp_basis=False):
        result = self.dict_SP_primal[s, r].obtain_solution_info(use_lp_basis=use_lp_basis)
        return {(s, r): result}

    def aux_solve_parallel_primal_optimize(self, s, r):
        self.dict_SP_primal[s, r].model.optimize()

    def aux_solve_parallel_primal_change_obj(self, s, r):
        self.dict_SP_primal[s, r].change_obj()




    def aux_solve_parallel_dual_update(self, s, r, use_lp_basis=False, VBasis=None, CBasis=None, VBasis_info:dict=None):
        self.dict_SP_dual[s, r].update_objective_cons(param_z=self.varValue.z)
        self.dict_SP_dual[s, r].write_lp_basis_info(use_lp_basis=use_lp_basis, VBasis=VBasis, CBasis=CBasis, VBasis_info=VBasis_info)

    def aux_solve_parallel_dual_optimize(self,s,r):
        self.dict_SP_dual[s, r].model.optimize()

    def aux_solve_parallel_dual_retrieval(self, s, r, use_lp_basis=False):
        result = self.dict_SP_dual[s, r].obtain_solution_info(use_lp_basis=use_lp_basis)
        return {(s, r): result}


    def aux_solve_parallel_update(self, s, r, use_lp_basis=False, VBasis=None, CBasis=None, VBasis_info:dict=None):
        self.dict_SP[s, r].update_objective_cons(param_z=self.varValue.z)
        self.dict_SP[s, r].write_lp_basis_info(use_lp_basis=use_lp_basis, VBasis=VBasis, CBasis=CBasis, VBasis_info=VBasis_info)

    def aux_solve_parallel_optimize(self,s,r):
        self.dict_SP[s, r].model.optimize()

    def aux_solve_parallel_retrieval(self, s, r, use_lp_basis=False):
        result = self.dict_SP[s, r].obtain_solution_info(use_lp_basis=use_lp_basis)
        return {(s, r): result}

    def assign_auxilary_var_lb(self, lb_dict):
        """
        add lb for each gamma(r,omega)
        :return:
        """
        for (s,r) in self.data.Sce_Trip_unexplored:
            self.MP.var.gamma[s,r].setAttr('lb', lb_dict.get((s,r), 0))

    def solve_uncompleted_MILP(self):
        """
        solve the MILP that includes only (s,r) fully explored by the tree-search algorithm
        :return:
        """
        t_milp = time.time()
        self.MP.update_master_timelimit_1_h()
        self.MP.model.optimize()
        t_milp = time.time() - t_milp
        lb = self.MP.model.ObjBound  # lower bound
        for (s,r) in self.data.Sce_Trip_unexplored:
            obj_s_r = self.cal_lb_unexplored_single_level_leader_obj(s,r)
            lb += obj_s_r * self.data.Scenarios_prob[s]*self.data.Trip_p_amount[s, r]

        z = {key: self.MP.var.z[key].X for key in self.MP.var.z if self.MP.var.z[key].X}
        ub = sum([self.data.Beta_hl[h,l]*z[h,l] for (h,l) in z])
        for s,r in self.data.Sce_Trip:
            ub += self.data.Scenarios_prob[s]*self.data.Trip_p_amount[s, r]*self.cal_lb_unexplored_bilevel_leader_obj(s, r, z)

        result = {
            'LB record': [lb],
            'UB record': [ub],
            'gap': (ub - lb)/ub,
            'MILPGap': self.MP.model.MIPGap,
            'pure solve time': t_milp
        }
        return result

    def cal_lb_unexplored_single_level_leader_obj(self, s, r):
        """
        solve the follower problem as the shortest path problem using Dijkstra algorithm.
        Consider only the leader obj, and the obj is an LB of the f in the leader obj
        :return:
        """
        G = nx.DiGraph()
        # asssume all bus legs are opened
        for (h, l) in self.data.Hub_pairs:
            G.add_edge(h, l, weight=self.data.tao_leader[h, l, s])
        for (i, j) in self.data.Node_pairs_for_mdl_sp[r]:
            G.add_edge(i, j, weight=self.data.gamma_leader[i, j, s])

        length = nx.dijkstra_path_length(G, source=self.data.Trip_origin[r], target=self.data.Trip_destination[r],
                                weight='weight')  # leader cost
        return length

    def cal_lb_unexplored_bilevel_leader_obj(self, s, r, opened_z):
        """
        solve the follower problem as a shortest path problem using Dijkstra algorithm.
        Consider it as a bilevel model.
        :return:
        """
        G = nx.DiGraph()
        # asssume all bus legs are opened
        for (h, l) in opened_z:
            G.add_edge(h, l, weight=self.data.tao_follower[h, l, s])
        for (i, j) in self.data.Node_pairs_for_mdl_sp[r]:
            G.add_edge(i, j, weight=self.data.gamma_follower[i, j, s])

        path = nx.dijkstra_path(G, source=self.data.Trip_origin[r], target=self.data.Trip_destination[r],
                                weight='weight')  # leader cost

        leader_cost = 0
        for idx in range(len(path) - 1):
            ori = path[idx]
            des = path[idx + 1]
            if ori in self.data.Hub and des in self.data.Hub:
                # use shuttle
                leader_cost += self.data.tao_leader[ori, des, s]
            else:
                leader_cost += self.data.gamma_leader[ori, des, s]

        return leader_cost

    def start_Benders_iteration(self, timelimit=3600, use_i_ccg=False):
        """
        Benders iterations
        :return:
        """

        parser = get_parser()
        self.args = parser.parse_args()
        MP_gap_scale = self.args.MP_gap_scale  # decrease MP_gap by a certain scale
        inexact_gap_threshold = self.args.inexact_gap_threshold  # criteria of exploration/exploitation
        stop_gap = self.args.stop_gap
        if not use_i_ccg:
            self.MP.set_params_additional_exact_ccg(ccg_time_limit=timelimit)  # set parameters to ensure MP is solved to optimal in each iteration

        # assign a lower bound for each gamma
        for s,r in self.data.Sce_Trip_unexplored:
            obj_min = self.cal_lb_unexplored_single_level_leader_obj(s, r)
            self.MP.var.gamma[s,r].setAttr('lb', obj_min)

        n_Benders_iter = 0
        ub = 1e20
        best_ub = 1e20
        lb = -1e20
        force_exploration = False  # if MIPGap=0, force to exploration
        force_exploitation = False  # if too much exploration executed, force to exploitation
        only_exploitation_always = False  # if no cuts are added, force the algo only to prove optimality
        n_explor_repeat = 0  # numer of continuously executed explorations
        n_exploit_repeat = 0  # numer of continuously executed exploitations
        start_time = time.time()
        # MP_solved_to_optimal = False
        while not ((best_ub - lb)/best_ub <= stop_gap):
            loop_start_time = time.time()
            with open(self.console_output_file, 'a') as file:
                print(
                    '\n--------Iteration: {}--------current lb={:.4f}, ub={:.4f}, best_ub={:.4f}, gap={:.4f}%, cal_time={:.2f}/{:.2f}/{}--------'.format(
                        n_Benders_iter, lb, ub, best_ub, (best_ub - lb) / best_ub*100, sum(self.time_record_solve_MP)+sum(self.time_record_solve_SP), time.time() - start_time,
                        timelimit), file=file)
            n_Benders_iter += 1

            if self.solve_MP_use_Benders:
                self.MP_Benders.update_varValue(varValue=self.varValue)  # input varValue info

            # if time.time() - start_time > timelimit:
            if len(self.time_record_solve_MP) > 0:
                if sum(self.time_record_solve_MP) + sum(self.time_record_solve_SP) > timelimit:
                    with open(self.console_output_file, 'a') as file:
                        print('time limit reached, terminate.', file=file)
                    break

            with open(self.console_output_file, 'a') as file:
                print('solve master...', file=file)
            tmp = time.time()
            with open(self.console_output_file, 'a') as file:
                if len(self.time_record_solve_MP) > 0:
                    optimize_time = sum(self.time_record_solve_MP) + sum(self.time_record_solve_SP)
                    remainint_time_MP = timelimit - optimize_time - self.time_record_solve_SP[-1]
                else:
                    optimize_time = time.time() - start_time
                    remainint_time_MP = timelimit - optimize_time
                print('update MP timelimit to: {}'.format(remainint_time_MP), file=file)
            self.MP.solve_model(use_i_ccg=use_i_ccg, remaining_time=remainint_time_MP, supposed_time_limit=self.MP.MP_timelimit)
            # MP_solved_to_optimal = self.MP.solved_to_optimal
            self.time_record_solve_MP.append(time.time() - tmp)
            # i_ub = self.MP.model.ObjBound
            # i_ub = sum(self.data.Beta_hl[h, l] * round(self.MP.var.z[h, l].X) for (h, l) in self.data.Hub_pairs) + sum(
            #     self.data.Scenarios_prob[s] * sum(
            #         self.data.Trip_p_amount[s,r] * self.MP.var.gamma[s, r].X for r in self.data.Trips) for s in
            #     self.data.Scenarios)  # inexact upper bound
            i_ub = self.MP.model.ObjBound  # inexact upper bound
            print('\t\t\t i_ub={:.4f}'.format(i_ub))
            if self.MP.model.MIPGap < 1e-6:
                i_ub = sum(self.data.Beta_hl[h, l] * round(self.MP.var.z[h, l].X) for (h, l) in self.data.Hub_pairs) + sum(
                    self.data.Scenarios_prob[s] * sum(
                        self.data.Trip_p_amount[s,r] * self.MP.var.gamma[s, r].X for r in self.data.Trips) for s in
                    self.data.Scenarios)  # inexact upper bound, if MP is solved to optimal, use the rounded z to avoid numerical mistakes
                print('\t\t\t MP gap < 1e-6, revise: i_ub={:.4f}'.format(i_ub))


            if self.MP.obj_lb_is_0:  # in exploitation
                if i_ub > lb:
                    lb = i_ub
            self.LB_record.append(lb)

            tmp = time.time()
            self.varValue.update_Master_var(var=self.MP.var)  # update z
            self.varValue.update_sp_obj_directly_from_mp(var=self.MP.var)  # update sub obj for s,r in self.data.explored
            self.time_retrieve_MP_solution.append(time.time() - tmp)

            with open(self.console_output_file, 'a') as file:
                print('solve sub...', file=file)
            if self.solution_appro == 'dual':
                self.solve_subproblems_dual()
            elif self.solution_appro == 'primal':
                self.solve_subproblems_primal(n_iters=n_Benders_iter, use_dual_also=self.primal_based_dual_to_ini, revise_dual_value=self.revise_dual_value)


            with open(self.console_output_file, 'a') as file:
                print('calculate ub...', file=file)
            tmp = time.time()
            ub = 0
            for s in self.data.Scenarios:
                for r in self.data.Trips:
                    # update upper bound
                    ub += self.data.Scenarios_prob[s] * self.data.Trip_p_amount[s,r] * self.varValue.latest_sub_obj[s, r]
            ub += sum(self.data.Beta_hl[h, l] * round(self.varValue.z[h, l]) for (h, l) in self.data.Hub_pairs)
            if ub < best_ub:
                best_ub = ub
                self.varValue.update_best_z()
            self.UB_record.append(best_ub)

            if use_i_ccg:
                with open(self.console_output_file, 'a') as file:
                    print('add cut in i-CCG criteria...', file=file)

                with open(self.console_output_file, 'a') as file:
                    print('***********inexact gap = ({:.3f}-{:.3f})/{:.3f}{:.4f}'.format(best_ub, i_ub, best_ub, (best_ub - i_ub) / best_ub), file=file)

                with open(self.console_output_file, 'a') as file:
                    print('\t\t\t force_exploration:{}'.format(force_exploration),
                          file=file)
                    print('\t\t\t force_exploitation:{}'.format(force_exploitation),
                          file=file)
                    print('\t\t\t only_exploitation_always:{}'.format(only_exploitation_always),
                          file=file)
                if ((best_ub - i_ub) / best_ub > inexact_gap_threshold or force_exploration) and not force_exploitation and not only_exploitation_always and (abs(ub-i_ub)/ub < 0.8 or self.MP.solved_to_optimal):
                    # if the MP gap is larger than 0.8, continue "exploitation" to optimize the MP
                    force_exploration = False  # if MP is already optimal, go to exploration in the next iter
                    n_explor_repeat += 1
                    with open(self.console_output_file, 'a') as file:
                        print('\t\t i-CCG: exploration', file=file)
                    # exploration
                    self.n_cut_in_iter = 0
                    self.n_combi_cut = 0
                    for s in self.data.Scenarios:
                        for r in self.data.Trips:
                            if not self.MP.agg_cut:
                                if self.varValue.gamma[s, r] - self.varValue.latest_sub_obj[s, r] >= -1e-5:
                                    # for explored subproblems, this if is always True, not cuts would be added
                                    continue  # when use disaggregated cut and the last cut is not violated, do not add cut again
                            self.n_cut_in_iter += 1
                            self.s_c_cut_stat[s,r] += 1
                            # with open(self.console_output_file, 'a') as file:
                            #     print('\t\t\t s={},r={}, deviation betwwen $zeta$ and sub_obj={}'.format(s, r,
                            #                 self.varValue.latest_sub_obj[s, r] - self.varValue.gamma[s, r]), file=file)
                            # (add cut only when necessary) generate c & c in the master problem
                            if self.solution_appro == 'dual':
                                self.MP.add_var_and_cons_ccg_dual(varValue=self.varValue, s=s, r=r)
                            elif self.solution_appro == 'primal':
                                self.MP.add_var_and_cons_ccg_primal(varValue=self.varValue, s=s, r=r)
                            else:
                                raise 'unknown param solution_appro in function start_Benders_iteration()'
                            if self.s_c_cut_stat[s,r] > 6:
                                # this (s,r) pair probably added cut with too large coefficient, add combinatorial cut for assistance
                                self.MP.add_combinatorial_cut(varValue=self.varValue, s=s, r=r)
                                self.n_cut_in_iter += 1
                                self.n_combi_cut += 1

                    with open(self.console_output_file, 'a') as file:
                        print('\t\t\t {} cuts are added, {} of them are combinatorial cuts'.format(self.n_cut_in_iter, self.n_combi_cut), file=file)
                    self.list_n_cut_in_iter.append(self.n_cut_in_iter)

                    if self.n_cut_in_iter == 0:
                        with open(self.console_output_file, 'a') as file:
                            print('\t\t\t no cuts are added, force to prove optimality', file=file)
                        only_exploitation_always = True
                    else:
                        if only_exploitation_always:
                            only_exploitation_always = False
                            with open(self.console_output_file, 'a') as file:
                                print('\t\t\t Switch only_exploitation_always to false', file=file)
                    # provide a lower bound for obj
                    self.MP.update_obj_lb(obj_lb=i_ub)
                    self.MP.obj_lb_is_0 = False
                    # let MIP_gap be larger, recover to the initial value
                    self.MP.update_MIP_gap(scale=1, to_original=True)
                    self.MP.MIP_gap = self.MP.original_MIP_gap

                    if n_explor_repeat >= 5:
                        with open(self.console_output_file, 'a') as file:
                            print('\t\t There are already 5 exploration, force to exploitation', file=file)
                        force_exploitation = True
                        n_explor_repeat = 0
                    n_exploit_repeat = 0  # set exploitation repeat counter to 0
                else:

                    with open(self.console_output_file, 'a') as file:
                        print('\t\t i-CCG: exploitation', file=file)
                    n_exploit_repeat += 1
                    # exploitation
                    if force_exploitation:
                        self.MP.update_MIP_gap(scale=MP_gap_scale * 0.1)  #forcuse more on the MP search,
                    else:
                        self.MP.update_MIP_gap(scale=MP_gap_scale)
                    if only_exploitation_always:
                        self.MP.assign_MIP_gap_constant(gap=1e-5)  # force to prove optimality
                    # self.MP.update_MP_timelimit(dt=MP_timelimit_dt)
                    # delete the inexact lower bound on obj
                    self.MP.update_obj_lb(obj_lb=lb)
                    self.MP.obj_lb_is_0 = True
                    if self.MP.solved_to_optimal:
                        with open(self.console_output_file, 'a') as file:
                            print('\t\t MP is solved to optimal, force next iter to exploration', file=file)
                        force_exploration = True
                        if only_exploitation_always:
                            only_exploitation_always = False
                            with open(self.console_output_file, 'a') as file:
                                print('\t\t\t Switch only_exploitation_always to false at the begining of the branch',
                                      file=file)

                    force_exploitation = False
                    n_explor_repeat = 0
                    # add a condition to force exploration
                    if n_exploit_repeat >= 5:
                        with open(self.console_output_file, 'a') as file:
                            print('\t\t There are already 5 exploitations, force to exploration', file=file)
                        force_exploration = True
                        n_exploit_repeat = 0

            else:
                with open(self.console_output_file, 'a') as file:
                    print('add cut in exact CCG criteria...', file=file)
                self.n_cut_in_iter = 0
                # add cuts
                for s in self.data.Scenarios:
                    for r in self.data.Trips:
                        # print('s={}, r={}, gamma={:.4f}, obj={:.4f}'.format(s,r,self.varValue.gamma[s, r],self.varValue.latest_sub_obj[s, r]))
                        if not self.MP.agg_cut:
                            if self.varValue.gamma[s, r] >= self.varValue.latest_sub_obj[s, r]:
                                continue  # when use disaggregated cut and the last cut is not violated, do not add cut again

                        # print('****add a cut')
                        # (add cut only when necessary) generate c & c in the master problem
                        if self.solution_appro == 'dual':
                            self.MP.add_var_and_cons_ccg_dual(varValue=self.varValue, s=s, r=r)
                        elif self.solution_appro == 'primal':
                            self.MP.add_var_and_cons_ccg_primal(varValue=self.varValue, s=s, r=r)
                        else:
                            raise 'unknown param solution_appro in function start_Benders_iteration()'
                        self.n_cut_in_iter += 1
                self.list_n_cut_in_iter.append(self.n_cut_in_iter)

            self.MP.update_var_index()  # update the index of master vars
            self.time_update_MP.append(time.time() - tmp)

            # record time consumption
            self.time_record.append(time.time() - loop_start_time)

        # TODO 20241107: This calculation seems not necessary, delete it.
        # if self.solution_appro == 'primal':
        #     self.solve_subproblems_primal(n_iters=n_Benders_iter, use_dual_also=self.primal_based_dual_to_ini, need_x=True)
        self.update_cal_infor(n_Benders_iter=n_Benders_iter)


    def start_Benders_iteration_classic(self, timelimit=3600, use_i_ccg=False):
        """
        Benders iterations
        :return:
        """

        parser = get_parser()
        self.args = parser.parse_args()
        stop_gap = self.args.stop_gap
        self.MP.set_params_additional_exact_ccg(ccg_time_limit=timelimit)  # set parameters to ensure MP is solved to optimal in each iteration

        # assign a lower bound for each gamma
        for s,r in self.data.Sce_Trip_unexplored:
            obj_min = self.cal_lb_unexplored_single_level_leader_obj(s, r)
            self.MP.var.gamma[s,r].setAttr('lb', obj_min)

        n_Benders_iter = 0
        ub = 1e20
        best_ub = 1e20
        lb = -1e20

        start_time = time.time()
        while not ((best_ub - lb)/best_ub <= stop_gap):
            loop_start_time = time.time()
            with open(self.console_output_file, 'a') as file:
                print('\n--------Iteration: {}--------current lb={:.4f}, ub={:.4f}, best_ub={:.4f}, gap={:.4f}%, cal_time={:.2f}/{:.2f}/{}--------'.format(
                        n_Benders_iter, lb, ub, best_ub, (best_ub - lb) / best_ub*100, sum(self.time_record_solve_MP)+sum(self.time_record_solve_SP),time.time() - start_time,
                        timelimit), file=file)
            n_Benders_iter += 1

            if len(self.time_record_solve_MP) > 0:
                if sum(self.time_record_solve_MP) + sum(self.time_record_solve_SP) > timelimit:
                    with open(self.console_output_file, 'a') as file:
                        print('time limit reached, terminate.', file=file)
                    break

            with open(self.console_output_file, 'a') as file:
                print('solve master...', file=file)

            tmp = time.time()
            with open(self.console_output_file, 'a') as file:
                if len(self.time_record_solve_MP) > 0:
                    optimize_time = sum(self.time_record_solve_MP) + sum(self.time_record_solve_SP)
                    remainint_time_MP = timelimit - optimize_time - self.time_record_solve_SP[-1]
                else:
                    optimize_time = time.time() - start_time
                    remainint_time_MP = timelimit - optimize_time
                print('update MP timelimit to: {}'.format(remainint_time_MP), file=file)
            self.MP.solve_model(use_i_ccg=use_i_ccg, remaining_time=remainint_time_MP, supposed_time_limit=self.MP.MP_timelimit)
            self.time_record_solve_MP.append(time.time() - tmp)

            lb = self.MP.model.ObjBound  # inexact upper bound
            if self.MP.model.MIPGap < 1e-6:
                lb = sum(self.data.Beta_hl[h, l] * round(self.MP.var.z[h, l].X) for (h, l) in self.data.Hub_pairs) + sum(
                    self.data.Scenarios_prob[s] * sum(
                        self.data.Trip_p_amount[s,r] * self.MP.var.gamma[s, r].X for r in self.data.Trips) for s in
                    self.data.Scenarios)  # inexact upper bound, if MP is solved to optimality, use the rounded z to avoid numerical mistakes
            self.LB_record.append(lb)

            tmp = time.time()
            self.varValue.update_Master_var(var=self.MP.var)  # update z
            self.varValue.update_sp_obj_directly_from_mp(var=self.MP.var)  # update sub obj for s,r in self.data.explored
            self.time_retrieve_MP_solution.append(time.time() - tmp)

            with open(self.console_output_file, 'a') as file:
                print('solve sub...', file=file)
            self.solve_subproblems_primal(n_iters=n_Benders_iter, use_dual_also=self.primal_based_dual_to_ini, revise_dual_value=self.revise_dual_value)

            with open(self.console_output_file, 'a') as file:
                print('calculate ub...', file=file)
            tmp = time.time()
            ub = 0
            for s in self.data.Scenarios:
                for r in self.data.Trips:
                    # update upper bound
                    ub += self.data.Scenarios_prob[s] * self.data.Trip_p_amount[s,r] * self.varValue.latest_sub_obj[s, r]
            ub += sum(self.data.Beta_hl[h, l] * round(self.varValue.z[h, l]) for (h, l) in self.data.Hub_pairs)
            if ub < best_ub:
                best_ub = ub
                self.varValue.update_best_z()
            self.UB_record.append(best_ub)

            self.n_cut_in_iter = 0
            # add cuts
            for s in self.data.Scenarios:
                for r in self.data.Trips:
                    if self.varValue.gamma[s, r] >= self.varValue.latest_sub_obj[s, r]:
                        continue  # when use disaggregated cut and the last cut is not violated, do not add cut again

                    # print('****add a cut')
                    # (add cut only when necessary) generate c & c in the master problem
                    self.MP.add_var_and_cons_ccg_primal(varValue=self.varValue, s=s, r=r)
                    self.n_cut_in_iter += 1
            self.list_n_cut_in_iter.append(self.n_cut_in_iter)

            tmp = time.time()
            self.MP.update_var_index()  # update the index of master vars
            self.time_update_MP.append(time.time() - tmp)

            # record time consumption
            self.time_record.append(time.time() - loop_start_time)

        self.update_cal_infor(n_Benders_iter=n_Benders_iter)

    def bd_callback(self, model, where):
        if where == GRB.Callback.MIPSOL:  # or (where==GRB.Callback.MIPNODE and model.cbGet(GRB.Callback.MIPNODE_STATUS)==GRB.OPTIMAL):
            tmp = time.time()
            with open(self.console_output_file, 'a') as file:
                print('\n--------Iteration: {}--------current lb={:.4f}, ub={:.4f}, best_ub={:.4f}, gap={:.4f}%, cal_time={:.2f}/{:.2f}/{}--------'.format(
                        model._n_Benders_iter, model._lb, model._ub, model._best_ub, (model._best_ub - model._lb) / model._best_ub*100, model.cbGet(GRB.Callback.RUNTIME)-model._i_o_time, time.time()-self.time_start,
                        model._timelimit), file=file)
            model._n_Benders_iter += 1

            if where == GRB.Callback.MIPSOL:
                z_value = {(h,l): round(model.cbGetSolution(self.MP.var.z[h,l])) for h,l in self.data.Hub_pairs}
                gamma_value = {(s, r): model.cbGetSolution(self.MP.var.gamma[s, r]) for (s, r) in self.data.Sce_Trip}
                model._lb = model.cbGet(GRB.Callback.MIPSOL_OBJBND)  # TODO revise 20260216 in MIPNODE
                mip_node = model.cbGet(GRB.Callback.MIPSOL_NODCNT)  # Current explored node count.
            else:
                z_value = {(h,l): model.cbGetNodeRel(self.MP.var.z[h,l]) for h,l in self.data.Hub_pairs}
                gamma_value = {(s, r): model.cbGetNodeRel(self.MP.var.gamma[s, r]) for (s, r) in self.data.Sce_Trip}
                model._lb = model.cbGet(GRB.Callback.MIPNODE_OBJBND)  # TODO revise 20260216 in MIPNODE
                mip_node = None  # Current explored node count.


            self.LB_record.append(model._lb)

            self.varValue.update_Master_var_callback(z=z_value, gamma=gamma_value, n_iters=self.MP.var.n_iters)
            self.varValue.update_sp_obj_directly_from_mp_callback(gamma=gamma_value)

            # solve subproblems
            with open(self.console_output_file, 'a') as file:
                print('\t\t solve subproblems', file=file)
            self.solve_subproblems_primal(n_iters=model._n_Benders_iter, use_dual_also=self.primal_based_dual_to_ini,
                                    revise_dual_value=self.revise_dual_value)  # update self.varValue.latest_sub_obj

            # update UB
            with open(self.console_output_file, 'a') as file:
                print('\t\t update UB', file=file)
            ub = 0
            for s in self.data.Scenarios:
                for r in self.data.Trips:
                    # update upper bound
                    ub += self.data.Scenarios_prob[s] * self.data.Trip_p_amount[s,r] * self.varValue.latest_sub_obj[s, r]
            ub += sum(self.data.Beta_hl[h, l] * round(self.varValue.z[h, l]) for (h, l) in self.data.Hub_pairs)
            model._ub = ub
            if ub < model._best_ub:
                model._best_ub = ub
                self.varValue.update_best_z()
            self.UB_record.append(model._best_ub)

            # add cuts
            with open(self.console_output_file, 'a') as file:
                print('\t\t Add cuts', file=file)
            self.n_cut_in_iter = 0
            for s in self.data.Scenarios:
                for r in self.data.Trips:
                    if self.varValue.gamma[s, r] >= self.varValue.latest_sub_obj[s, r]:
                        continue  # when use disaggregated cut and the last cut is not violated, do not add cut again
                    model.cbLazy(
                        self.MP.var.gamma[s, r] >= -self.varValue.pi1[s, r, self.data.Trip_origin[r]] + self.varValue.pi1[s, r,
                        self.data.Trip_destination[r]] - gp.quicksum(
                            self.MP.var.z[h, l] * self.varValue.pi2[s, r, h, l] for (h, l) in
                            self.data.Hub_pairs) - gp.quicksum(
                            self.varValue.pi3[s, r, h, l] for (h, l) in self.data.Hub_pairs) - gp.quicksum(
                            self.varValue.pi4[s, r, i, j] for (i, j) in self.data.Node_pairs_for_mdl_sp[r]) + self.varValue.t[
                            s, r] * (
                                self.MP.var.theta1[s, r, self.data.Trip_origin[r]] -
                                self.MP.var.theta1[s, r, self.data.Trip_destination[r]] + gp.quicksum(
                            self.MP.var.delta[s, r, h, l] for (h, l) in self.data.Hub_pairs) + gp.quicksum(
                            self.MP.var.theta3[s, r, h, l] for (h, l) in self.data.Hub_pairs) + gp.quicksum(
                            self.MP.var.theta4[s, r, i, j] for (i, j) in self.data.Node_pairs_for_mdl_sp[r]))
                        )
                    self.n_cut_in_iter += 1
            self.list_n_cut_in_iter.append(self.n_cut_in_iter)

            with open(self.console_output_file, 'a') as file:
                print('\t\t update calculation info', file=file)
            self.MP.update_var_index()  # update the index of master vars
            self.update_cal_infor(n_Benders_iter=model._n_Benders_iter, callback=True, node_count=mip_node)
            model._i_o_time += time.time() - tmp - self.time_record_solve_SP[-1]  # substract the time for solving SPs
            run_time = model.cbGet(GRB.Callback.RUNTIME)  # gurobi model run time
            if run_time - model._i_o_time > model._timelimit:
                model.terminate()
            if (self.UB_record[-1] - self.LB_record[-1]) / self.UB_record[-1] < 1e-5:
                with open(self.console_output_file, 'a') as file:
                    print('\t\t Optimality gap satisfies the criterion. Model erminates inside the callback.', file=file)
                model.terminate()

    def start_Benders_iteration_callback(self, timelimit=3600, use_i_ccg=False):
        """
        Benders iterations using callback
        :return:
        """
        parser = get_parser()
        self.args = parser.parse_args()
        stop_gap = self.args.stop_gap
        self.MP.set_params_additional_exact_ccg(ccg_time_limit=timelimit)  # set parameters to ensure MP is solved to optimal in each iteration

        # assign a lower bound for each gamma
        for s,r in self.data.Sce_Trip_unexplored:
            obj_min = self.cal_lb_unexplored_single_level_leader_obj(s, r)
            self.MP.var.gamma[s,r].setAttr('lb', obj_min)


        self.MP.add_theta_vars_callback()  # add theta varsa and associated constrs. They will be used to add C&CG cuts in the callback function.
        self.MP.model._n_Benders_iter = 0
        self.MP.model._start_time = time.time()
        self.MP.model._timelimit = timelimit
        self.MP.model._i_o_time = 0
        self.MP.model._lb = -1e20
        self.MP.model._ub = 1e20
        self.MP.model._best_ub = 1e20
        self.MP.model.setParam('TimeLimit', 3*timelimit)  # 3 times of timelimit. Terminate the model until optimization time exceeds timelimit.
        self.MP.model.setParam('LazyConstraints', 1)  # 3 times of timelimit. Terminate the model until optimization time exceeds timelimit.

        self.MP.model.optimize(self.bd_callback)


        # with open(self.console_output_file, 'a') as file:
        #     print('\t\t ub={:.2f}, lb={:.2f}'.format(ub, self.LB_record[-1]), file=file)
        if (self.UB_record[-1] - self.LB_record[-1]) / self.UB_record[-1] > 1e-5 and self.MP.model.SolCount > 0:  # when self.MP.model.SolCount==0, do not run this function sinse varValues or Bound may not be available.
            with open(self.console_output_file, 'a') as file:
                print('\t\t Model terminated but the solution is not optimal.', file=file)

            z_value = {(h, l): round(self.MP.var.z[h, l].X) for h, l in self.data.Hub_pairs}
            model_bound = self.MP.model.ObjBound
            # if the last UB or LB is not recorded, update.
            if self.LB_record[-1] != model_bound:
                self.LB_record.append(self.MP.model.ObjBound)
            gamma_value = {(s, r): self.MP.var.gamma[s, r].X for (s, r) in self.data.Sce_Trip}
            self.varValue.update_Master_var_callback(z=z_value, gamma=gamma_value, n_iters=self.MP.var.n_iters)
            self.varValue.update_sp_obj_directly_from_mp_callback(gamma=gamma_value)

            # solve subproblems
            self.solve_subproblems_primal(n_iters=self.MP.model._n_Benders_iter,
                                          use_dual_also=self.primal_based_dual_to_ini,
                                          revise_dual_value=self.revise_dual_value)
            # update UB again using new self.varValue.latest_sub_obj
            ub = 0
            for s in self.data.Scenarios:
                for r in self.data.Trips:
                    # update upper bound
                    ub += self.data.Scenarios_prob[s] * self.data.Trip_p_amount[s, r] * \
                          self.varValue.latest_sub_obj[
                              s, r]
            ub += sum(self.data.Beta_hl[h, l] * round(self.varValue.z[h, l]) for (h, l) in self.data.Hub_pairs)

            if ub < self.MP.model._best_ub:
                self.MP.model._best_ub = ub
                self.varValue.update_best_z()
            self.UB_record.append(self.MP.model._best_ub)

            self.update_cal_infor(n_Benders_iter=self.MP.model._n_Benders_iter, callback=True,
                                  node_count=self.record.get('n_node_BnB', None))



    def update_cal_infor(self, n_Benders_iter, callback=False, node_count=None):
        self.record = {'time model creation':self.time_mdl_creation,
                'time cost': self.time_record,
                'total time cost': sum(self.time_record),
                'time cost MP': self.time_record_solve_MP,
                'total time cost MP': sum(self.time_record_solve_MP),
                'time cost SP': self.time_record_solve_SP,
                'total time cost SP': sum(self.time_record_solve_SP),
                'time retrieve MP info': self.time_retrieve_MP_solution,
                'total time retrieve MP info': sum(self.time_retrieve_MP_solution),
                'time retrieve SP info': self.time_retrieve_SP_solution,
                'total time retrieve SP info': sum(self.time_retrieve_SP_solution),
                'time update MP': self.time_update_MP,
                'total time update MP': sum(self.time_update_MP),
                'time update SP': self.time_update_SP,
                'total time update SP': sum(self.time_update_SP),
                'LB record': self.LB_record,
                'UB record': self.UB_record,
                'gap': (min(self.UB_record) - self.LB_record[-1]) / (min(self.UB_record)+1e-8) if len(self.UB_record) else None,
                'pure solve time': sum(self.time_record_solve_MP)+sum(self.time_record_solve_SP) if not callback else self.MP.model.Runtime - self.MP.model._i_o_time,
                'n iters': n_Benders_iter,
                'list_n_cut_in_iter': self.list_n_cut_in_iter,
                'n_node_BnB': self.MP.model.NodeCount if len(self.UB_record) and not callback else node_count,  # when node_count is received, use it; otherwise, use the default value: None.
                'unexplored_sub_prob': len(self.unexplred_probles),
                'tree_search_time': self.search_time_rec,
                'tree_search_iter': self.search_iter_rec,
                       }

    def update_result_ccg(self, result, read_data_time, tag=None):
        """
        collect results
        :param result: the result dictionary
        :return:
        """
        record = self.record
        result['file_name'].append(os.path.basename(self.data.file_name)[:-4])
        result['node-hub-trip-passenger'].append(re.findall(r'\d+', self.data.file_name)[:4])
        result['tag'].append(tag)
        result['agg_cut'].append(self.agg_cut)
        result['arc elim'].append(self.data.arc_elimination)
        result['Delta type'].append(self.data.Delta_type)
        result['parallel method'].append(self.parallel_method)
        result['lp warm start'].append(self.use_lp_basis)
        result['solution approach'].append(self.solution_appro)
        result['primal also dual'].append(self.primal_based_dual_to_ini)
        result['time read data'].append(read_data_time)

        result['time model creation'].append(record.get('time model creation', None))
        result['solution time'].append(record.get('time cost', None))
        result['total solution time'].append(record.get('total time cost', None))
        result['solution time MP'].append(record.get('time cost MP', None))
        result['total solution time MP'].append(record.get('total time cost MP', None))
        result['solution time SP'].append(record.get('time cost SP', None))
        result['total solution time SP'].append(record.get('total time cost SP', None))

        result['time retrieve MP info'].append(record.get('time retrieve MP info', None))
        result['total time retrieve MP info'].append(record.get('total time retrieve MP info', None))
        result['time retrieve SP info'].append(record.get('time retrieve SP info', None))
        result['total time retrieve SP info'].append(record.get('total time retrieve SP info', None))
        result['time update MP'].append(record.get('time update MP', None))
        result['total time update MP'].append(record.get('total time update MP', None))
        result['time update SP'].append(record.get('time update SP', None))
        result['total time update SP'].append(record.get('total time update SP', None))

        result['LB record'].append(record.get('LB record', None))
        result['UB record'].append(record.get('UB record', None))
        if len(record.get('LB record',[])) == 0:
            result['final_lb'].append(None)
            result['final_ub'].append(None)
        else:
            result['final_lb'].append(max(record.get('LB record', [None])))
            result['final_ub'].append(min(record.get('UB record', [None])))
        result['MILPGap'].append(record.get('MILPGap', None))
        result['gap'].append(record.get('gap', None))
        result['pure solve time'].append(record.get('pure solve time', None))
        result['n iters'].append(record.get('n iters', None))
        result['n_cut_each_iter'].append(record.get('list_n_cut_in_iter', None))
        result['n_node_BnB'].append(record.get('n_node_BnB', 0))
        result['unexplored_sub_prob'].append(record.get('unexplored_sub_prob', 0))

        if 'time preprocess' in result.keys():
            result['time preprocess'].append(self.time_preprocess_tree_search)
            search_time = record.get('tree_search_time', {'a': None})
            search_time = list(search_time.values())
            result['min time preprocess'].append(min(search_time))
            result['max time preprocess'].append(max(search_time))

            search_iter = record.get('tree_search_iter', {'a': None})
            search_iter = list(search_iter.values())
            result['min iter preprocess'].append(min(search_iter))
            result['max iter preprocess'].append(max(search_iter))
            result['ave iter preprocess'].append(np.mean(search_iter))



        return result

    def plot_result(self):
        plt.plot(self.LB_record, 'b-.')
        plt.plot(self.UB_record, 'r-*')
        plt.show()

    def reassure_routines(self):
        """
        re-calculate follower decision for all s,r, then update self.varValue
        :return:
        """
        routine_finder = RoutineFinder(data=self.data, param_z=self.varValue.best_z)
        pool = mp.Pool(processes=ALLOW_CORE)
        result = pool.starmap(routine_finder.solve_model,
                              [(s, r) for (s,r) in self.data.Sce_Trip])
        pool.close()
        pool.join()
        for idx in range(len(self.data.Sce_Trip)):
            s,r = self.data.Sce_Trip[idx]
            item = result[idx]
            self.varValue.update_x_y(s, r, item['x_sol'], item['y_sol'])

        self.varValue.update_best_x_y()


    def collect_solution_info(self, solution_type, model_type='normal', real_tao=None, real_gamma=None):
        """
        collect the solution information
        @:param solution_type: 'determ', 'stochastic', 'stochastic_fixed'
        :return: the information of leader and follower decisions
        """
        self.reassure_routines()
        """Collect Trip Information"""
        trip_info = {'trip name': [],
                     'scenario': [],
                     'solution type': [],
                     'origination': [],
                     'destination': [],
                     'amounts': [],
                     'income_level': [],
                     'scenario_prob': [],
                     'stops': [],
                     'hubs': [],
                     'follower cost': [],
                     'leader cost': [],
                     'gamma': [],
                     'follower travel time': [],
                     'follower travel distance': [],
                     'cost per mile': [],
                     }

        trip_info['trip name'] = [r for r in self.data.Trips for s in self.data.Scenarios]
        trip_info['solution type'] = [solution_type for r in self.data.Trips for s in self.data.Scenarios]
        trip_info['origination'] = [self.data.Trip_origin[r] for r in self.data.Trips for s in self.data.Scenarios]
        trip_info['destination'] = [self.data.Trip_destination[r] for r in self.data.Trips for s in self.data.Scenarios]
        trip_info['amounts'] = [self.data.Trip_p_amount[s,r] for r in self.data.Trips for s in self.data.Scenarios]
        trip_info['income_level'] = [self.data.node_income_level[self.data.Trip_destination[r]] for r in self.data.Trips
                                     for s in self.data.Scenarios]
        trip_info['scenario'] = [s for r in self.data.Trips for s in self.data.Scenarios]
        trip_info['scenario_prob'] = [self.data.Scenarios_prob[s] for r in self.data.Trips for s in self.data.Scenarios]

        trip_info['stops'] = [
            [[i, j] for (i, j) in self.data.Node_pairs_for_mdl_sp[r] if self.varValue.y.get((s, r, i, j), 0) > 0] for r in
            self.data.Trips for s in self.data.Scenarios]
        trip_info['hubs'] = [[[h, l] for (h, l) in self.data.Hub_pairs if self.varValue.x.get((s,r,h,l),0)>0] for r in
                             self.data.Trips for s in self.data.Scenarios]
        trip_info['follower cost'] = [self.data.Trip_p_amount[s,r] * np.sum([
            self.data.tao_follower[h, l, s] * self.varValue.x.get((s, r, h, l),0) for (h, l) in self.data.Hub_pairs]) +
                                    self.data.Trip_p_amount[s,r] * np.sum([
            self.data.gamma_follower[i, j, s] * self.varValue.y.get((s, r, i, j),0) for (i, j) in self.data.Node_pairs_for_mdl_sp[r]]) for r in
                                    self.data.Trips for s in self.data.Scenarios]

        trip_info['leader cost'] = [self.data.Trip_p_amount[s, r] * np.sum([
            self.data.tao_leader[h, l, s] * self.varValue.x.get((s, r, h, l), 0) for (h, l) in self.data.Hub_pairs]) +
                                    self.data.Trip_p_amount[s, r] * np.sum([
            self.data.gamma_leader[i, j, s] * self.varValue.y.get((s, r, i, j), 0) for (i, j) in
            self.data.Node_pairs_for_mdl_sp[r]]) for r in
                                    self.data.Trips for s in self.data.Scenarios]

        trip_info['follower travel time'] = [np.sum(
            [self.data.Travel_time[i, j, s] * self.varValue.y.get((s, r, i, j), 0) for (i, j) in
             self.data.Node_pairs_for_mdl_sp[r]]) + np.sum(
                [(self.data.Travel_time[h, l, s] + self.data.S_wait_bus) * self.varValue.x.get((s, r, h, l), 0) for
                 (h, l) in self.data.Hub_pairs]) for r in self.data.Trips for s in self.data.Scenarios]

        trip_info['follower travel distance'] = [np.sum(
            [self.data.Dist[i, j] * self.varValue.y.get((s, r, i, j), 0) for (i, j) in
             self.data.Node_pairs_for_mdl_sp[r]]) + np.sum(
            [self.data.Dist[h, l] * self.varValue.x.get((s, r, h, l), 0) for
             (h, l) in self.data.Hub_pairs]) for r in self.data.Trips for s in self.data.Scenarios]

        trip_info['cost per mile'] = [trip_info['follower cost'][i] / trip_info['follower travel distance'][i] for i in
                                      range(len(trip_info['follower cost']))]


        trip_info['gamma'] = [self.varValue.gamma[s,r] for r in self.data.Trips for s in self.data.Scenarios]
        df_trip_info = pd.DataFrame.from_dict(trip_info, orient='columns')

        """Collect hub Information"""
        hub_info = {'hub leg': [],
                    'solution type': [],
                    'opening cost': [],
                    'used times_low': [],
                    'per1': [],
                    'used times_middle': [],
                    'per2': [],
                    'used times_high': [],
                    'per3': [],
                    'used times_total': [],
                    'per4': [],
                    'served customers_low': [],
                    'per5': [],
                    'served customers_middle': [],
                    'per6': [],
                    'served customers_high': [],
                    'per7': [],
                    'served customers_total': [],
                    'per8': [],
                     }
        n_high_trips = np.sum([1 for r in self.data.Trips if self.data.node_income_level[self.data.Trip_destination[r]]=='High'])
        n_middle_trips = np.sum([1 for r in self.data.Trips if self.data.node_income_level[self.data.Trip_destination[r]]=='Middle'])
        n_low_trips = np.sum([1 for r in self.data.Trips if self.data.node_income_level[self.data.Trip_destination[r]]=='Low'])

        n_high_trips_amount = np.sum([self.data.Trip_p_amount[s,r]*self.data.Scenarios_prob[s] for r in self.data.Trips for s in self.data.Scenarios if self.data.node_income_level[self.data.Trip_destination[r]]=='High'])
        n_middle_trips_amount = np.sum([self.data.Trip_p_amount[s,r]*self.data.Scenarios_prob[s] for r in self.data.Trips for s in self.data.Scenarios if self.data.node_income_level[self.data.Trip_destination[r]]=='Middle'])
        n_low_trips_amount = np.sum([self.data.Trip_p_amount[s,r]*self.data.Scenarios_prob[s] for r in self.data.Trips for s in self.data.Scenarios if self.data.node_income_level[self.data.Trip_destination[r]]=='Low'])

        opened_hl_pairs = [(h,l) for (h,l) in self.data.Hub_pairs if self.varValue.z[h, l] > 0.1]
        hub_info['hub leg'] = [[h, l] for (h, l) in opened_hl_pairs]
        hub_info['solution type'] = [solution_type for (h, l) in opened_hl_pairs]
        hub_info['opening cost'] = [self.data.Beta_hl[h, l] for (h, l) in opened_hl_pairs]
        hub_info['used times_total'] = [
            np.sum([1 for r in self.data.Trips for s in self.data.Scenarios if self.varValue.x.get((s, r, h, l),0) > 0.1]) for
            (h, l) in opened_hl_pairs]  # how many times is the hub pari used
        hub_info['per4'] = [i/(n_low_trips+n_middle_trips+n_high_trips) for i in hub_info['used times_total']]
        hub_info['used times_low'] = [np.sum(
                [1 for r in self.data.Trips for s in self.data.Scenarios if self.varValue.x.get((s, r, h, l), 0) > 0.1
                 if self.data.node_income_level[self.data.Trip_destination[r]] == 'Low'])
            for (h, l) in opened_hl_pairs]  # how many times is the hub pari used
        hub_info['per1'] = [i/n_low_trips for i in hub_info['used times_low']]
        hub_info['used times_high'] = [np.sum(
            [1 for r in self.data.Trips for s in self.data.Scenarios if self.varValue.x.get((s, r, h, l), 0) > 0.1
             if self.data.node_income_level[self.data.Trip_destination[r]] == 'High'])
            for (h, l) in opened_hl_pairs]  # how many times is the hub pari used
        hub_info['per3'] = [i/n_high_trips for i in hub_info['used times_high']]
        hub_info['used times_middle'] = [np.sum(
            [1 for r in self.data.Trips for s in self.data.Scenarios if self.varValue.x.get((s, r, h, l), 0) > 0.1
             if self.data.node_income_level[self.data.Trip_destination[r]] == 'Middle'])
            for (h, l) in opened_hl_pairs]  # how many times is the hub pari used
        hub_info['per2'] = [i/n_middle_trips for i in hub_info['used times_middle']]
        hub_info['served customers_total'] = [np.sum(
            [self.data.Trip_p_amount[s,r]*self.data.Scenarios_prob[s] for r in self.data.Trips for s in self.data.Scenarios if self.varValue.x.get((s, r, h, l), 0) > 0.1])
            for (h, l) in opened_hl_pairs]  # how many times is the hub pari used
        hub_info['per8'] = [i/(n_low_trips_amount+n_middle_trips_amount+n_high_trips_amount) for i in hub_info['served customers_total']]
        hub_info['served customers_high'] = [np.sum(
            [self.data.Trip_p_amount[s,r]*self.data.Scenarios_prob[s] for r in self.data.Trips for s in self.data.Scenarios if self.varValue.x.get((s, r, h, l), 0) > 0.1
             if self.data.node_income_level[self.data.Trip_destination[r]] == 'High'])
            for (h, l) in opened_hl_pairs]  # how many times is the hub pari used
        hub_info['per7'] = [i/n_high_trips_amount for i in hub_info['served customers_high']]
        hub_info['served customers_low'] = [np.sum(
            [self.data.Trip_p_amount[s,r]*self.data.Scenarios_prob[s] for r in self.data.Trips for s in self.data.Scenarios if
             self.varValue.x.get((s, r, h, l), 0) > 0.1
             if self.data.node_income_level[self.data.Trip_destination[r]] == 'Low'])
            for (h, l) in opened_hl_pairs]  # how many times is the hub pari used
        hub_info['per5'] = [i/n_low_trips_amount for i in hub_info['served customers_low']]
        hub_info['served customers_middle'] = [np.sum(
            [self.data.Trip_p_amount[s,r]*self.data.Scenarios_prob[s] for r in self.data.Trips for s in self.data.Scenarios if
             self.varValue.x.get((s, r, h, l), 0) > 0.1
             if self.data.node_income_level[self.data.Trip_destination[r]] == 'Middle'])
            for (h, l) in opened_hl_pairs]  # how many times is the hub pari used
        hub_info['per6'] = [i/n_middle_trips_amount for i in hub_info['served customers_middle']]
        df_hub_info = pd.DataFrame.from_dict(hub_info, orient='columns')
        
        # objective information
        obj_info = {'Facility cost': sum(self.data.Beta_hl[h, l] * self.varValue.z[h, l] for (h, l) in self.data.Hub_pairs),
                    'Transport cost (leader)':sum(
            self.data.Scenarios_prob[s] * sum(
                self.data.Trip_p_amount[s,r] * (sum(
                    self.data.tao_leader[h, l, s] * self.varValue.x.get((s, r, h, l),0) for (h, l) in
                    self.data.Hub_pairs) + sum(
                    self.data.gamma_leader[i, j, s] * self.varValue.y.get((s, r, i, j),0) for (i, j) in self.data.Node_pairs_for_mdl_sp[r])) for r
                in self.data.Trips) for s in
            self.data.Scenarios),
                    'Transport cost (follower)':sum(
            self.data.Scenarios_prob[s] * sum(
                self.data.Trip_p_amount[s,r] * (sum(
                    self.data.tao_follower[h, l, s] * self.varValue.x.get((s, r, h, l),0) for (h, l) in
                    self.data.Hub_pairs) + sum(
                    self.data.gamma_follower[i, j, s] * self.varValue.y.get((s, r, i, j), 0) for (i, j) in
                    self.data.Node_pairs_for_mdl_sp[r])) for r in self.data.Trips) for s in
            self.data.Scenarios)
                    }
        if model_type == 'single_leader':
            # use the correct single level tao and gamma, otherwise it's the same as the leader
            obj_info['Transport cost (follower)'] = sum(
            self.data.Scenarios_prob[s] * sum(
                self.data.Trip_p_amount[s,r] * (sum(
                    real_tao[h, l, s] * self.varValue.x.get((s, r, h, l),0) for (h, l) in
                    self.data.Hub_pairs) + sum(
                    real_gamma[i, j, s] * self.varValue.y.get((s, r, i, j), 0) for (i, j) in
                    self.data.Node_pairs_for_mdl_sp[r])) for r in self.data.Trips) for s in
            self.data.Scenarios)
        if model_type == 'single_follower':
            obj_info['Transport cost (leader)'] = sum(
            self.data.Scenarios_prob[s] * sum(
                self.data.Trip_p_amount[s,r] * (sum(
                    real_tao[h, l, s] * self.varValue.x.get((s, r, h, l),0) for (h, l) in
                    self.data.Hub_pairs) + sum(
                    real_gamma[i, j, s] * self.varValue.y.get((s, r, i, j),0) for (i, j) in self.data.Node_pairs_for_mdl_sp[r])) for r
                in self.data.Trips) for s in
            self.data.Scenarios)
        obj_info['Total = Facility cost + Transport cost (leader)'] = obj_info['Facility cost'] + obj_info[
            'Transport cost (leader)']

        obj_detail_info = {
            'file_name': [os.path.basename(self.data.file_name)[:-5]] * 9,
            'solution_type': [],
            'income': [],
            'transport_mode': [],
            'n_trips': [],
            'n_amount': [],
            'distance': [],
            'distance_product_amount': [],
            'leader_cost': [],
            'follower_cost': [],
            'facility_cost': [obj_info['Facility cost']] + [None] * 8,
            'total_cost': [obj_info['Total = Facility cost + Transport cost (leader)']] + [None] * 8,
            'travel_time': [],
            'cost_per_mile': [],
        }
        for income in ['High', 'Middle', 'Low']:
            for trans_mode in ['shuttle', 'shuttle+bus', 'all']:
                obj_detail_info['solution_type'].append(solution_type)
                obj_detail_info['income'].append(income)
                obj_detail_info['transport_mode'].append(trans_mode)
                if trans_mode == 'shuttle':
                    df = df_trip_info[(df_trip_info['income_level']==income) & (df_trip_info['hubs'].str.len() == 0)]
                elif trans_mode == 'shuttle+bus':
                    df = df_trip_info[(df_trip_info['income_level'] == income) & (df_trip_info['hubs'].str.len() >= 1)]
                else:
                    df = df_trip_info[df_trip_info['income_level'] == income]
                obj_detail_info['n_trips'].append(df['scenario_prob'].sum())
                obj_detail_info['n_amount'].append((df['amounts']*df['scenario_prob']).sum())
                obj_detail_info['distance'].append((df['follower travel distance']*df['scenario_prob']).sum())
                obj_detail_info['distance_product_amount'].append((df['amounts']*df['follower travel distance']*df['scenario_prob']).sum())
                obj_detail_info['leader_cost'].append((df['amounts']*df['leader cost']*df['scenario_prob']).sum())
                obj_detail_info['follower_cost'].append((df['amounts']*df['follower cost']*df['scenario_prob']).sum())
                obj_detail_info['travel_time'].append((df['amounts']*df['follower travel time']*df['scenario_prob']).sum())
                obj_detail_info['cost_per_mile'].append((df['amounts']*df['cost per mile']*df['scenario_prob']).sum())


        df_obj_detail_info = pd.DataFrame.from_dict(obj_detail_info)

        return df_trip_info.copy(), df_hub_info.copy(), obj_info.copy(), df_obj_detail_info.copy()

    def collect_out_sample_solution_info(self, solution_type):
        # out of sample cost
        obj_out_sample_info = {'Scenario': self.data.Scenarios,
                               'solution type':solution_type,
                               'Facility cost': sum(
            self.data.Beta_hl[h, l] * self.varValue.z[h, l] for (h, l) in self.data.Hub_pairs),
                               'Transport cost (leader)': sum(self.data.Scenarios_prob[s]*sum(
                self.data.Trip_p_amount[s,r] * (sum(
                    self.data.tao_leader[h, l, s] * self.varValue.x.get((s, r, h, l),0) for (h, l) in
                    self.data.Hub_pairs) + sum(
                    self.data.gamma_leader[i, j, s] * self.varValue.y.get((s, r, i, j), 0) for (i, j) in self.data.Node_pairs_for_mdl_sp[r])) for r
                in self.data.Trips) for s in
            self.data.Scenarios),
                               'Transport cost (follower)': sum(self.data.Scenarios_prob[s] * sum(
                                       self.data.Trip_p_amount[s,r] * (sum(
                                           self.data.tao_follower[h, l, s] * self.varValue.x.get((s, r, h, l),0) for (h, l) in
                                           self.data.Hub_pairs) + sum(
                                           self.data.gamma_follower[i, j, s] * self.varValue.y.get((s, r, i, j), 0) for (i, j) in
                                           self.data.Node_pairs_for_mdl_sp[r])) for r
                                       in self.data.Trips) for s in
                                   self.data.Scenarios)
                               }
        df_obj_out_sample = pd.DataFrame.from_dict(obj_out_sample_info)
        df_obj_out_sample['Total = Facility cost + Transport cost (leader)'] = df_obj_out_sample['Facility cost'] + \
                                                                               df_obj_out_sample[
                                                                                   'Transport cost (leader)']
        df_obj_out_sample['Stochastic is better'] = None

        return df_obj_out_sample

def write_result(file_name=None):
    # write result
    current_directory = os.path.dirname(os.path.abspath(__file__))
    upper_2_dir = os.path.dirname(current_directory)
    tmp = os.path.join(upper_2_dir, 'output')
    if file_name is None:
        file_name = os.path.join(tmp, 'result_CCG.xlsx')
    else:
        file_name = os.path.join(tmp, file_name)

    df = pd.DataFrame.from_dict(result_ccg, orient='columns')
    df.to_excel(file_name, index=False)




if __name__ == '__main__':
    result = {
        'file': [],
        'ub': [],
        'lb': [],
        'MIPGap': [],
        'solve_time': []
    }

    result_ccg = {'node-hub-trip-passenger': [],
              'tag': [],
              'agg_cut': [],
              'parallel method':[],
              'arc elim': [],
              'Delta type': [],
              'lp warm start': [],
              'solution approach': [],
              'primal also dual': [],
              'time read data': [],
              'time model creation': [],
              'solution time': [],
              'total solution time': [],
              'solution time MP': [],
              'total solution time MP': [],
              'solution time SP': [],
              'total solution time SP': [],
              'time retrieve MP info': [],
              'total time retrieve MP info': [],
              'time retrieve SP info': [],
              'total time retrieve SP info': [],
              'time update MP': [],
              'total time update MP': [],
              'time update SP': [],
              'total time update SP': [],
              'LB record': [],
              'UB record': [],
              'final_lb': [],
              'final_ub': [],
              'gap': [],
              'pure solve time': [],
              'n iters': [],
              'n_cut_each_iter': []
              }


    current_directory = os.path.dirname(os.path.abspath(__file__))
    upper_2_dir = os.path.dirname(current_directory)
    file_path = os.path.join(upper_2_dir, 'data', 'cluster')
    file_list = os.listdir(file_path)
    file_list_original = [i for i in file_list if '.xlsx' in i]
    file_list = [os.path.join(file_path, f) for f in file_list_original]
    file_list.sort()

    excel_output_file = os.path.join(upper_2_dir, 'output', 'cal_info_comb_bilevel.xlsx')

    for file_name in file_list:
        data = Modeldata(file_name=file_name, arc_elimination=True, Delta_type='MST', Delta_value=5)
        console_output_file = os.path.join(upper_2_dir, 'output',
                                           'console_info_{}.txt'.format(os.path.basename(data.file_name)[:-5]))
        with open(console_output_file, 'a') as file:
            print('================BEGIN NEW FILE======================', file=file)
            print('execute: ' + file_name, file=file)

        CCG_iterator_stoch = IterateComb(data=data, agg_cut=False,
                                     solution_appro='primal',
                                     primal_based_dual_to_ini=False,
                                     use_lp_basis=False,
                                     parallel_method='process',
                                     agg_cut_part=False,
                                     solve_MP_use_Benders=False,
                                     revise_dual_value=True)
        set_hub = [23, 47, 60, 79, 87, 69]
        opened_z = [(h,l) for h in set_hub for l in set_hub if h!=l]
        # opened_z = {(23,47), (60,79)}  # 23, 47, 60, 79, 87, 69
        diff_sum = 0
        for s,r in CCG_iterator_stoch.unexplred_probles:
            if r in data.Trips:
                obj1 = CCG_iterator_stoch.cal_lb_unexplored_single_level_leader_obj(s=1,r=r)
                obj2 = CCG_iterator_stoch.cal_lb_unexplored_bilevel_leader_obj(s=1, r=r, opened_z=opened_z)
                diff_sum += obj2 - obj1
                print('r={}, obj1={:.4f}, obj2={:.4f}'.format(r, obj1, obj2))

        print('diff_sum={:.4f}'.format(diff_sum))
        exit()





        parser = get_parser()
        args = parser.parse_args()
        CCG_timelimit = args.CCG_timelimit
        CCG_iterator_stoch.start_Benders_iteration(timelimit=CCG_timelimit, use_i_ccg=True)
        df_trip_info_stoch, df_hub_info_stoch, obj_info_stoch,_ = CCG_iterator_stoch.collect_solution_info(solution_type='stochastic')
        # dict_z_stoch, _, _ = write_solution(CCG_iterator=CCG_iterator_stoch, file_tag='stoch')
        result_ccg = CCG_iterator_stoch.update_result_ccg(result=result_ccg,
                                                      read_data_time=0, tag='stochastic')

        write_result(file_name='calculation_info_CCG_and_combi.xlsx')

        file_path = os.path.join(upper_2_dir, 'output',
                                 os.path.basename(file_name)[:-5] + 'opt_solution_info.xlsx')
        with pd.ExcelWriter(file_path) as writer:
            df_trip_info_stoch.to_excel(excel_writer=writer, sheet_name='trip_info', index=False)
            df_hub_info_stoch.to_excel(excel_writer=writer, sheet_name='hub_info', index=False)
            # obj_info_stoch.to_excel(excel_writer=writer, sheet_name='obj_info_in_sample', index=True)

        del CCG_iterator_stoch












