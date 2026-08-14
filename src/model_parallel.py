# -*- coding: utf-8 -*-
"""
  Author: Author
  Email : liusuri@mail.dlut.edu.cn
   Time : 2024/11/01 17:22
Function: Model with subproblems solved in parallel, restrict the set N to reduce model size
Remark: 20250827: The following classes are used in other files:
                    a. VarValue
                    b. SubModel
                    c. SubShortestPathWithStrongDualModel
                    d. OneLayerModel
                  The class Iteration is no longer required. It's associated with the very early version of the cutting plane algorithm.

"""

import os

import gurobipy as gp
import pandas as pd
from gurobipy import GRB
from read_data import Modeldata
import time
from matplotlib import pyplot as plt
import re
import multiprocessing as mp
from random import sample
import datetime
from multiprocessing.pool import ThreadPool
import json
import numpy as np
import gc
import psutil
import sys
from pympler import asizeof
import argparse

ALLOW_CORE = 16  # number of CPU cores allowed in the response search


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--CCG_timelimit", type=int, help="time limit of CCG algo", default=3600*12)
    parser.add_argument("--MP_timelimit", type=int, help="time limit of MP", default=720)  # spend less time on MP. The i-ccg algo will solve MP repeatedly when required.
    parser.add_argument("--use_i_ccg", type=int, help="whether use i-CCG algo", default=True)
    parser.add_argument("--MP_gap_scale", type=int, help="MIPGap = MIP * scale", default=0.15)
    parser.add_argument("--inexact_gap_threshold", type=int, help="inexact_gap_threshold", default=0.05)
    parser.add_argument("--stop_gap", type=int, help="stop gap CCG", default=1e-5)
    parser.add_argument("--MP_ini_gap", type=int, help="initial gap of MP", default=0.1)

    parser.add_argument("--Single_Level_timelimit", type=int, help="timelimit of sinvle level algorithm", default=3600)

    return parser


class MasterModel():
    def __init__(self, data: Modeldata, agg_cut: bool, solution_appro: bool=False, agg_cut_part=False, on_linux=False, use_mdl_copy=False, copy_mdl=None):
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
        # self.MP_timelimit = 900  # initial MP time limit
        self.solved_to_optimal = False
        self.obj_lb_is_0 = True  # the initial lower bound of obj

        self.data = data
        self.agg_cut = agg_cut
        self.solution_appro = solution_appro
        self.agg_cut_part = agg_cut_part
        self.on_linux = on_linux
        current_directory = os.path.dirname(os.path.abspath(__file__))
        upper_2_dir = os.path.dirname(current_directory)
        data_file = os.path.basename(data.file_name)[:-5]
        # log_file_prefix = upper_2_dir + '/output/' + data_file
        log_file_prefix = os.path.join(upper_2_dir, 'output')
        log_file_prefix = os.path.join(log_file_prefix, data_file)

        self.console_output_file = os.path.join(upper_2_dir, 'output', 'console_info.txt')

        if not use_mdl_copy:
            self.env = gp.Env(log_file_prefix + '_Master')
            self.model = gp.Model('MasterModel', self.env)
            if ':' not in os.path.abspath(__file__):
                current_directory = os.path.dirname(os.path.abspath(__file__))
                upper_2_dir = os.path.dirname(current_directory)  # resembles '..'
                # model_to_disk_file = os.path.join(upper_2_dir, 'model2disk')  # write the model to disk instead of memory
                # self.model.setParam('NodefileDir', model_to_disk_file)
                # self.model.setParam('NodefileStart', 0.45)  # 当使用内存达到45%时开始写入磁盘
            self.var = self.Variable(model=self.model, data=data, agg_cut=agg_cut, solution_appro=self.solution_appro, agg_cut_part=agg_cut_part)
            self.cons = self.Constraint()
            self.add_constraints()
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

        def __init__(self, model, data: Modeldata, agg_cut: bool, solution_appro: bool, agg_cut_part=False):
            # vars in the master problem
            self.z = model.addVars(data.Hub_pairs, vtype=GRB.BINARY, name='z')
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


    def add_var_and_cons_ccg_primal(self, varValue, s, r):
            # self.theta1 = model.addVars(
            #     gp.tuplelist((s, r, i) for s in data.Scenarios for r in data.Trips for i in data.Node),
            #     vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name='theta1')
            # self.theta2 = model.addVars(
            #     gp.tuplelist((s, r, h, l) for s in data.Scenarios for r in data.Trips for (h, l) in data.Hub_pairs),
            #     vtype=GRB.CONTINUOUS, name='theta2')
            # self.theta3 = model.addVars(
            #     gp.tuplelist((s, r, h, l) for s in data.Scenarios for r in data.Trips for (h, l) in data.Hub_pairs),
            #     vtype=GRB.CONTINUOUS, name='theta3')
            # self.theta4 = model.addVars(gp.tuplelist(
            #     (s, r, i, j) for s in data.Scenarios for r in data.Trips for (i, j) in data.Node_pairs_for_mdl_sp[r]),
            #                             vtype=GRB.CONTINUOUS, name='theta4')
            # self.delta = model.addVars(
            #     gp.tuplelist((s, r, h, l) for s in data.Scenarios for r in data.Trips for (h, l) in data.Hub_pairs),
            #     vtype=GRB.CONTINUOUS, name='delta')

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
                self.var.z[key].setAttr('lb', given_z.get(str(key), 0))
                self.var.z[key].setAttr('ub', given_z.get(str(key), 1))

    def free_z(self):
        for key in self.var.z:
            self.var.z[key].setAttr('lb', 0)
            self.var.z[key].setAttr('ub', 1)

    def solve_model(self, use_i_ccg=False, varValue=None):
        self.model.optimize()
        while self.model.solCount <= 0 and self.model.Status != 3:
            with open(self.console_output_file, 'a') as file:
                print('No feasible MP solution is found, continue to optimize', file=file)
            self.model.optimize()
        if self.model.Status == 3:
            with open(self.console_output_file, 'a') as file:
                print('\n\n!!!!!!!!!!!!!!MP infeasible!!!!!!!!!!!!!!\n\n', file=file)
        if abs(self.model.MIPGap) < 5e-5:
            # if Status code is 2, that means gap is in the permitted MIPGap rather really optimal!!
            self.solved_to_optimal = True
        else:
            self.solved_to_optimal = False
        with open(self.console_output_file, 'a') as file:
            print('Master model is solved, gap = {}'.format(self.model.MIPGap), file=file)
        # self.present_solutions()

        if len(varValue.pi1) > 0:
            for s,r in self.data.Sce_Trip:
                if self.model.getConstrByName('gamma_lb_{}_{}_{}'.format(s,r,self.var.n_iters)):
                    rhs_round = -varValue.pi1[s, r, self.data.Trip_origin[r]] + varValue.pi1[s, r,
                        self.data.Trip_destination[r]] - sum(
                        round(self.var.z[h, l].X) * varValue.pi2[s, r, h, l] for (h, l) in self.data.Hub_pairs) - sum(
                        varValue.pi3[s, r, h, l] for (h, l) in self.data.Hub_pairs) - sum(
                        varValue.pi4[s, r, i, j] for (i, j) in self.data.Node_pairs_for_mdl_sp[r]) + varValue.t[s, r] * (
                            self.var.theta1[s, r, self.data.Trip_origin[r]].X -
                            self.var.theta1[s, r, self.data.Trip_destination[r]].X + sum(
                        self.var.delta[s, r, h, l].X for (h, l) in self.data.Hub_pairs) + sum(
                        self.var.theta3[s, r, h, l].X for (h, l) in self.data.Hub_pairs) + sum(
                        self.var.theta4[s, r, i, j].X for (i, j) in self.data.Node_pairs_for_mdl_sp[r]))

                    if self.var.gamma[s, r].X < rhs_round - 0.1:
                        with open(self.console_output_file, 'a') as file:
                            print('\t\t\t <<<<s={}, r={}, gamma={:.4f}, rhs={:.4f}'.format(s,r,self.var.gamma[s,r].X, rhs_round), file=file)




    def update_obj_lb(self, obj_lb=0):
        """
        update the lower bound of obj
        :return:
        """
        c = self.model.getConstrByName('obj_lb')
        if c:
            self.model.remove(c)
        self.model.addConstr(self.obj >= obj_lb, name='obj_lb')

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

    def update_MP_timelimit(self, dt):
        self.MP_timelimit += dt
        self.model.setParam('TimeLimit', self.MP_timelimit)


    def set_params(self):
        self.model.setParam('TimeLimit', self.args.MP_timelimit)
        # self.model.setParam('FeasibilityTo', 1e-1)
        # self.model.setParam('Presolve', 0)
        self.model.setParam('LogToConsole', 0)
        self.model.setParam('Heuristics', 0.1)  # default is 0.05
        # self.model.setParam('PrePasses', 3)  # limit the time on presolve
        # self.model.setParam('Cuts', 2)  # aggressive cut generation
        # self.model.setParam('MIPFocus', 1)  # finding feasible solutions quickly
        self.model.setParam('MIPGap', self.MIP_gap)  # MIP Gap
        # self.model.setParam('Method', 0)  # primal simplex for the initial root relaxation, otherwise barrier will be chosed, which is numerically sensitive

    def present_solutions(self):
        print('print the solutions of master problem: z')
        for key in self.var.z:
            if self.var.z[key].X > 1e-4:
                print(key, self.var.z[key].X)


class SubModel():
    def __init__(self, data: Modeldata, s, r):
        """
        :param data: input data
        :param s: parameter s in the mathematical model
        :param r: parameter r in the mathematical model
        """
        self.data = data
        self.s = s
        self.r = r

    def declare_model(self):
        log_name = self.data.file_name[:-5]
        self.env = gp.Env()
        self.model = gp.Model('SubModel', self.env)
        self.var = self.Variable(model=self.model, data=self.data, r=self.r)
        self.cons = self.Constraint()
        self.add_constraints()
        self.set_params()

    def create_and_solve_by_with(self, s, r, param_z, use_lp_basis=False, VBasis=None, CBasis=None, VBasis_info=None):
        """
        create and solve using the with clause
        :param param_z: only the varvalue of z is required
        :param ini_sol: initial solution
        :param use_lp_basis: whether use lp basis for warm start
        """
        # log_name = self.data.file_name[:-5]
        # env_name = log_name + '_sub_{}_{}'.format(s, r)
        with gp.Env() as env, gp.Model('SubModel', env=env) as model:
            # add variables
            pi1 = model.addVars(self.data.Node_of_trip[r], vtype=GRB.CONTINUOUS, name='pi_1', lb=-GRB.INFINITY)
            pi2 = model.addVars(self.data.Hub_pairs, vtype=GRB.CONTINUOUS, name='pi_2')
            pi3 = model.addVars(self.data.Hub_pairs, vtype=GRB.CONTINUOUS, name='pi_3')
            pi4 = model.addVars(self.data.Node_pairs_for_mdl_sp[r], vtype=GRB.CONTINUOUS, name='pi_4')
            nu1 = model.addVars(self.data.Hub_pairs, vtype=GRB.CONTINUOUS, name='nu_1')
            nu2 = model.addVars(self.data.Node_pairs_for_mdl_sp[r], vtype=GRB.CONTINUOUS, name='nu_2')
            t = model.addVar(vtype=GRB.CONTINUOUS, name='t')

            # add constraint
            model.addConstrs((
                pi1[h] - pi1[l] + pi2[h, l] + pi3[h, l] + self.data.tao_follower[
                    h, l, s] * t >= -self.data.tao_leader[h, l, s] for (h, l) in
            self.data.Hub_pairs),
                name='c_tau')
            model.addConstrs((
                pi1[i] - pi1[j] + pi4[i, j] + self.data.gamma_follower[
                    i, j, s] * t >= -self.data.gamma_leader[i, j, s] for (i, j) in
            self.data.Node_pairs_for_mdl_sp[r]),
                name='c_gamma')

            i = self.data.Trip_origin[r]
            model.addConstr((
                    gp.quicksum(
                        -nu1[i, l] for l in self.data.Hub if (i, l) in self.data.Hub_pairs) + gp.quicksum(
                nu1[l, i] for l in self.data.Hub if (l, i) in self.data.Hub_pairs) - gp.quicksum(
                nu2[i, j] for j in self.data.Node_of_trip[r] if (i, j) in self.data.Node_pairs_for_mdl_sp[r]) + gp.quicksum(
                nu2[j, i] for j in self.data.Node_of_trip[r] if (j, i) in self.data.Node_pairs_for_mdl_sp[r]) +
                    t == 0), name='c_nu_oringn')

            i = self.data.Trip_destination[r]
            model.addConstr((
                    gp.quicksum(
                        -nu1[i, l] for l in self.data.Hub if (i, l) in self.data.Hub_pairs) + gp.quicksum(
                nu1[l, i] for l in self.data.Hub if (l, i) in self.data.Hub_pairs) - gp.quicksum(
                nu2[i, j] for j in self.data.Node_of_trip[r] if (i, j) in self.data.Node_pairs_for_mdl_sp[r]) + gp.quicksum(
                nu2[j, i] for j in self.data.Node_of_trip[r] if (j, i) in self.data.Node_pairs_for_mdl_sp[r]) -
                    t == 0), name='c_nu_destination')

            model.addConstrs((
                gp.quicksum(-nu1[i, l] for l in self.data.Hub if (i, l) in self.data.Hub_pairs) + gp.quicksum(
                    nu1[l, i] for l in self.data.Hub if (l, i) in self.data.Hub_pairs) - gp.quicksum(
                    nu2[i, j] for j in self.data.Node_of_trip[r] if (i, j) in self.data.Node_pairs_for_mdl_sp[r]) + gp.quicksum(
                    nu2[j, i] for j in self.data.Node_of_trip[r] if (j, i) in self.data.Node_pairs_for_mdl_sp[r]) == 0 for i in
                self.data.Node_of_trip[r] if i not in [self.data.Trip_origin[r], self.data.Trip_destination[r]]),
                name='c_nu_normal')

            model.addConstrs(
                (-nu1[h, l] + t >= 0 for (h, l) in self.data.Hub_pairs), name='nu1_t')
            model.addConstrs(
                (-nu2[i, j] + t >= 0 for (i, j) in self.data.Node_pairs_for_mdl_sp[r]), name='nu2_t')

            # update objective and constraints
            obj = -pi1[self.data.Trip_origin[r]] + pi1[self.data.Trip_destination[r]] - gp.quicksum(
                param_z[h, l] * pi2[h, l] for (h, l) in self.data.Hub_pairs) - gp.quicksum(
                pi3) - gp.quicksum(pi4) - gp.quicksum(
                self.data.tao_follower[h, l, s] * nu1[h, l] for (h, l) in self.data.Hub_pairs) - gp.quicksum(
                self.data.gamma_follower[i, j, s] * nu2[i, j] for (i, j) in self.data.Node_pairs_for_mdl_sp[r])
            model.setObjective(obj, sense=GRB.MAXIMIZE)

            model.addConstrs((
                - nu1[h, l] + param_z[h, l] * t >= 0 for (h, l) in self.data.Hub_pairs),
                name='nu_z_t')

            # warm start
            if use_lp_basis:
                if VBasis is not None:
                    model.update()
                    model.setAttr('VBasis', model.getVars(), VBasis)
                    model.setAttr('CBasis', model.getConstrs(), CBasis)
                    model.setParam('LPWarmStart', 2)  # enable LP warm start
                if VBasis_info is not None:
                    model.setAttr('VBasis', pi1.values(), VBasis_info['pi1'])
                    model.setAttr('VBasis', pi2.values(), VBasis_info['pi2'])
                    model.setAttr('VBasis', pi3.values(), VBasis_info['pi3'])
                    model.setAttr('VBasis', pi4.values(), VBasis_info['pi4'])
                    model.setAttr('VBasis', t, VBasis_info['t'])

            # model.setParam('TimeLimit', 360)  # 10 minutes
            model.setParam('LogToConsole', 0)
            model.optimize()

            if s == self.data.Scenarios[0] and use_lp_basis:
                x = {(h, l): - model.getConstrByName(f'c_tau[{h},{l}]').pi for (h, l) in self.data.Hub_pairs}
                y = {(i, j): - model.getConstrByName(f'c_gamma[{i},{j}]').pi for (i, j) in self.data.Node_pairs_for_mdl_sp[r]}
                VBasis = model.getAttr('VBasis', model.getVars())
                CBasis = model.getAttr('CBasis', model.getConstrs())
            else:
                x = {}
                y = {}
                VBasis = None
                CBasis = None

            result = {'pi1': {key: pi1[key].X for key in pi1},
                      'pi2': {key: pi2[key].X for key in pi2},
                      'pi3': {key: pi3[key].X for key in pi3},
                      'pi4': {key: pi4[key].X for key in pi4},
                      't': t.X,
                      'x': x,
                      'y': y,
                      'obj': obj.getValue(),
                      'VBasis': VBasis,
                      'CBasis': CBasis}
            return {(s,r): result}

    class Variable():
        """class of variables"""

        def __init__(self, model, data: Modeldata, r):
            self.pi1 = model.addVars(data.Node, vtype=GRB.CONTINUOUS, name='pi_1', lb=-GRB.INFINITY)
            self.pi2 = model.addVars(data.Hub_pairs, vtype=GRB.CONTINUOUS, name='pi_2')
            self.pi3 = model.addVars(data.Hub_pairs, vtype=GRB.CONTINUOUS, name='pi_3')
            self.pi4 = model.addVars(data.Node_pairs_for_mdl_sp[r], vtype=GRB.CONTINUOUS, name='pi_4')
            self.nu1 = model.addVars(data.Hub_pairs, vtype=GRB.CONTINUOUS, name='nu_1')
            self.nu2 = model.addVars(data.Node_pairs_for_mdl_sp[r], vtype=GRB.CONTINUOUS, name='nu_2')
            self.t = model.addVar(vtype=GRB.CONTINUOUS, name='t')



    class Constraint():
        """class of constraint"""

        def __init__(self):
            self.nu_z_t = None  # the constraints related to the value of z (master var)

    def add_constraints(self):
        self.cons.c_tau = self.model.addConstrs((
            self.var.pi1[h] - self.var.pi1[l] + self.var.pi2[h, l] + self.var.pi3[h, l] + self.data.tao_follower[
                h, l, self.s] * self.var.t >= -self.data.tao_leader[h, l, self.s] for (h, l) in self.data.Hub_pairs),
            name='c_tau')
        self.cons.c_gamma = self.model.addConstrs((
            self.var.pi1[i] - self.var.pi1[j] + self.var.pi4[i, j] + self.data.gamma_follower[
                i, j, self.s] * self.var.t >= -self.data.gamma_leader[i, j, self.s] for (i, j) in self.data.Node_pairs_for_mdl_sp[self.r]),
            name='c_gamma')

        # self.cons.c_tau = self.model.addConstrs((
        #     self.var.pi1[h] - self.var.pi1[l] + self.var.pi2[h, l] + self.var.pi3[h, l] + self.data.tao_follower[
        #         h, l, self.s] * self.var.t >= -1 for (h, l) in self.data.Hub_pairs),
        #     name='c_tau')
        # self.cons.c_gamma = self.model.addConstrs((
        #     self.var.pi1[i] - self.var.pi1[j] + self.var.pi4[i, j] + self.data.gamma_follower[
        #         i, j, self.s] * self.var.t >= -1 for (i, j) in self.data.Node_pairs_for_mdl_sp[r]),
        #     name='c_gamma')

        i = self.data.Trip_origin[self.r]
        self.cons.c_nu_oringn = self.model.addConstr((
                gp.quicksum(-self.var.nu1[i, l] for l in self.data.Hub if (i, l) in self.data.Hub_pairs) + gp.quicksum(
            self.var.nu1[l, i] for l in self.data.Hub if (l, i) in self.data.Hub_pairs) - gp.quicksum(
            self.var.nu2[i, j] for j in self.data.Node_of_trip[self.r] if (i, j) in self.data.Node_pairs_for_mdl_sp[self.r]) + gp.quicksum(
            self.var.nu2[j, i] for j in self.data.Node_of_trip[self.r] if (j, i) in self.data.Node_pairs_for_mdl_sp[self.r]) +
                self.var.t == 0), name='c_nu_oringn')

        i = self.data.Trip_destination[self.r]
        self.cons.c_nu_destination = self.model.addConstr((
                gp.quicksum(-self.var.nu1[i, l] for l in self.data.Hub if (i, l) in self.data.Hub_pairs) + gp.quicksum(
            self.var.nu1[l, i] for l in self.data.Hub if (l, i) in self.data.Hub_pairs) - gp.quicksum(
            self.var.nu2[i, j] for j in self.data.Node_of_trip[self.r] if (i, j) in self.data.Node_pairs_for_mdl_sp[self.r]) + gp.quicksum(
            self.var.nu2[j, i] for j in self.data.Node_of_trip[self.r] if (j, i) in self.data.Node_pairs_for_mdl_sp[self.r]) -
                self.var.t == 0), name='c_nu_destination')

        self.cons.c_nu_normal = self.model.addConstrs((
            gp.quicksum(-self.var.nu1[i, l] for l in self.data.Hub if (i, l) in self.data.Hub_pairs) + gp.quicksum(
                self.var.nu1[l, i] for l in self.data.Hub if (l, i) in self.data.Hub_pairs) - gp.quicksum(
                self.var.nu2[i, j] for j in self.data.Node_of_trip[self.r] if (i, j) in self.data.Node_pairs_for_mdl_sp[self.r]) + gp.quicksum(
                self.var.nu2[j, i] for j in self.data.Node_of_trip[self.r] if (j, i) in self.data.Node_pairs_for_mdl_sp[self.r]) == 0 for i in
            self.data.Node_of_trip[self.r] if i not in [self.data.Trip_origin[self.r], self.data.Trip_destination[self.r]]),
            name='c_nu_normal')

        self.cons.nu1_t = self.model.addConstrs(
            (-self.var.nu1[h, l] + self.var.t >= 0 for (h, l) in self.data.Hub_pairs), name='nu1_t')
        self.cons.nu2_t = self.model.addConstrs(
            (-self.var.nu2[i, j] + self.var.t >= 0 for (i, j) in self.data.Node_pairs_for_mdl_sp[self.r]), name='nu2_t')

    def update_objective_cons(self, param_z):
        """
        update the objective and one constriant
        :param param_z: z (type=dict) in varValue
        :return: None
        """
        r = self.r
        s = self.s
        self.obj = -self.var.pi1[self.data.Trip_origin[r]] + self.var.pi1[self.data.Trip_destination[r]] - gp.quicksum(
            param_z[h, l] * self.var.pi2[h, l] for (h, l) in self.data.Hub_pairs) - gp.quicksum(
            self.var.pi3) - gp.quicksum(self.var.pi4) - gp.quicksum(
            self.data.tao_follower[h, l, s] * self.var.nu1[h, l] for (h, l) in self.data.Hub_pairs) - gp.quicksum(
            self.data.gamma_follower[i, j, s] * self.var.nu2[i, j] for (i, j) in self.data.Node_pairs_for_mdl_sp[r])
        self.model.setObjective(self.obj, sense=GRB.MAXIMIZE)

        if self.cons.nu_z_t is not None:
            self.model.remove(self.cons.nu_z_t)
            self.model.update()
        self.cons.nu_z_t = self.model.addConstrs((
            - self.var.nu1[h, l] + param_z[h, l] * self.var.t >= 0 for (h, l) in self.data.Hub_pairs), name='nu_z_t')


    def write_lp_basis_info(self, use_lp_basis=False, VBasis=None, CBasis=None, VBasis_info:dict=None):
        if use_lp_basis:
            if VBasis is not None:
                self.model.update()
                self.model.setAttr('VBasis', self.model.getVars(), VBasis)
                self.model.setAttr('CBasis', self.model.getConstrs(), CBasis)
                self.model.setParam('LPWarmStart', 2)  # enable LP warm start
            if VBasis_info is not None:
                self.model.update()
                self.model.setAttr('VBasis', list(self.var.pi1.values()), VBasis_info['pi1'])
                self.model.setAttr('VBasis', list(self.var.pi2.values()), VBasis_info['pi2'])
                self.model.setAttr('VBasis', list(self.var.pi3.values()), VBasis_info['pi3'])
                self.model.setAttr('VBasis', list(self.var.pi4.values()), VBasis_info['pi4'])
                self.model.setAttr('VBasis', self.var.t, VBasis_info['t'])
                self.model.setParam('LPWarmStart', 2)  # enable LP warm start

    def obtain_solution_info(self, use_lp_basis=False):
        if self.s == self.data.Scenarios[0] and use_lp_basis:
            x = {(h, l): - self.model.getConstrByName(f'c_tau[{h},{l}]').pi for (h, l) in self.data.Hub_pairs}
            y = {(i, j): - self.model.getConstrByName(f'c_gamma[{i},{j}]').pi for (i, j) in self.data.Node_pairs_for_mdl_sp[self.r]}
            VBasis = self.model.getAttr('VBasis', self.model.getVars())
            CBasis = self.model.getAttr('CBasis', self.model.getConstrs())
        else:
            VBasis = None
            CBasis = None
            x = None
            y = None


        result = {'pi1': {key: self.var.pi1[key].X for key in self.var.pi1},
                  'pi2': {key: self.var.pi2[key].X for key in self.var.pi2},
                  'pi3': {key: self.var.pi3[key].X for key in self.var.pi3},
                  'pi4': {key: self.var.pi4[key].X for key in self.var.pi4},
                  't': self.var.t.X,
                  'x': x,
                  'y': y,
                  'obj': self.model.getObjective().getValue(),
                  'VBasis': VBasis,
                  'CBasis': CBasis
                  }
        return result

    # def solve_model(self, use_lp_basis=False, VBasis=None, CBasis=None, VBasis_info:dict=None):
    #     """
    #     solve model
    #     :param VBasis_info: this info is only from primal model
    #     :return:
    #     """
    #     tmp = time.time()
    #     if use_lp_basis:
    #         if VBasis is not None:
    #             self.model.update()
    #             self.model.setAttr('VBasis', self.model.getVars(), VBasis)
    #             self.model.setAttr('CBasis', self.model.getConstrs(), CBasis)
    #             self.model.setParam('LPWarmStart', 2)  # enable LP warm start
    #         if VBasis_info is not None:
    #             self.model.update()
    #             self.model.setAttr('VBasis', self.var.pi1.values(), VBasis_info['pi1'])
    #             self.model.setAttr('VBasis', self.var.pi2.values(), VBasis_info['pi2'])
    #             self.model.setAttr('VBasis', self.var.pi3.values(), VBasis_info['pi3'])
    #             self.model.setAttr('VBasis', self.var.pi4.values(), VBasis_info['pi4'])
    #             self.model.setAttr('VBasis', self.var.t, VBasis_info['t'])
    #             self.model.setParam('LPWarmStart', 2)  # enable LP warm start
    #     t_update = time.time() - tmp
    #
    #     tmp = time.time()
    #     self.model.optimize()
    #     t_optimize = time.time() - tmp
    #
    #
    #     # TODO: in dual, disable this lines.
    #     # TODO: check, time consumption on .pi
    #
    #
    #     tmp = time.time()
    #     if self.s == self.data.Scenarios[0] and use_lp_basis:
    #         x = {(h, l): - self.model.getConstrByName(f'c_tau[{h},{l}]').pi for (h, l) in self.data.Hub_pairs}
    #         y = {(i, j): - self.model.getConstrByName(f'c_gamma[{i},{j}]').pi for (i, j) in self.data.Node_pairs_for_mdl_sp[r]}
    #         VBasis = self.model.getAttr('VBasis', self.model.getVars())
    #         CBasis = self.model.getAttr('CBasis', self.model.getConstrs())
    #     else:
    #         VBasis = None
    #         CBasis = None
    #         x = None
    #         y = None
    #     time_retrieval = time.time() - tmp
    #
    #     result = {'pi1': {key: self.var.pi1[key].X for key in self.var.pi1},
    #               'pi2': {key: self.var.pi2[key].X for key in self.var.pi2},
    #               'pi3': {key: self.var.pi3[key].X for key in self.var.pi3},
    #               'pi4': {key: self.var.pi4[key].X for key in self.var.pi4},
    #               't': self.var.t.X,
    #               'x': x,
    #               'y': y,
    #               'obj': self.model.getObjective().getValue(),
    #               'VBasis': VBasis,
    #               'CBasis': CBasis,
    #               't_optimize': t_optimize,
    #               't_update': t_update,
    #               't_retrieval': time_retrieval
    #               }
    #     return result

    def set_params(self):
        # self.model.setParam('TimeLimit', 360)  # 10 minutes
        # self.model.setParam('InfUnbdInfo', 1)
        # self.model.setParam('FeasibilityTo', 1e-1)
        # self.model.setParam('Presolve', 0)
        # self.model.setParam('Heuristics', 0)
        self.model.setParam('LogToConsole', 0)

    def present_solutions(self):
        print('print the solutions')


class SubShortestPathWithStrongDualModel():
    def __init__(self, data: Modeldata, s, r):
        """
        shortest pats model
        :param data:
        :param s:
        :param r:
        """
        self.data = data
        self.s = s
        self.r = r
        self.cons_conn = None

        current_directory = os.path.dirname(os.path.abspath(__file__))
        upper_2_dir = os.path.dirname(current_directory)
        self.console_output_file = os.path.join(upper_2_dir, 'output', 'console_info.txt')

    def declare_model(self):
        log_name = self.data.file_name[:-5]
        self.env = gp.Env()
        self.model = gp.Model('SubModelPrimal', self.env)
        self.var = self.Variable(model=self.model, data=self.data, r=self.r)
        self.cons = self.Constraint()
        self.add_constraints()
        self.set_params()

    class Variable():
        def __init__(self, model, data: Modeldata, r):
            self.x = model.addVars(
                gp.tuplelist((h, l) for (h, l) in data.Hub_pairs),
                vtype=GRB.CONTINUOUS, name='x')
            self.y = model.addVars(
                gp.tuplelist((i, j) for (i, j) in data.Node_pairs_for_mdl_sp[r]),
                vtype=GRB.CONTINUOUS, name='y')

    class Constraint():
        def __init__(self):
            self.strong_dual = None

    def add_constraints(self):
        # add constraints
        i = self.data.Trip_origin[self.r]
        self.model.addConstr((gp.quicksum(
            self.var.x[i, h] - self.var.x[h, i] for h in self.data.Hub if
            (i, h) in self.data.Hub_pairs) + gp.quicksum(
            self.var.y[i, j] for j in self.data.Node_of_trip[self.r] if
            (i, j) in self.data.Node_pairs_for_mdl_sp[self.r]) - gp.quicksum(self.var.y[j, i] for j in self.data.Node_of_trip[self.r] if
            (j, i) in self.data.Node_pairs_for_mdl_sp[self.r]) == 1),
                        name='c_flb_or')
        i = self.data.Trip_destination[self.r]
        self.model.addConstr((gp.quicksum(
            self.var.x[i, h] - self.var.x[h, i] for h in self.data.Hub if
            (i, h) in self.data.Hub_pairs) + gp.quicksum(
            self.var.y[i, j] for j in self.data.Node_of_trip[self.r] if
            (i, j) in self.data.Node_pairs_for_mdl_sp[self.r]) - gp.quicksum(self.var.y[j, i] for j in self.data.Node_of_trip[self.r] if
            (j, i) in self.data.Node_pairs_for_mdl_sp[self.r]) == -1),
                        name='c_flb_de')
        self.model.addConstrs((gp.quicksum(
            self.var.x[i, h] - self.var.x[h, i] for h in self.data.Hub if
            (i, h) in self.data.Hub_pairs) + gp.quicksum(
            self.var.y[i, j] for j in self.data.Node_of_trip[self.r] if
            (i, j) in self.data.Node_pairs_for_mdl_sp[self.r]) - gp.quicksum(self.var.y[j, i] for j in self.data.Node_of_trip[self.r] if
            (j, i) in self.data.Node_pairs_for_mdl_sp[self.r]) == 0 for i in
                          self.data.Node_of_trip[self.r] if
                          i not in [self.data.Trip_origin[self.r],
                                    self.data.Trip_destination[self.r]]),
                         name='c_flb_normal')

        self.model.addConstrs((self.var.x[h, l] <= 1 for (h, l) in self.data.Hub_pairs), name='x_ub')
        self.model.addConstrs((self.var.y[i, j] <= 1 for (i, j) in self.data.Node_pairs_for_mdl_sp[self.r]), name='y_ub')

    def set_obj_follower(self):
        # set tao and gamma in obj as follower params
        self.obj_follower = gp.quicksum(
                self.data.tao_follower[h, l, self.s] * self.var.x[h, l] for (h, l) in self.data.Hub_pairs)
        self.obj_follower += gp.quicksum(
                self.data.gamma_follower[i, j, self.s] * self.var.y[i, j] for (i, j) in self.data.Node_pairs_for_mdl_sp[self.r])
        self.model.setObjective(self.obj_follower)

    def set_obj_leader(self):
        self.obj_leader = gp.quicksum(
                self.data.tao_leader[h, l, self.s] * self.var.x[h, l] for (h, l) in self.data.Hub_pairs)
        self.obj_leader += gp.quicksum(
                self.data.gamma_leader[i, j, self.s] * self.var.y[i, j] for (i, j) in self.data.Node_pairs_for_mdl_sp[self.r])
        self.model.setObjective(self.obj_leader)

    def update_objective_cons(self, param_z):
        if self.cons_conn is not None:
            # remove the constraints from last iteration
            self.model.remove(self.cons_conn)

        self.cons_conn = self.model.addConstrs(
            (self.var.x[h, l] <= param_z[h, l] for (h, l) in self.data.Hub_pairs),
            name='conn_enable')

    def set_params(self):
        # self.model.setParam('TimeLimit', 360)  # 10 minutes
        # self.model.setParam('InfUnbdInfo', 1)
        # self.model.setParam('FeasibilityTo', 1e-1)
        # self.model.setParam('Presolve', 0)
        # self.model.setParam('Heuristics', 0)
        self.model.setParam('LogToConsole', 0)
        # self.model.setParam('LPWarmStart', 2)


    def add_strong_dual(self):
        eta = self.model.getObjective().getValue()
        self.cons.strong_dual = self.model.addConstr(self.obj_follower <= eta, name='strong_dual')  # add strong duality constraint

    def delete_strong_dual(self):
        self.model.remove(self.cons.strong_dual)

    def write_lp_basis_info(self, use_lp_basis=False, VBasis=None, CBasis=None):
        if use_lp_basis:
            self.model.update()
            self.model.setParam('LPWarmStart', 2)  # enable LP warm start
            if VBasis is not None:
                self.model.setAttr('VBasis', self.model.getVars(), VBasis)
            if CBasis is not None:
                self.model.setAttr('CBasis', self.model.getConstrs(), CBasis)
        self.set_obj_follower()

    def obtain_solution_info(self, use_lp_basis=False):
        pi1 = {i: - self.model.getConstrByName(f'c_flb_normal[{i}]').pi for i in
               self.data.Node_of_trip[self.r] if
               i not in [self.data.Trip_origin[self.r],
                         self.data.Trip_destination[self.r]]}
        pi1[self.data.Trip_origin[self.r]] = - self.model.getConstrByName('c_flb_or').pi
        pi1[self.data.Trip_destination[self.r]] = - self.model.getConstrByName('c_flb_de').pi
        pi2 = {(h, l): - self.model.getConstrByName(f'conn_enable[{h},{l}]').pi for (h, l) in self.data.Hub_pairs}
        pi3 = {(h, l): - self.model.getConstrByName(f'x_ub[{h},{l}]').pi for (h, l) in self.data.Hub_pairs}
        pi4 = {(i, j): - self.model.getConstrByName(f'y_ub[{i},{j}]').pi for (i, j) in self.data.Node_pairs_for_mdl_sp[self.r]}

        if self.s == self.data.Scenarios[0] and use_lp_basis:
            VBasis = self.model.getAttr('VBasis', self.model.getVars())
            CBasis = self.model.getAttr('CBasis', self.model.getConstrs())[:-1]
        else:
            VBasis = None
            CBasis = None

        result = {
                  'x': {key: self.var.x[key].X for key in self.var.x},
                  'y': {key: self.var.y[key].X for key in self.var.y},
                  'pi1': pi1,
                  'pi2': pi2,
                  'pi3': pi3,
                  'pi4': pi4,
                  't': - self.model.getConstrByName('strong_dual').pi,
                  'obj': self.obj_leader.getValue(),
                  'VBasis': VBasis,
                  'CBasis': CBasis
                  }
        self.delete_strong_dual()
        return result

    def change_obj(self):
        self.add_strong_dual()
        self.set_obj_leader()

    def create_and_solve_by_with_revise_dual(self, s, r, param_z, use_lp_basis=False, VBasis=None, CBasis=None,
                                     VBasis_info=None, need_x=True):
        """
        create and solve using the with clause, meahwhile revise the dual variables
        :param param_z: only the varvalue of z is required
        :param ini_sol: initial solution
        :param use_lp_basis: whether use lp basis for warm start
        """
        # log_name = self.data.file_name[:-5]
        # env_name = log_name + '_sub_{}_{}'.format(s, r)
        with gp.Env() as env, gp.Model('SubModel', env=env) as model:
            # add variables
            pi1 = model.addVars(self.data.Node_of_trip[r], vtype=GRB.CONTINUOUS, name='pi_1', lb=-GRB.INFINITY)
            pi2 = model.addVars(self.data.Hub_pairs, vtype=GRB.CONTINUOUS, name='pi_2')
            pi3 = model.addVars(self.data.Hub_pairs, vtype=GRB.CONTINUOUS, name='pi_3')
            pi4 = model.addVars(self.data.Node_pairs_for_mdl_sp[r], vtype=GRB.CONTINUOUS, name='pi_4')
            nu1 = model.addVars(self.data.Hub_pairs, vtype=GRB.CONTINUOUS, name='nu_1')
            nu2 = model.addVars(self.data.Node_pairs_for_mdl_sp[r], vtype=GRB.CONTINUOUS, name='nu_2')
            t = model.addVar(vtype=GRB.CONTINUOUS, name='t')

            # set pi2 to 0 when z_{hl}=1
            for h,l in self.data.Hub_pairs:
                if param_z[h,l] > 0.5:
                    pi2[h,l].setAttr('ub', 0)

            # add constraint
            c_tau = model.addConstrs((
                pi1[h] - pi1[l] + pi2[h, l] + pi3[h, l] + self.data.tao_follower[
                    h, l, s] * t >= -self.data.tao_leader[h, l, s] for (h, l) in
                self.data.Hub_pairs),
                name='c_tau')
            c_gamma = model.addConstrs((
                pi1[i] - pi1[j] + pi4[i, j] + self.data.gamma_follower[
                    i, j, s] * t >= -self.data.gamma_leader[i, j, s] for (i, j) in
                self.data.Node_pairs_for_mdl_sp[r]),
                name='c_gamma')

            i = self.data.Trip_origin[r]
            model.addConstr((
                    gp.quicksum(
                        -nu1[i, l] for l in self.data.Hub if (i, l) in self.data.Hub_pairs) + gp.quicksum(
                nu1[l, i] for l in self.data.Hub if (l, i) in self.data.Hub_pairs) - gp.quicksum(
                nu2[i, j] for j in self.data.Node_of_trip[r] if
                (i, j) in self.data.Node_pairs_for_mdl_sp[r]) + gp.quicksum(
                nu2[j, i] for j in self.data.Node_of_trip[r] if (j, i) in self.data.Node_pairs_for_mdl_sp[r]) +
                    t == 0), name='c_nu_oringn')

            i = self.data.Trip_destination[r]
            model.addConstr((
                    gp.quicksum(
                        -nu1[i, l] for l in self.data.Hub if (i, l) in self.data.Hub_pairs) + gp.quicksum(
                nu1[l, i] for l in self.data.Hub if (l, i) in self.data.Hub_pairs) - gp.quicksum(
                nu2[i, j] for j in self.data.Node_of_trip[r] if
                (i, j) in self.data.Node_pairs_for_mdl_sp[r]) + gp.quicksum(
                nu2[j, i] for j in self.data.Node_of_trip[r] if (j, i) in self.data.Node_pairs_for_mdl_sp[r]) -
                    t == 0), name='c_nu_destination')

            model.addConstrs((
                gp.quicksum(-nu1[i, l] for l in self.data.Hub if (i, l) in self.data.Hub_pairs) + gp.quicksum(
                    nu1[l, i] for l in self.data.Hub if (l, i) in self.data.Hub_pairs) - gp.quicksum(
                    nu2[i, j] for j in self.data.Node_of_trip[r] if
                    (i, j) in self.data.Node_pairs_for_mdl_sp[r]) + gp.quicksum(
                    nu2[j, i] for j in self.data.Node_of_trip[r] if
                    (j, i) in self.data.Node_pairs_for_mdl_sp[r]) == 0 for i in
                self.data.Node_of_trip[r] if i not in [self.data.Trip_origin[r], self.data.Trip_destination[r]]),
                name='c_nu_normal')

            model.addConstrs(
                (-nu1[h, l] + t >= 0 for (h, l) in self.data.Hub_pairs), name='nu1_t')
            model.addConstrs(
                (-nu2[i, j] + t >= 0 for (i, j) in self.data.Node_pairs_for_mdl_sp[r]), name='nu2_t')

            # update objective and constraints
            obj = -pi1[self.data.Trip_origin[r]] + pi1[self.data.Trip_destination[r]] - gp.quicksum(
                param_z[h, l] * pi2[h, l] for (h, l) in self.data.Hub_pairs) - gp.quicksum(
                pi3) - gp.quicksum(pi4) - gp.quicksum(
                self.data.tao_follower[h, l, s] * nu1[h, l] for (h, l) in self.data.Hub_pairs) - gp.quicksum(
                self.data.gamma_follower[i, j, s] * nu2[i, j] for (i, j) in self.data.Node_pairs_for_mdl_sp[r])
            model.setObjective(obj, sense=GRB.MAXIMIZE)

            model.addConstrs((
                - nu1[h, l] + param_z[h, l] * t >= 0 for (h, l) in self.data.Hub_pairs),
                name='nu_z_t')

            # model.setParam('TimeLimit', 360)  # 10 minutes
            model.setParam('LogToConsole', 0)
            sol_time = time.time()
            model.optimize()
            sol_time = time.time() - sol_time
            if model.Status != GRB.OPTIMAL:
                with open(self.console_output_file, 'a') as file:
                    print('The subproblme model is not solved to optimal, s={}, r={}, StatusCode:{}'.format(s,r,model.Status), file=file)
                    tmp_z = {key: param_z[key] for key in param_z if param_z[key]>0}
                    print('Z={}'.format(tmp_z), file=file)

                current_directory = os.path.dirname(os.path.abspath(__file__))
                upper_2_dir = os.path.dirname(current_directory)
                lp_path = os.path.join(upper_2_dir, 'output', 'infeas_model.lp')
                model.write(lp_path)


            # revise dual values
            model.addConstr(obj == obj.getValue())
            max_var = model.addVar()  # maximum of all variables
            # model.addConstrs(max_var >= pi1[i] for i in self.data.Node_of_trip[r])
            # model.addConstrs(max_var >= - pi1[i] for i in self.data.Node_of_trip[r])
            model.addConstrs(max_var >= pi2[h,l] for (h,l) in self.data.Hub_pairs)
            # model.addConstrs(max_var >= pi3[h,l] for (h,l) in self.data.Hub_pairs)
            # model.addConstrs(max_var >= pi4[i,j] for (i,j) in self.data.Node_pairs_for_mdl_sp[r])
            model.addConstr(max_var >= t)

            model.setObjective(max_var, sense=GRB.MINIMIZE)
            model.optimize()


            if need_x:
                # x = {(h, l): - model.getConstrByName(f'c_tau[{h},{l}]').pi for (h, l) in self.data.Hub_pairs}
                x = {(h, l): - c_tau.pi for (h, l) in self.data.Hub_pairs}
                x = {key: (0 if value < 0.1 else value) for key,value in x.items()}
                # y = {(i, j): - model.getConstrByName(f'c_gamma[{i},{j}]').pi for (i, j) in
                #      self.data.Node_pairs_for_mdl_sp[r]}
                y = {(i, j): - c_gamma.pi for (i, j) in self.data.Node_pairs_for_mdl_sp[r]}
                y = {key: (0 if value < 0.1 else value) for key, value in y.items()}
            else:
                x = {}
                y = {}


            result = {'pi1': {key: pi1[key].X for key in pi1},
                      'pi2': {key: pi2[key].X for key in pi2},
                      'pi3': {key: pi3[key].X for key in pi3},
                      'pi4': {key: pi4[key].X for key in pi4},
                      't': t.X,
                      'x': x,
                      'y': y,
                      'obj': obj.getValue(),
                      'VBasis': VBasis,
                      'CBasis': CBasis,
                      'sol_time': sol_time}

            return {(s, r): result}


    def create_and_solve_by_with(self, s, r, z_param, use_lp_basis=False, VBasis=None, CBasis=None, need_x=True):
        with gp.Env() as env, gp.Model('SubModel', env=env) as model:
            # add variables
            pi1 = model.addVars(self.data.Node_of_trip[r], vtype=GRB.CONTINUOUS, name='pi_1', lb=-GRB.INFINITY)
            pi2 = model.addVars(self.data.Hub_pairs, vtype=GRB.CONTINUOUS, name='pi_2')
            pi3 = model.addVars(self.data.Hub_pairs, vtype=GRB.CONTINUOUS, name='pi_3')
            pi4 = model.addVars(self.data.Node_pairs_for_mdl_sp[r], vtype=GRB.CONTINUOUS, name='pi_4')
            nu1 = model.addVars(self.data.Hub_pairs, vtype=GRB.CONTINUOUS, name='nu_1')
            nu2 = model.addVars(self.data.Node_pairs_for_mdl_sp[r], vtype=GRB.CONTINUOUS, name='nu_2')
            t = model.addVar(vtype=GRB.CONTINUOUS, name='t')

            # set pi2 to 0 when z_{hl}=1
            for h,l in self.data.Hub_pairs:
                if z_param[h,l] > 0.5:
                    pi2[h,l].setAttr('ub', 0)

            # add constraint
            c_tau = model.addConstrs((
                pi1[h] - pi1[l] + pi2[h, l] + pi3[h, l] + self.data.tao_follower[
                    h, l, s] * t >= -self.data.tao_leader[h, l, s] for (h, l) in
                self.data.Hub_pairs),
                name='c_tau')
            c_gamma = model.addConstrs((
                pi1[i] - pi1[j] + pi4[i, j] + self.data.gamma_follower[
                    i, j, s] * t >= -self.data.gamma_leader[i, j, s] for (i, j) in
                self.data.Node_pairs_for_mdl_sp[r]),
                name='c_gamma')

            i = self.data.Trip_origin[r]
            model.addConstr((
                    gp.quicksum(
                        -nu1[i, l] for l in self.data.Hub if (i, l) in self.data.Hub_pairs) + gp.quicksum(
                nu1[l, i] for l in self.data.Hub if (l, i) in self.data.Hub_pairs) - gp.quicksum(
                nu2[i, j] for j in self.data.Node_of_trip[r] if
                (i, j) in self.data.Node_pairs_for_mdl_sp[r]) + gp.quicksum(
                nu2[j, i] for j in self.data.Node_of_trip[r] if (j, i) in self.data.Node_pairs_for_mdl_sp[r]) +
                    t == 0), name='c_nu_oringn')

            i = self.data.Trip_destination[r]
            model.addConstr((
                    gp.quicksum(
                        -nu1[i, l] for l in self.data.Hub if (i, l) in self.data.Hub_pairs) + gp.quicksum(
                nu1[l, i] for l in self.data.Hub if (l, i) in self.data.Hub_pairs) - gp.quicksum(
                nu2[i, j] for j in self.data.Node_of_trip[r] if
                (i, j) in self.data.Node_pairs_for_mdl_sp[r]) + gp.quicksum(
                nu2[j, i] for j in self.data.Node_of_trip[r] if (j, i) in self.data.Node_pairs_for_mdl_sp[r]) -
                    t == 0), name='c_nu_destination')

            model.addConstrs((
                gp.quicksum(-nu1[i, l] for l in self.data.Hub if (i, l) in self.data.Hub_pairs) + gp.quicksum(
                    nu1[l, i] for l in self.data.Hub if (l, i) in self.data.Hub_pairs) - gp.quicksum(
                    nu2[i, j] for j in self.data.Node_of_trip[r] if
                    (i, j) in self.data.Node_pairs_for_mdl_sp[r]) + gp.quicksum(
                    nu2[j, i] for j in self.data.Node_of_trip[r] if
                    (j, i) in self.data.Node_pairs_for_mdl_sp[r]) == 0 for i in
                self.data.Node_of_trip[r] if i not in [self.data.Trip_origin[r], self.data.Trip_destination[r]]),
                name='c_nu_normal')

            model.addConstrs(
                (-nu1[h, l] + t >= 0 for (h, l) in self.data.Hub_pairs), name='nu1_t')
            model.addConstrs(
                (-nu2[i, j] + t >= 0 for (i, j) in self.data.Node_pairs_for_mdl_sp[r]), name='nu2_t')

            # update objective and constraints
            obj = -pi1[self.data.Trip_origin[r]] + pi1[self.data.Trip_destination[r]] - gp.quicksum(
                z_param[h, l] * pi2[h, l] for (h, l) in self.data.Hub_pairs) - gp.quicksum(
                pi3) - gp.quicksum(pi4) - gp.quicksum(
                self.data.tao_follower[h, l, s] * nu1[h, l] for (h, l) in self.data.Hub_pairs) - gp.quicksum(
                self.data.gamma_follower[i, j, s] * nu2[i, j] for (i, j) in self.data.Node_pairs_for_mdl_sp[r])
            model.setObjective(obj, sense=GRB.MAXIMIZE)

            model.addConstrs((
                - nu1[h, l] + z_param[h, l] * t >= 0 for (h, l) in self.data.Hub_pairs),
                name='nu_z_t')

            # model.setParam('TimeLimit', 360)  # 10 minutes
            model.setParam('LogToConsole', 0)
            sol_time = time.time()
            model.optimize()
            sol_time = time.time() - sol_time

            if need_x:
                # x = {(h, l): - model.getConstrByName(f'c_tau[{h},{l}]').pi for (h, l) in self.data.Hub_pairs}
                x = {(h, l): - c_tau[h,l].pi for (h, l) in self.data.Hub_pairs}
                x = {key: (0 if value < 0.1 else value) for key,value in x.items()}
                # y = {(i, j): - model.getConstrByName(f'c_gamma[{i},{j}]').pi for (i, j) in
                #      self.data.Node_pairs_for_mdl_sp[r]}
                y = {(i, j): - c_gamma[i,j].pi for (i, j) in self.data.Node_pairs_for_mdl_sp[r]}
                y = {key: (0 if value < 0.1 else value) for key, value in y.items()}
            else:
                x = {}
                y = {}

            result = {'pi1': {key: pi1[key].X for key in pi1},
                      'pi2': {key: pi2[key].X for key in pi2},
                      'pi3': {key: pi3[key].X for key in pi3},
                      'pi4': {key: pi4[key].X for key in pi4},
                      't': t.X,
                      'x': x,
                      'y': y,
                      'obj': obj.getValue(),
                      'VBasis': VBasis,
                      'CBasis': CBasis,
                      'sol_time': sol_time}

            return {(s, r): result}


    def create_and_solve_by_with_old_use_primal_space(self, s, r, z_param, use_lp_basis=False, VBasis=None, CBasis=None, need_x=True):
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
                y[j, i] for j in self.data.Node_of_trip[r] if (j, i) in self.data.Node_pairs_for_mdl_sp[r]) == 1), name='c_flb_or')
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
                y[j, i] for j in self.data.Node_of_trip[r] if (j, i) in self.data.Node_pairs_for_mdl_sp[r]) == 0 for i in
                                                                  self.data.Node_of_trip[r] if
                                                                  i not in [self.data.Trip_origin[r],
                                                                            self.data.Trip_destination[r]]),
                                                                 name='c_flb_normal')
            model.addConstrs(
                (x[h, l] <= z_param[h, l] for (h, l) in self.data.Hub_pairs),
                name='conn_enable')

            model.addConstrs((x[h, l] <= 1 for (h, l) in self.data.Hub_pairs), name='x_ub')
            model.addConstrs((y[i, j] <= 1 for (i, j) in self.data.Node_pairs_for_mdl_sp[r]), name='y_ub')

            # set objective - follower
            obj_follower = gp.quicksum(
                self.data.tao_follower[h, l, s] * x[h, l] for (h, l) in self.data.Hub_pairs)
            obj_follower += gp.quicksum(
                self.data.gamma_follower[i, j, s] * y[i, j] for (i, j) in self.data.Node_pairs_for_mdl_sp[r])
            model.setObjective(obj_follower)

            # warm start
            if use_lp_basis:
                if VBasis is not None or CBasis is not None:
                    model.update()
                    model.setParam('LPWarmStart', 2)  # enable LP warm start
                    if VBasis is not None:
                        model.setAttr('VBasis', model.getVars(), VBasis)
                    if CBasis is not None:
                        model.setAttr('CBasis', model.getConstrs(), CBasis)

            # model.setParam('TimeLimit', 360)  # 10 minutes
            model.setParam('LogToConsole', 0)
            model.optimize()
            # add strong duality
            if model.Status != 2:
                with open(self.console_output_file, 'a') as file:
                    print('model is not solved to optimal, StatusCode:{}'.format(model.Status), file=file)
                model.write('nonoptimal_sub.lp')
            eta = model.getObjective().getValue()
            model.addConstr(obj_follower <= eta, name='strong_dual')  # add strong duality constraint
            # set objective -leader
            obj_leader = gp.quicksum(
                self.data.tao_leader[h, l, s] * x[h, l] for (h, l) in self.data.Hub_pairs)
            obj_leader += gp.quicksum(
                self.data.gamma_leader[i, j, s] * y[i, j] for (i, j) in self.data.Node_pairs_for_mdl_sp[r])
            model.setObjective(obj_leader)
            model.optimize()

            pi1 = {i: - model.getConstrByName(f'c_flb_normal[{i}]').pi for i in
                   self.data.Node_of_trip[r] if
                   i not in [self.data.Trip_origin[r],
                             self.data.Trip_destination[r]]}
            pi1[self.data.Trip_origin[r]] = - model.getConstrByName('c_flb_or').pi
            pi1[self.data.Trip_destination[r]] = - model.getConstrByName('c_flb_de').pi
            pi2 = {(h, l): - model.getConstrByName(f'conn_enable[{h},{l}]').pi for (h, l) in self.data.Hub_pairs}
            pi3 = {(h, l): - model.getConstrByName(f'x_ub[{h},{l}]').pi for (h, l) in self.data.Hub_pairs}
            pi4 = {(i, j): - model.getConstrByName(f'y_ub[{i},{j}]').pi for (i, j) in self.data.Node_pairs_for_mdl_sp[r]}

            if s == self.data.Scenarios[0] and use_lp_basis:
                VBasis = model.getAttr('VBasis', model.getVars())
                CBasis = model.getAttr('CBasis', model.getConstrs())[:-1]
            else:
                VBasis = None
                CBasis = None

            if need_x:
                x = {key: x[key].X for key in x if x[key].X}
                y = {key: y[key].X for key in y if y[key].X}
            else:
                x = None
                y = None

            result = {
                'x': x,
                'y': y,
                'pi1': pi1,
                'pi2': pi2,
                'pi3': pi3,
                'pi4': pi4,
                't': - model.getConstrByName('strong_dual').pi,
                'obj': obj_leader.getValue(),
                'VBasis': VBasis,
                'CBasis': CBasis
            }

            return {(s,r): result}


class VarValue:
    """
    Value of variables, which will be transferred between the master and the subproblem
    for CCG algorithm
    """

    def __init__(self):
        # master var
        self.z = {}
        # sub var
        self.pi1 = {}
        self.pi2 = {}
        self.pi3 = {}
        self.pi4 = {}
        self.t = {}

        self.pi1_all = {}
        self.pi2_all = {}
        self.pi3_all = {}
        self.pi4_all = {}
        self.t_all = {}

        self.latest_sub_obj = {}  # obj of the dual subproblem in CCG, not the primal shortest path problem


        self.x = {}
        self.y = {}

        # master vars for test
        self.theta1 = {}
        self.theta2 = {}
        self.theta3 = {}
        self.theta4 = {}
        self.delta = {}
        self.gamma = {}
        self.gamma_single = 0
        self.n_iters = 0

        self.theta1_all = {}
        self.theta2_all = {}
        self.theta3_all = {}
        self.theta4_all = {}

        self.VBasis = {}
        self.CBasis = {}

        self.best_z = {}
        self.best_x = {}
        self.best_y = {}

    def update_Master_var(self, var: MasterModel.Variable):
        self.z = {key: round(var.z[key].X) for key in var.z}
        # print('*************the value of z:*************')
        # for key in var.z:
        #     if self.z[key]:
        #         print(self.z[key])
        if var.gamma_exist:
            self.gamma = {key: var.gamma[key].X for key in var.gamma}
        else:
            if var.use_part_agg:
                self.gamma_part = {key: var.gamma_agg_part[key].X for key in var.gamma_agg_part}
            else:
                self.gamma_single = var.gamma_agg.X
        self.n_iters = var.n_iters

    def update_Master_var_callback(self, z, gamma, n_iters):
        self.z = z.copy()
        self.gamma = gamma.copy()
        self.n_iters = n_iters

    def update_sp_obj_directly_from_mp(self, var):
        """
        update sub obj for s,r in self.data.explored
        :param var:
        :return:
        """
        for key in var.gamma:
            self.latest_sub_obj.update({key: var.gamma[key].X})

    def update_sp_obj_directly_from_mp_callback(self, gamma:dict):
        """
        update sub obj for s,r in self.data.explored
        :param var:
        :return:
        """
        self.latest_sub_obj.update(gamma)


    def update_Master_var_Benders(self, z, gamma, n_iters):
        self.z = z.copy()
        self.gamma = gamma.copy()
        self.n_iters = n_iters

    def update_Master_theta(self, var: MasterModel.Variable):
        self.theta1.update(
            {key_o: {key_i: var.theta1[key_o][key_i].X for key_i in var.theta1[key_o]} for key_o in var.theta1})
        self.theta2.update(
            {key_o: {key_i: var.theta2[key_o][key_i].X for key_i in var.theta2[key_o]} for key_o in var.theta2})
        self.theta3.update(
            {key_o: {key_i: var.theta3[key_o][key_i].X for key_i in var.theta3[key_o]} for key_o in var.theta3})
        self.theta4.update(
            {key_o: {key_i: var.theta4[key_o][key_i].X for key_i in var.theta4[key_o]} for key_o in var.theta4})
        self.delta.update(
            {key_o: {key_i: var.delta[key_o][key_i].X for key_i in var.delta[key_o]} for key_o in var.delta})

        # self.n_iters = var.n_iters
        # print('update master variables')


    def update_Sub_var(self, s, r, pi1, pi2, pi3, pi4, t, obj):
        """
        update sub vars value
        :param s: scenario
        :param r: trip
        :param var: including pi1, pi2, pi3, pi4, t
        :return:
        """
        self.pi1.update({(s, r,) + (key,): pi1[key] for key in pi1})
        self.pi2.update({(s, r,) + key: pi2[key] for key in pi2})
        self.pi3.update({(s, r,) + key: pi3[key] for key in pi3})
        self.pi4.update({(s, r,) + key: pi4[key] for key in pi4})

        self.t.update({(s, r,): t})
        self.latest_sub_obj.update({(s,r,): obj})
        # print('update sub variables')

    def update_best_z(self):
        self.best_z = self.z.copy()

    def update_best_x_y(self):
        self.best_x = self.x.copy()
        self.best_y = self.y.copy()

    def update_Sub_var_primal(self, s, r, x, y, pi1, pi2, pi3, pi4, t, obj):
        """
        update sub var value, approach 2, primal formulation
        :param s:
        :param r:
        :param pi1:
        :param pi2:
        :param pi3:
        :param pi4:
        :param t:
        :param x:
        :param y:
        :return:
        """

        self.pi1.update({(s, r,) + (key,): pi1[key] for key in pi1})
        self.pi2.update({(s, r,) + key: pi2[key] for key in pi2})
        self.pi3.update({(s, r,) + key: pi3[key] for key in pi3})
        self.pi4.update({(s, r,) + key: pi4[key] for key in pi4})

        self.t.update({(s, r,): t})

        # if x is not None:
        #     self.x = {k:v for k,v in self.x.items() if not (k[0]==s and k[1]==r)}
        #     self.y = {k:v for k,v in self.y.items() if not (k[0]==s and k[1]==r)}

        if y is not None:
            self.x.update({(s, r,) + key: x[key] for key in x})
            self.y.update({(s, r,) + key: y[key] for key in y})

        self.latest_sub_obj.update({(s, r,): obj})

        # if VBasis is not None or CBasis is not None:
        #     self.VBasis.update({(r,): VBasis})
        #     self.CBasis.update({(r,): CBasis})

    def record_pi_value(self):
        self.pi1_all.update({self.n_iters: self.pi1.copy()})
        self.pi2_all.update({self.n_iters: self.pi2.copy()})
        self.pi3_all.update({self.n_iters: self.pi3.copy()})
        self.pi4_all.update({self.n_iters: self.pi4.copy()})
        self.t_all.update({self.n_iters: self.t.copy()})

    def update_x_y(self, s, r, x_sol, y_sol):
        if len(x_sol):
            self.x.update({(s,r,)+key: x_sol[key] for key in x_sol})

        if len(y_sol):
            self.y.update({(s,r,)+key: y_sol[key] for key in y_sol})


class OneStageModelFullMultiplierDepends():
    """Here we have full multipliers for x>=0, but only added when using KKT condition"""
    def __init__(self, data: Modeldata, use_KKT=True, use_sos1=True, use_bigM=True, add_strong_dual=False, add_why_no_need_M=False):
        """
        initial function
        :param data: modeldata
        :param use_KKT: use KKT or not, remark20240623, actually means when using KKT, whether add multipliers for x>=0 and y>=0
        :param use_sos1: use SOS1 constraints (True) or use bigM (False)
        :param add_strong_dual: whether including strong duality when using SOS1, valid only when use_sos1=True
        """
        self.data = data
        self.use_KKT = use_KKT
        self.use_sos1 = use_sos1
        self.use_bigM = use_bigM
        self.add_strong_dual = add_strong_dual # if self.use_sos1 else False
        self.add_why_no_need_M = add_why_no_need_M
        current_directory = os.path.dirname(os.path.abspath(__file__))
        upper_2_dir = os.path.dirname(current_directory)
        data_file = os.path.basename(data.file_name)[:-5]
        log_file_prefix = os.path.join(upper_2_dir, 'output')
        log_file_prefix = os.path.join(log_file_prefix, data_file)
        self.env = gp.Env(log_file_prefix + '_SingleLevel')
        self.n_node_BnB = 0
        tmp = time.time()
        self.model = gp.Model('OneStageModel', self.env)
        self.var = self.Variable(model=self.model, data=data, use_kkt=self.use_KKT)
        self.cons = self.Constraint()
        self.add_constraints()
        self.set_objective()
        self.time_mdl_creation = time.time() - tmp

    class Variable():
        """class of variables"""
        def __init__(self, model, data: Modeldata, use_kkt: bool):
            # vars in the master problem
            self.z = model.addVars(data.Hub_pairs, vtype=GRB.BINARY, name='z')
            self.x = model.addVars(
                gp.tuplelist((s, r, h, l) for s in data.Scenarios for r in data.Trips for (h, l) in data.Hub_pairs),
                vtype=GRB.CONTINUOUS, ub=1, name='x')
            self.y = model.addVars(
                gp.tuplelist((s, r, i, j) for s in data.Scenarios for r in data.Trips for (i, j) in data.Node_pairs_for_mdl_sp[r]),
                vtype=GRB.CONTINUOUS, ub=1, name='y')
            self.alpha = model.addVars(gp.tuplelist((s, r, i) for (s, r) in data.Sce_Trip for i in data.Node),
                                       vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name='alpha')
            self.beta = model.addVars(
                gp.tuplelist((s, r, h, l) for (s, r) in data.Sce_Trip for (h, l) in data.Hub_pairs),
                vtype=GRB.CONTINUOUS, name='beta')
            self.mu = model.addVars(
                gp.tuplelist((s, r, h, l) for (s, r) in data.Sce_Trip for (h, l) in data.Hub_pairs),
                vtype=GRB.CONTINUOUS, name='mu')
            self.zeta = model.addVars(
                gp.tuplelist((s, r, i, j) for (s, r) in data.Sce_Trip for (i, j) in data.Node_pairs_for_mdl_sp[r]),
                vtype=GRB.CONTINUOUS, name='zeta')

            if use_kkt:
                # add multipliers for x>=0 and y>=0 if use KKT condition
                self.multi_noneg_x = model.addVars(
                    gp.tuplelist((s, r, h, l) for (s, r) in data.Sce_Trip for (h, l) in data.Hub_pairs),
                    vtype=GRB.CONTINUOUS, name='multi_noneg_x')
                self.multi_noneg_y = model.addVars(
                    gp.tuplelist((s, r, i, j) for (s, r) in data.Sce_Trip for (i, j) in data.Node_pairs_for_mdl_sp[r]),
                    vtype=GRB.CONTINUOUS, name='multi_noneg_y')
                self.delta_noneg_x = model.addVars(
                    gp.tuplelist((s, r, h, l) for (s, r) in data.Sce_Trip for (h, l) in data.Hub_pairs),
                    vtype=GRB.BINARY, name='delta_noneg_x')
                self.delta_noneg_y = model.addVars(
                    gp.tuplelist((s, r, i, j) for (s, r) in data.Sce_Trip for (i, j) in data.Node_pairs_for_mdl_sp[r]),
                    vtype=GRB.BINARY, name='delta_noneg_y')


            self.delta_2 = model.addVars(
                gp.tuplelist((s, r, h, l) for (s, r) in data.Sce_Trip for (h, l) in data.Hub_pairs),
                vtype=GRB.BINARY, name='delta_2')
            self.delta_3 = model.addVars(
                gp.tuplelist((s, r, h, l) for (s, r) in data.Sce_Trip for (h, l) in data.Hub_pairs),
                vtype=GRB.BINARY, name='delta_3')
            self.delta_4 = model.addVars(
                gp.tuplelist((s, r, i, j) for (s, r) in data.Sce_Trip for (i, j) in data.Node_pairs_for_mdl_sp[r]),
                vtype=GRB.BINARY, name='delta_4')

            self.sos_aux_2 = model.addVars(
                gp.tuplelist((s, r, h, l) for (s, r) in data.Sce_Trip for (h, l) in data.Hub_pairs),
                vtype=GRB.CONTINUOUS, name='sos_aux_2')
            self.sos_aux_3 = model.addVars(
                gp.tuplelist((s, r, h, l) for (s, r) in data.Sce_Trip for (h, l) in data.Hub_pairs),
                vtype=GRB.CONTINUOUS, name='sos_aux_3')
            self.sos_aux_4 = model.addVars(
                gp.tuplelist((s, r, i, j) for (s, r) in data.Sce_Trip for (i, j) in data.Node_pairs_for_mdl_sp[r]),
                vtype=GRB.CONTINUOUS, name='sos_aux_4')

            self.aux_obj = model.addVars(
                gp.tuplelist((s, r, h, l) for (s, r) in data.Sce_Trip for (h, l) in data.Hub_pairs),
                vtype=GRB.CONTINUOUS, name='aux_obj')

    def fix_z(self, given_z: dict):
        """
        TODO: this is not correct, as we only recorded opened z! This function doesn't fix all z!
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

    class Constraint():
        """class of constraint"""

        def __init__(self):
            self.c_flb_or = {}
            self.c_flb_de = {}
            self.c_flb_normal = {}
            self.conn_enable = {}
            self.obj_relation = {}
            self.dual_tao = {}
            self.dual_gamma = {}
            self.kkt_1 = {}
            self.kkt_2 = {}
            self.kkt_3 = {}
            self.kkt_4 = {}
            self.kkt_5 = {}
            self.kkt_6 = {}
            self.kkt_7 = {}
            self.kkt_8 = {}
            self.kkt_9 = {}
            self.kkt_10 = {}

            self.sos_2 = {}
            self.sos_3 = {}
            self.sos_4 = {}
            self.sos_5 = {}
            self.sos_6 = {}

            self.kkt_aux_2_def = {}
            self.kkt_aux_3_def = {}
            self.kkt_aux_4_def = {}


    def add_constraints(self):
        # leader flow balance constraint
        self.cons.leader_flow_balance = self.model.addConstrs((gp.quicksum(
            self.var.z[h, l] for l in self.data.Hub if (h, l) in self.data.Hub_pairs) == gp.quicksum(
            self.var.z[l, h] for l in self.data.Hub if (l, h) in self.data.Hub_pairs) for h in
                                                               self.data.Hub), name='leader_flow_balance')
        # primal feasibility
        for (s, r) in self.data.Sce_Trip:
            # flow balance
            i = self.data.Trip_origin[r]
            self.cons.c_flb_or[s, r] = self.model.addConstr((gp.quicksum(
                self.var.x[s, r, i, h] - self.var.x[s, r, h, i] for h in self.data.Hub if
                (i, h) in self.data.Hub_pairs) + gp.quicksum(
                self.var.y[s, r, i, j] for j in self.data.Node_of_trip[r] if
                (i, j) in self.data.Node_pairs_for_mdl_sp[r]) - gp.quicksum(self.var.y[s, r, j, i] for j in self.data.Node_of_trip[r] if
                (j, i) in self.data.Node_pairs_for_mdl_sp[r]) == 1), name='c_flb_or[{},{}]'.format(s, r))
            i = self.data.Trip_destination[r]
            self.cons.c_flb_de[s, r] = self.model.addConstr((gp.quicksum(
                self.var.x[s, r, i, h] - self.var.x[s, r, h, i] for h in self.data.Hub if
                (i, h) in self.data.Hub_pairs) + gp.quicksum(
                self.var.y[s, r, i, j] for j in self.data.Node_of_trip[r] if
                (i, j) in self.data.Node_pairs_for_mdl_sp[r]) - gp.quicksum(self.var.y[s, r, j, i] for j in self.data.Node_of_trip[r] if
                (j, i) in self.data.Node_pairs_for_mdl_sp[r]) == -1),
                                                            name='c_flb_de[{},{}]'.format(s, r))
            self.cons.c_flb_normal[s, r] = self.model.addConstrs((gp.quicksum(
                self.var.x[s, r, i, h] - self.var.x[s, r, h, i] for h in self.data.Hub if
                (i, h) in self.data.Hub_pairs) + gp.quicksum(
                self.var.y[s, r, i, j] for j in self.data.Node_of_trip[r] if
                (i, j) in self.data.Node_pairs_for_mdl_sp[r]) - gp.quicksum(self.var.y[s, r, j, i] for j in self.data.Node_of_trip[r] if
                (j, i) in self.data.Node_pairs_for_mdl_sp[r]) == 0 for i in
                                                                  self.data.Node_of_trip[r] if
                                                                  i not in [self.data.Trip_origin[r],
                                                                            self.data.Trip_destination[r]]),
                                                                 name='c_flb_normal[{},{}]'.format(s, r))
            # Connection enable
            self.cons.conn_enable[s, r] = self.model.addConstrs(
                (self.var.x[s, r, h, l] <= self.var.z[h, l] for (h, l) in self.data.Hub_pairs),
                name='conn_enable[{},{}]'.format(s, r))

            # Stationarity
            if self.use_KKT:
                self.cons.dual_tao[s, r] = self.model.addConstrs(
                    (self.var.alpha[s, r, h] - self.var.alpha[s, r, l] + self.var.beta[s, r, h, l] + self.var.mu[
                        s, r, h, l] - self.var.multi_noneg_x[s, r, h, l] == -self.data.tao_follower[
                        h, l, s] for (h, l) in self.data.Hub_pairs), name='dual_tao[{},{}]'.format(s, r))
                self.cons.dual_gamma[s, r] = self.model.addConstrs(
                    (self.var.alpha[s, r, i] - self.var.alpha[s, r, j] + self.var.zeta[
                         s, r, i, j] -
                     self.var.multi_noneg_y[s, r, i, j] == - self.data.gamma_follower[i, j, s] for (i, j) in
                     self.data.Node_pairs_for_mdl_sp[r]), name='dual_gamma[{},{}]'.format(s, r))
            else:
                self.cons.dual_tao[s, r] = self.model.addConstrs(
                    (self.var.alpha[s, r, h] - self.var.alpha[s, r, l] + self.var.beta[s, r, h, l] + self.var.mu[
                        s, r, h, l] >= -self.data.tao_follower[
                        h, l, s] for (h, l) in self.data.Hub_pairs), name='dual_tao[{},{}]'.format(s, r))
                self.cons.dual_gamma[s, r] = self.model.addConstrs(
                    (self.var.alpha[s, r, i] - self.var.alpha[s, r, j] + self.var.zeta[
                         s, r, i, j] >= -
                     self.data.gamma_follower[i, j, s] for (i, j) in
                     self.data.Node_pairs_for_mdl_sp[r]), name='dual_gamma[{},{}]'.format(s, r))

        self.cons.aux_obj_1 = self.model.addConstrs(
            (self.var.aux_obj[s, r, h, l] <= self.data.bigM * self.var.z[h, l] for (s, r) in self.data.Sce_Trip for
             (h, l) in self.data.Hub_pairs), name='aux_obj_1')
        self.cons.aux_obj_2 = self.model.addConstrs(
            (self.var.aux_obj[s, r, h, l] <= self.var.beta[s, r, h, l] for (s, r) in self.data.Sce_Trip
             for
             (h, l) in self.data.Hub_pairs), name='aux_obj_2')
        self.cons.aux_obj_3 = self.model.addConstrs(
            (self.var.aux_obj[s, r, h, l] >= self.var.beta[s, r, h, l] - self.data.bigM * (1 - self.var.z[h, l]) for (s, r) in
             self.data.Sce_Trip for
             (h, l) in self.data.Hub_pairs), name='aux_obj_3')




        # complementary slackness
        if self.use_bigM:
            self._add_cons_kkt_bigM()
        if self.use_sos1:
            self._add_cons_kkt_sos1()
        if self.add_strong_dual:
            self._add_cons_strong_duality()  # calculate the value of follower obj, but add constr obj_dual>=obj_primal only when not use KKT
        if self.add_why_no_need_M:
            self._add_cons_why_no_need_M()

    def _add_cons_kkt_bigM(self):
        for (s, r) in self.data.Sce_Trip:
            self.cons.kkt_1[s, r] = self.model.addConstrs(
                (self.var.beta[s, r, h, l] <= self.data.bigM * self.var.delta_2[s, r, h, l] for (h, l) in
                 self.data.Hub_pairs), name='kkt_1_{}_{}'.format(s,r))
            self.cons.kkt_2[s,r] = self.model.addConstrs(
                (-(self.var.x[s, r, h, l] - self.var.z[h, l]) <= self.data.bigM * (1 - self.var.delta_2[s, r, h, l]) for
                 (h, l) in self.data.Hub_pairs), name='kkt_2_{}_{}'.format(s,r))
            self.cons.kkt_3[s,r] = self.model.addConstrs(
                (self.var.mu[s, r, h, l] <= self.data.bigM * self.var.delta_3[s, r, h, l] for (h, l) in
                 self.data.Hub_pairs), name='kkt_3_{}_{}'.format(s,r))
            self.cons.kkt_4[s,r] = self.model.addConstrs((-(
                    self.var.x[s, r, h, l] - 1) <= self.data.bigM * (
                                                             1 - self.var.delta_3[s, r, h, l]) for (h, l) in
                                                     self.data.Hub_pairs), name='kkt_4_{}_{}'.format(s,r))
            self.cons.kkt_5[s,r] = self.model.addConstrs(
                (self.var.zeta[s, r, i, j] <= self.data.bigM * self.var.delta_4[s, r, i, j] for (i, j) in
                 self.data.Node_pairs_for_mdl_sp[r]), name='kkt_5_{}_{}'.format(s,r))
            self.cons.kkt_6[s,r] = self.model.addConstrs((-(
                    self.var.y[s, r, i, j] - 1) <= self.data.bigM * (
                                                             1 - self.var.delta_4[s, r, i, j]) for (i, j) in
                                                     self.data.Node_pairs_for_mdl_sp[r]), name='kkt_6_{}_{}'.format(s,r))
            if self.use_KKT:
                self.cons.kkt_7[s,r] = self.model.addConstrs(
                    (self.var.multi_noneg_x[s, r, h, l] <= self.data.bigM * self.var.delta_noneg_x[s, r, h, l] for (h, l) in
                     self.data.Hub_pairs), name='kkt_7_{}_{}'.format(s,r))
                self.cons.kkt_8[s,r] = self.model.addConstrs((
                    self.var.x[s, r, h, l] <= self.data.bigM * (
                            1 - self.var.delta_noneg_x[s, r, h, l]) for (h, l) in
                    self.data.Hub_pairs), name='kkt_8_{}_{}'.format(s,r))
                self.cons.kkt_9[s,r] = self.model.addConstrs(
                    (self.var.multi_noneg_y[s, r, i, j] <= self.data.bigM * self.var.delta_noneg_y[s, r, i, j] for (i, j) in
                     self.data.Node_pairs_for_mdl_sp[r]), name='kkt_9_{}_{}'.format(s,r))
                self.cons.kkt_10[s,r] = self.model.addConstrs((
                    self.var.y[s, r, i, j] <= self.data.bigM * (
                            1 - self.var.delta_noneg_y[s, r, i, j]) for (i, j) in
                    self.data.Node_pairs_for_mdl_sp[r]), name='kkt_10_{}_{}'.format(s,r))

    def _add_cons_kkt_sos1(self):
        for (s, r) in self.data.Sce_Trip:
            self.cons.kkt_aux_2_def[s,r] = self.model.addConstrs(
                (self.var.sos_aux_2[s, r, h, l] == - (self.var.x[s, r, h, l] - self.var.z[h, l]) for (h, l) in
                 self.data.Hub_pairs), name='kkt_aux_2_def_{}_{}'.format(s,r))
            self.cons.kkt_aux_3_def[s,r] = self.model.addConstrs(
                (self.var.sos_aux_3[s, r, h, l] == - (self.var.x[s, r, h, l] - 1) for (h, l) in
                 self.data.Hub_pairs), name='kkt_aux_3_def_{}_{}'.format(s,r))
            self.cons.kkt_aux_4_def[s,r] = self.model.addConstrs(
                (self.var.sos_aux_4[s, r, i, j] == - (self.var.y[s, r, i, j] - 1) for (i, j) in
                 self.data.Node_pairs_for_mdl_sp[r]), name='kkt_aux_4_def_{}_{}'.format(s,r))

            # index start from 2, to be compatible with the variable index delta
            for (h, l) in self.data.Hub_pairs:
                self.cons.sos_2[s, r, h, l] = self.model.addSOS(GRB.SOS_TYPE1, [self.var.beta[s, r, h, l],
                                                                                self.var.sos_aux_2[s, r, h, l]], [1, 2])
                self.cons.sos_3[s, r, h, l] = self.model.addSOS(GRB.SOS_TYPE1, [self.var.mu[s, r, h, l],
                                                                                self.var.sos_aux_3[s, r, h, l]], [1, 2])
            for (i, j) in self.data.Node_pairs_for_mdl_sp[r]:
                self.cons.sos_4[s, r, i, j] = self.model.addSOS(GRB.SOS_TYPE1, [self.var.zeta[s, r, i, j],
                                                                                self.var.sos_aux_4[s, r, i, j]], [1, 2])
            if self.use_KKT:
                for (h, l) in self.data.Hub_pairs:
                    self.cons.sos_5[s, r, h, l] = self.model.addSOS(GRB.SOS_TYPE1, [self.var.multi_noneg_x[s, r, h, l],
                                                                                    self.var.x[s, r, h, l]], [1, 2])
                for (i, j) in self.data.Node_pairs_for_mdl_sp[r]:
                    self.cons.sos_5[s, r, i, j] = self.model.addSOS(GRB.SOS_TYPE1, [self.var.multi_noneg_y[s, r, i, j],
                                                                                    self.var.y[s, r, i, j]], [1, 2])

    def check_KKT(self):
        print('>>>>>>>>>>>>check KKT correctness')
        for (s, r) in self.data.Sce_Trip:
            print('----------------c:beta----------------')
            for (h, l) in self.data.Hub_pairs:
                r1 = self.var.beta[s, r, h, l]
                r2 = self.var.x[s, r, h, l] - self.var.z[h, l]
                if r1.X * r2.getValue() > 1e-8:
                    print('s={}, r={}, h={}, l={},\t multiplier={:.6f},  LinExp={:.6f},  production={:.6f}'.format(s,r,h,l,r1.X,r2.getValue(),r1.X * r2.getValue()))

            print('----------------c:mu----------------')
            for (h, l) in self.data.Hub_pairs:
                r1 = self.var.mu[s, r, h, l]
                r2 = self.var.x[s, r, h, l] - 1
                if r1.X * r2.getValue():
                    print('s={}, r={}, h={}, l={},\t multiplier={:.6f},  LinExp={:.6f},  production={:.6f}'.format(s, r, h, l, r1.X,
                                                                                                         r2.getValue(),
                                                                                                         r1.X * r2.getValue()))

            print('----------------c:zeta----------------')
            for (i, j) in self.data.Node_pairs_for_mdl_sp[r]:
                r1 = self.var.zeta[s, r, i, j]
                r2 = self.var.y[s, r, i, j] - 1
                if r1.X * r2.getValue():
                    print('s={}, r={}, h={}, l={},\t multiplier={:.6f},  LinExp={:.6f},  production={:.6f}'.format(s, r, i, j, r1.X,
                                                                                                         r2.getValue(),
                                                                                                         r1.X * r2.getValue()))

    def fix_follower_var(self, varValue):
        # take optimal solution of mdl into this model to verify the feasibility
        print('fix the vars x,y,z... ...')
        for (s, r) in self.data.Sce_Trip:
            for (h, l) in self.data.Hub_pairs:
                self.var.x[s, r, h, l].setAttr('lb', varValue.x[s, r, h, l])
                self.var.x[s, r, h, l].setAttr('ub', varValue.x[s, r, h, l])
            for (i, j) in self.data.Node_pairs_for_mdl_sp[r]:
                self.var.y[s, r, i, j].setAttr('lb', varValue.y[s, r, i, j])
                self.var.y[s, r, i, j].setAttr('ub', varValue.y[s, r, i, j])

        for (h,l) in self.data.Hub_pairs:
            self.var.z[h,l].setAttr('lb', varValue.z[h,l])
            self.var.z[h, l].setAttr('ub', varValue.z[h, l])

    def _add_cons_strong_duality(self):
        self.obj_dual = {}
        self.obj_primal = {}
        for (s, r) in self.data.Sce_Trip:
            self.obj_dual[s,r] = -self.var.alpha[s, r, self.data.Trip_origin[r]] + self.var.alpha[
                s, r, self.data.Trip_destination[r]] - gp.quicksum(
                self.var.aux_obj[s, r, h, l] for (h, l) in self.data.Hub_pairs) - gp.quicksum(
                self.var.mu[s, r, h, l] for (h, l) in self.data.Hub_pairs) - gp.quicksum(
                self.var.mu[s, r, h, l] for (h, l) in self.data.Hub_pairs) - gp.quicksum(
                self.var.zeta[s, r, i, j] for (i, j) in self.data.Node_pairs_for_mdl_sp[r])

            self.obj_primal[s, r] = gp.quicksum(
                self.data.tao_follower[h, l, s] * self.var.x[s, r, h, l] for (h, l) in
                self.data.Hub_pairs) + gp.quicksum(
                self.data.gamma_follower[i, j, s] * self.var.y[s, r, i, j] for (i, j) in self.data.Node_pairs_for_mdl_sp[r])
            # print('>>>>>>check strong duality correctness\nobj_primal is: {}, obj_dual is: {}'.format(self.obj_primal[s,r].getValue(), self.obj_dual[s,r].getValue()))
        if self.add_strong_dual:
            # if use strong duality (self.use_KKT=False) or use SOS1+strong duality (self.add_strong_dual=True)
            self.model.addConstrs((self.obj_dual[s, r] >= self.obj_primal[s, r] for (s, r) in self.data.Sce_Trip),
                                 name='strong_duality')

    def _add_cons_why_no_need_M(self):
        """
        add the cut from the paper "why there is no need to use big M"
        Set z_{h,l}=0. Don't need to solve an extra LP because z_{h,l}=0 is always feasible in LP (7).
        :return:
        """
        self.obj_dual = {}
        self.obj_primal = {}
        for (s, r) in self.data.Sce_Trip:
            self.obj_dual[s,r] = -self.var.alpha[s, r, self.data.Trip_origin[r]] + self.var.alpha[
                s, r, self.data.Trip_destination[r]] - gp.quicksum(
                self.var.mu[s, r, h, l] for (h, l) in self.data.Hub_pairs) - gp.quicksum(
                self.var.mu[s, r, h, l] for (h, l) in self.data.Hub_pairs) - gp.quicksum(
                self.var.zeta[s, r, i, j] for (i, j) in self.data.Node_pairs_for_mdl_sp[r])

            self.obj_primal[s, r] = gp.quicksum(
                self.data.tao_follower[h, l, s] * self.var.x[s, r, h, l] for (h, l) in
                self.data.Hub_pairs) + gp.quicksum(
                self.data.gamma_follower[i, j, s] * self.var.y[s, r, i, j] for (i, j) in self.data.Node_pairs_for_mdl_sp[r])
            # print('>>>>>>check strong duality correctness\nobj_primal is: {}, obj_dual is: {}'.format(self.obj_primal[s,r].getValue(), self.obj_dual[s,r].getValue()))
        if self.add_why_no_need_M:
            # if use strong duality (self.use_KKT=False) or use SOS1+strong duality (self.add_strong_dual=True)
            self.model.addConstrs((self.obj_dual[s, r] >= self.obj_primal[s, r] for (s, r) in self.data.Sce_Trip),
                                 name='why_no_need_M')

    def present_obj_follower(self):
        print('>>>>> present the obj of follower')
        for (s, r) in self.data.Sce_Trip:
            print('s={}, r={}, obj_primal={}, obj_dual={}'.format(s, r, self.obj_primal[s, r].getValue(),
                                                                  self.obj_dual[s, r].getValue()))


    def set_objective(self):

        obj = gp.quicksum(self.data.Beta_hl[h, l] * self.var.z[h, l] for (h, l) in self.data.Hub_pairs)
        for (s, r) in self.data.Sce_Trip:
            obj += self.data.Scenarios_prob[s] * self.data.Trip_p_amount[s,r] * gp.quicksum(
                self.data.tao_leader[h, l, s] * self.var.x[s, r, h, l] for (h, l) in self.data.Hub_pairs)
            obj += self.data.Scenarios_prob[s] * self.data.Trip_p_amount[s,r] * gp.quicksum(
                self.data.gamma_leader[i, j, s] * self.var.y[s, r, i, j] for (i, j) in self.data.Node_pairs_for_mdl_sp[r])

        self.model.setObjective(obj)

    def solve_model(self):
        parser = get_parser()
        args = parser.parse_args()
        self.model.setParam('TimeLimit', args.Single_Level_timelimit)
        # self.model.setParam('FeasibilityTo', 1e-1)
        # self.model.setParam('Presolve', 0)
        # self.model.setParam('Heuristics', 0)
        self.model.setParam('LogToConsole', 0)
        # self.model.setParam('intfeastol', 1e-2)
        tmp = time.time()
        self.model.optimize()
        self.solution_time = time.time() - tmp
        # print('*******************One stage model is solved, obj = {}*******************'.format(self.model.getObjective().getValue()))
        if self.model.Status == 3:
            self.model.computeIIS()
            self.model.write('oneStage.ilp')
            exit()
        self.n_node_BnB = self.model.NodeCount

    def update_result_record(self, result, read_data_time):
        """
        # update result record
        :param result: the result records to be updated
        :param read_data_time: time for reading modeldata
        :return: no return
        """
        result['file_name'].append(os.path.basename(self.data.file_name))
        result['node-hub-trip-passenger'].append(re.findall(r'\d+', self.data.file_name)[:4])
        result['arc elim'].append(self.data.arc_elimination)
        result['Delta type'].append(self.data.Delta_type)
        result['use kkt'].append(self.use_KKT)
        result['use sos1'].append(self.use_sos1)
        result['use bigM'].append(self.use_bigM)
        result['use strong duality'].append(self.add_strong_dual)
        result['use why no need'].append(self.add_why_no_need_M)
        result['time read data'].append(read_data_time)
        result['time model creation'].append(self.time_mdl_creation)
        result['solution time'].append(self.solution_time)
        result['n_node_BnB'].append(self.n_node_BnB)

        if self.model.SolCount > 0:
            result['obj'].append(self.model.getObjective().getValue())
            result['gap'].append(self.model.MIPGap)

        else:
            if self.model.Status == 9:
                result['obj'].append('TimeLimitReached')
                result['gap'].append('unknown')
            else:
                result['obj'].append('Infeasible')
                result['gap'].append('unknown')


        return result


    def present_solutions(self):
        print('print the solutions: z')
        for key in self.var.z:
            if self.var.z[key].X > 1e-4:
                print(key, self.var.z[key].X)
        print('print the solutions: x')
        for key in self.var.x:
            if self.var.x[key].X > 1e-4:
                print(key, self.var.x[key].X)
        print('print the solutions: y')
        for key in self.var.y:
            if self.var.y[key].X > 1e-4:
                print(key, self.var.y[key].X)



    def collect_out_sample_solution_info(self, solution_type):
        # out of sample cost
        obj_out_sample_info = {'Scenario': self.data.Scenarios,
                               'solution type':solution_type,
                               'Facility cost': sum(
            self.data.Beta_hl[h, l] * self.var.z[h, l].X for (h, l) in self.data.Hub_pairs),
                               'Transport cost (leader)': sum(self.data.Scenarios_prob[s]*sum(
                self.data.Trip_p_amount[s,r] * (sum(
                    self.data.tao_leader[h, l, s] * self.var.x[s, r, h, l].X for (h, l) in
                    self.data.Hub_pairs) + sum(
                    self.data.gamma_leader[i, j, s] * self.var.y[s, r, i, j].X for (i, j) in self.data.Node_pairs_for_mdl_sp[r])) for r
                in self.data.Trips) for s in
            self.data.Scenarios),
                               'Transport cost (follower)': sum(self.data.Scenarios_prob[s] * sum(
                                       self.data.Trip_p_amount[s,r] * (sum(
                                           self.data.tao_follower[h, l, s] * self.var.x[s, r, h, l].X for (h, l) in
                                           self.data.Hub_pairs) + sum(
                                           self.data.gamma_follower[i, j, s] * self.var.y[s, r, i, j].X for (i, j) in
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


class OneLayerModel():
    """dr and fr are identical, one layer model"""
    def __init__(self, data: Modeldata, use_KKT=False, use_sos1=False, add_strong_dual=False):
        """
        initial function
        :param data: modeldata
        :param use_KKT: use KKT or not
        :param use_sos1: use SOS1 constraints or not
        :param add_strong_dual: whether including strong duality when using SOS1, valid only when use_sos1=True
        """
        self.data = data
        self.use_KKT = use_KKT
        self.use_sos1 = use_sos1
        self.add_strong_dual = add_strong_dual if self.use_sos1 else False
        log_name = data.file_name[:-5]
        tmp = time.time()
        current_directory = os.path.dirname(os.path.abspath(__file__))
        upper_2_dir = os.path.dirname(current_directory)
        data_file = os.path.basename(data.file_name)[:-5]
        log_file_prefix = os.path.join(upper_2_dir, 'output')
        log_file_prefix = os.path.join(log_file_prefix, data_file)
        self.env = gp.Env(log_file_prefix + '_OneLayer')
        self.model = gp.Model('OneLayerModel', self.env)
        self.var = self.Variable(model=self.model, data=data, use_kkt=self.use_KKT)
        self.cons = self.Constraint()
        self.add_constraints(use_KKT=self.use_KKT, use_sos1=self.use_sos1, add_strong_dual=self.add_strong_dual)
        self.set_objective()
        self.time_mdl_creation = time.time() - tmp

    class Variable():
        """class of variables"""
        def __init__(self, model, data: Modeldata, use_kkt: bool):
            # vars in the master problem
            self.z = model.addVars(data.Hub_pairs, vtype=GRB.BINARY, name='z')
            self.x = model.addVars(
                gp.tuplelist((s, r, h, l) for s in data.Scenarios for r in data.Trips for (h, l) in data.Hub_pairs),
                vtype=GRB.CONTINUOUS, ub=1, name='x')
            self.y = model.addVars(
                gp.tuplelist((s, r, i, j) for s in data.Scenarios for r in data.Trips for (i, j) in data.Node_pairs_for_mdl_sp[r]),
                vtype=GRB.CONTINUOUS, ub=1, name='y')


    def fix_z(self, open_idx):
        for key in self.data.Hub_pairs:
            if key in open_idx:
                self.var.z[key].setAttr('lb', 1)
            else:
                self.var.z[key].setAttr('ub', 0)

    class Constraint():
        """class of constraint"""

        def __init__(self):
            self.c_flb_or = {}
            self.c_flb_de = {}
            self.c_flb_normal = {}
            self.conn_enable = {}
            self.obj_relation = {}
            self.dual_tao = {}
            self.dual_gamma = {}
            self.kkt_1 = {}
            self.kkt_2 = {}
            self.kkt_3 = {}
            self.kkt_4 = {}
            self.kkt_5 = {}
            self.kkt_6 = {}
            self.kkt_7 = {}
            self.kkt_8 = {}
            self.kkt_9 = {}
            self.kkt_10 = {}

            self.sos_2 = {}
            self.sos_3 = {}
            self.sos_4 = {}
            self.sos_5 = {}
            self.sos_6 = {}

            self.kkt_aux_2_def = {}
            self.kkt_aux_3_def = {}
            self.kkt_aux_4_def = {}


    def add_constraints(self, use_KKT:bool, use_sos1: bool, add_strong_dual:bool):
        # leader flow balance constraint
        self.cons.leader_flow_balance = self.model.addConstrs((gp.quicksum(
            self.var.z[h, l] for l in self.data.Hub if (h, l) in self.data.Hub_pairs) == gp.quicksum(
            self.var.z[l, h] for l in self.data.Hub if (l, h) in self.data.Hub_pairs) for h in
                                                               self.data.Hub), name='leader_flow_balance')
        # primal feasibility
        for (s, r) in self.data.Sce_Trip:
            # flow balance
            i = self.data.Trip_origin[r]
            self.cons.c_flb_or[s, r] = self.model.addConstr((gp.quicksum(
                self.var.x[s, r, i, h] - self.var.x[s, r, h, i] for h in self.data.Hub if
                (i, h) in self.data.Hub_pairs) + gp.quicksum(
                self.var.y[s, r, i, j] for j in self.data.Node_of_trip[r] if
                (i, j) in self.data.Node_pairs_for_mdl_sp[r]) - gp.quicksum(
                self.var.y[s, r, j, i] for j in self.data.Node_of_trip[r] if
                (j, i) in self.data.Node_pairs_for_mdl_sp[r]) == 1), name='c_flb_or[{},{}]'.format(s, r))
            i = self.data.Trip_destination[r]
            self.cons.c_flb_de[s, r] = self.model.addConstr((gp.quicksum(
                self.var.x[s, r, i, h] - self.var.x[s, r, h, i] for h in self.data.Hub if
                (i, h) in self.data.Hub_pairs) + gp.quicksum(
                self.var.y[s, r, i, j] for j in self.data.Node_of_trip[r] if
                (i, j) in self.data.Node_pairs_for_mdl_sp[r]) - gp.quicksum(
                self.var.y[s, r, j, i] for j in self.data.Node_of_trip[r] if
                (j, i) in self.data.Node_pairs_for_mdl_sp[r]) == -1),
                                                            name='c_flb_de[{},{}]'.format(s, r))
            self.cons.c_flb_normal[s, r] = self.model.addConstrs((gp.quicksum(
                self.var.x[s, r, i, h] - self.var.x[s, r, h, i] for h in self.data.Hub if
                (i, h) in self.data.Hub_pairs) + gp.quicksum(
                self.var.y[s, r, i, j] for j in self.data.Node_of_trip[r] if
                (i, j) in self.data.Node_pairs_for_mdl_sp[r]) - gp.quicksum(
                self.var.y[s, r, j, i] for j in self.data.Node_of_trip[r] if
                (j, i) in self.data.Node_pairs_for_mdl_sp[r]) == 0 for i in
                                                                  self.data.Node_of_trip[r] if
                                                                  i not in [self.data.Trip_origin[r],
                                                                            self.data.Trip_destination[r]]),
                                                                 name='c_flb_normal[{},{}]'.format(s, r))
            # Connection enable
            self.cons.conn_enable[s, r] = self.model.addConstrs(
                (self.var.x[s, r, h, l] <= self.var.z[h, l] for (h, l) in self.data.Hub_pairs),
                name='conn_enable[{},{}]'.format(s, r))

    def set_objective(self):

        obj = gp.quicksum(self.data.Beta_hl[h, l] * self.var.z[h, l] for (h, l) in self.data.Hub_pairs)
        for (s, r) in self.data.Sce_Trip:
            obj += self.data.Scenarios_prob[s] * self.data.Trip_p_amount[s,r] * gp.quicksum(
                self.data.tao_leader[h, l, s] * self.var.x[s, r, h, l] for (h, l) in self.data.Hub_pairs)
            obj += self.data.Scenarios_prob[s] * self.data.Trip_p_amount[s,r] * gp.quicksum(
                self.data.gamma_leader[i, j, s] * self.var.y[s, r, i, j] for (i, j) in self.data.Node_pairs_for_mdl_sp[r])

        self.model.setObjective(obj)

    def solve_model(self):
        self.model.setParam('TimeLimit', 3600)  # 10 minutes
        # self.model.setParam('FeasibilityTo', 1e-1)
        # self.model.setParam('Presolve', 0)
        # self.model.setParam('Heuristics', 0)
        self.model.setParam('LogToConsole', 1)
        # self.model.setParam('intfeastol', 1e-2)
        tmp = time.time()
        self.model.optimize()
        self.solution_time = time.time() - tmp
        # print('*******************One stage model is solved, obj = {}*******************'.format(self.model.getObjective().getValue()))

    def calculate_sub_obj(self):
        def cal_obj(s,r):
            obj_leader = sum(
                self.data.tao_leader[h, l, s] * self.var.x[s, r, h, l].X for (h, l) in self.data.Hub_pairs)
            obj_leader += sum(
                self.data.gamma_leader[i, j, s] * self.var.y[s, r, i, j].X for (i, j) in self.data.Node_pairs_for_mdl_sp[r])
            return obj_leader
        self.leader_obj = {(s,r): cal_obj(s,r) for (s,r) in self.data.Sce_Trip}



    def update_result_record(self, result, read_data_time):
        """
        # update result record
        :param result: the result records to be updated
        :param read_data_time: time for reading modeldata
        :return: no return
        """
        result['node-hub-trip-passenger'].append(re.findall(r'\d+', self.data.file_name)[:4])
        result['arc elim'].append(self.data.arc_elimination)
        result['Delta type'].append(self.data.Delta_type)
        result['use kkt'].append(self.use_KKT)
        result['use sos1'].append(self.use_sos1)
        result['use strong duality'].append(self.add_strong_dual)
        result['time read data'].append(read_data_time)
        result['time model creation'].append(self.time_mdl_creation)
        result['solution time'].append(self.solution_time)

        return result

    def update_result_record_ccg_form(self, result, read_data_time, tag):
        result['file_name'].append(os.path.basename(self.data.file_name)[:-4])
        result['node-hub-trip-passenger'].append(re.findall(r'\d+', self.data.file_name)[:4])
        result['tag'].append(tag)
        result['agg_cut'].append(True)
        result['arc elim'].append(True)
        result['Delta type'].append('MST')
        result['parallel method'].append('Process')
        result['lp warm start'].append(False)
        result['solution approach'].append('Process')
        result['primal also dual'].append(False)
        result['time read data'].append(read_data_time)

        result['time model creation'].append(0)
        result['solution time'].append(0)
        result['total solution time'].append(0)
        result['solution time MP'].append(0)
        result['total solution time MP'].append(0)
        result['solution time SP'].append(0)
        result['total solution time SP'].append(0)

        result['time retrieve MP info'].append(0)
        result['total time retrieve MP info'].append(0)
        result['time retrieve SP info'].append(0)
        result['total time retrieve SP info'].append(0)
        result['time update MP'].append(0)
        result['total time update MP'].append(0)
        result['time update SP'].append(0)
        result['total time update SP'].append(0)

        result['LB record'].append(self.model.ObjBound)
        result['UB record'].append(self.model.getObjective().getValue())
        result['final_lb'].append(self.model.ObjBound)
        result['final_ub'].append(self.model.getObjective().getValue())
        result['gap'].append(self.model.MIPGap)
        result['pure solve time'].append(self.solution_time)
        result['n iters'].append(0)
        result['n_cut_each_iter'].append(0)
        result['n_node_BnB'].append(0)
        result['unexplored_sub_prob'].append(0)
        result['MILPGap'].append(None)

        if 'time preprocess' in result.keys():
            result['time preprocess'].append(0)
            result['min time preprocess'].append(0)
            result['max time preprocess'].append(0)

            result['min iter preprocess'].append(0)
            result['max iter preprocess'].append(0)
            result['ave iter preprocess'].append(0)

        return result


    def present_solutions(self):
        print('print the solutions: z')
        for key in self.var.z:
            if self.var.z[key].X > 1e-4:
                print(key, self.var.z[key].X)
        print('print the solutions: x')
        for key in self.var.x:
            if self.var.x[key].X > 1e-4:
                print(key, self.var.x[key].X)
        print('print the solutions: y')
        for key in self.var.y:
            if self.var.y[key].X > 1e-4:
                print(key, self.var.y[key].X)



    def collect_solution_info(self, solution_type):
        """
        collect the solution information
        @:param solution_type: 'determ', 'stochastic', 'stochastic_fixed'
        :return: the information of leader and follower decisions
        """
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
                     'follower travel time': [],
                     'follower travel distance': [],
                     'cost per mile': [],
                     }
        self.varValue = VarValue()
        self.varValue.x = {key: round(self.var.x[key].X) for key in self.var.x if self.var.x[key].X > 0.1}
        self.varValue.y = {key: round(self.var.y[key].X) for key in self.var.y if self.var.y[key].X > 0.1}
        self.varValue.z = {key: round(self.var.z[key].X) for key in self.var.z}

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
        trip_info['leader cost'] = [self.data.Trip_p_amount[s,r] * np.sum([
            self.data.tao_leader[h, l, s] * self.varValue.x.get((s, r, h, l),0) for (h, l) in self.data.Hub_pairs]) +
                                    self.data.Trip_p_amount[s,r] * np.sum([
            self.data.gamma_leader[i, j, s] * self.varValue.y.get((s, r, i, j),0) for (i, j) in self.data.Node_pairs_for_mdl_sp[r]]) for r in
                                    self.data.Trips for s in self.data.Scenarios]
        trip_info['follower travel time'] = [np.sum(
            [self.data.Travel_time[i, j, s] * self.varValue.y.get((s, r, i, j), 0) for (i, j) in
             self.data.Node_pairs_for_mdl_sp[r]]) + np.sum(
                [(self.data.Travel_time[h, l, s] + self.data.S_wait_bus) * self.varValue.x.get((s, r, h, l), 0) for
                 (h, l) in self.data.Hub_pairs]) for r in self.data.Trips for s in self.data.Scenarios]

        trip_info['follower travel distance'] = [np.sum(
            [self.data.Dist[i, j] * self.varValue.y.get((s, r, i, j), 0) for (i, j) in
             self.data.Node_pairs_for_mdl_sp[r]]) + np.sum(
            [self.data.Dist[h, l]* self.varValue.x.get((s, r, h, l), 0) for
             (h, l) in self.data.Hub_pairs]) for r in self.data.Trips for s in self.data.Scenarios]

        trip_info['cost per mile'] = [trip_info['follower cost'][i] / max(trip_info['follower travel distance'][i],1e-4) for i in
                                      range(len(trip_info['follower cost']))]


        df_trip_info = pd.DataFrame.from_dict(trip_info, orient='columns')

        """Collect hub Information"""
        hub_info = {'hub leg': [],
                    'solution type': [],
                    'opening cost': [],
                    'used times_low': [],
                    'used times_middle': [],
                    'used times_high': [],
                    'used times_total': [],
                    'served customers_low': [],
                    'served customers_middle': [],
                    'served customers_high': [],
                    'served customers_total': [],
                     }
        opened_hl_pairs = [(h,l) for (h,l) in self.data.Hub_pairs if self.varValue.z[h, l] > 0.1]
        hub_info['hub leg'] = [[h, l] for (h, l) in opened_hl_pairs]
        hub_info['solution type'] = [solution_type for (h, l) in opened_hl_pairs]
        hub_info['opening cost'] = [self.data.Beta_hl[h, l] for (h, l) in opened_hl_pairs]
        hub_info['used times_total'] = [
            np.sum([1 for r in self.data.Trips for s in self.data.Scenarios if self.varValue.x.get((s, r, h, l),0) > 0.1]) for
            (h, l) in opened_hl_pairs]  # how many times is the hub pari used
        hub_info['used times_low'] = [np.sum(
                [1 for r in self.data.Trips for s in self.data.Scenarios if self.varValue.x.get((s, r, h, l), 0) > 0.1
                 if self.data.node_income_level[self.data.Trip_destination[r]] == 'Low'])
            for (h, l) in opened_hl_pairs]  # how many times is the hub pari used
        hub_info['used times_high'] = [np.sum(
            [1 for r in self.data.Trips for s in self.data.Scenarios if self.varValue.x.get((s, r, h, l), 0) > 0.1
             if self.data.node_income_level[self.data.Trip_destination[r]] == 'High'])
            for (h, l) in opened_hl_pairs]  # how many times is the hub pari used
        hub_info['used times_middle'] = [np.sum(
            [1 for r in self.data.Trips for s in self.data.Scenarios if self.varValue.x.get((s, r, h, l), 0) > 0.1
             if self.data.node_income_level[self.data.Trip_destination[r]] == 'Middle'])
            for (h, l) in opened_hl_pairs]  # how many times is the hub pari used
        hub_info['served customers_total'] = [np.sum(
            [self.data.Trip_p_amount[s,r] for r in self.data.Trips for s in self.data.Scenarios if self.varValue.x.get((s, r, h, l), 0) > 0.1])
            for (h, l) in opened_hl_pairs]  # how many times is the hub pari used
        hub_info['served customers_high'] = [np.sum(
            [self.data.Trip_p_amount[s,r] for r in self.data.Trips for s in self.data.Scenarios if self.varValue.x.get((s, r, h, l), 0) > 0.1
             if self.data.node_income_level[self.data.Trip_destination[r]] == 'High'])
            for (h, l) in opened_hl_pairs]  # how many times is the hub pari used
        hub_info['served customers_low'] = [np.sum(
            [self.data.Trip_p_amount[s,r] for r in self.data.Trips for s in self.data.Scenarios if
             self.varValue.x.get((s, r, h, l), 0) > 0.1
             if self.data.node_income_level[self.data.Trip_destination[r]] == 'Low'])
            for (h, l) in opened_hl_pairs]  # how many times is the hub pari used
        hub_info['served customers_middle'] = [np.sum(
            [self.data.Trip_p_amount[s,r] for r in self.data.Trips for s in self.data.Scenarios if
             self.varValue.x.get((s, r, h, l), 0) > 0.1
             if self.data.node_income_level[self.data.Trip_destination[r]] == 'Middle'])
            for (h, l) in opened_hl_pairs]  # how many times is the hub pari used
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
        # use the correct single level tao and gamma, otherwise it's the same as the leader
        obj_info['Transport cost (follower)'] = sum(
        self.data.Scenarios_prob[s] * sum(
            self.data.Trip_p_amount[s,r] * (sum(
                self.data.tao_leader[h, l, s] * self.varValue.x.get((s, r, h, l),0) for (h, l) in
                self.data.Hub_pairs) + sum(
                self.data.gamma_leader[i, j, s] * self.varValue.y.get((s, r, i, j), 0) for (i, j) in
                self.data.Node_pairs_for_mdl_sp[r])) for r in self.data.Trips) for s in
        self.data.Scenarios)
        obj_info['Total = Facility cost + Transport cost (leader)'] = obj_info['Facility cost'] + obj_info[
            'Transport cost (leader)']

        obj_detail_info = {
            'file_name': [os.path.basename(self.data.file_name)[:-5]] * 6,
            'solution_type': [],
            'income': [],
            'transport_mode': [],
            'n_trips': [],
            'n_amount': [],
            'distance': [],
            'distance_product_amount': [],
            'leader_cost': [],
            'follower_cost': [],
            'facility_cost': [obj_info['Facility cost']] + [None] * 5,
            'total_cost': [obj_info['Total = Facility cost + Transport cost (leader)']] + [None] * 5,
            'travel_time': [],
            'cost_per_mile': [],
        }
        for income in ['High', 'Middle', 'Low']:
            for trans_mode in ['shuttle', 'shuttle+bus']:
                obj_detail_info['solution_type'].append(solution_type)
                obj_detail_info['income'].append(income)
                obj_detail_info['transport_mode'].append(trans_mode)
                if trans_mode == 'shuttle':
                    df = df_trip_info[(df_trip_info['income_level'] == income) & (df_trip_info['hubs'].str.len() == 0)]
                else:
                    df = df_trip_info[(df_trip_info['income_level'] == income) & (df_trip_info['hubs'].str.len() >= 1)]
                obj_detail_info['n_trips'].append(len(df))
                obj_detail_info['n_amount'].append((df['amounts']*df['scenario_prob']).sum())
                obj_detail_info['distance'].append((df['follower travel distance']*df['scenario_prob']).sum())
                obj_detail_info['distance_product_amount'].append((df['amounts']*df['follower travel distance']*df['scenario_prob']).sum())
                obj_detail_info['leader_cost'].append((df['amounts']*df['leader cost']*df['scenario_prob']).sum())
                obj_detail_info['follower_cost'].append((df['amounts']*df['follower cost']*df['scenario_prob']).sum())
                obj_detail_info['travel_time'].append((df['amounts']*df['follower travel time']*df['scenario_prob']).sum())
                obj_detail_info['cost_per_mile'].append((df['amounts']*df['cost per mile']*df['scenario_prob']).sum())

        df_obj_detail_info = pd.DataFrame.from_dict(obj_detail_info)

        return df_trip_info.copy(), df_hub_info.copy(), obj_info.copy(), df_obj_detail_info.copy()

def write_result(file_name=None):
    # write result
    current_directory = os.path.dirname(os.path.abspath(__file__))
    upper_2_dir = os.path.dirname(current_directory)
    tmp = os.path.join(upper_2_dir, 'output')
    if file_name is None:
        file_name = os.path.join(tmp, 'result_CCG.xlsx')
    else:
        file_name = os.path.join(tmp, file_name)

    df = pd.DataFrame.from_dict(result, orient='columns')
    df.to_excel(file_name, index=False)

def write_result_KKT(result_kkt, numerical_result_file):
    # write result
    df = pd.DataFrame.from_dict(result_kkt, orient='columns')
    df.to_excel(numerical_result_file, index=False)

def write_solution(CCG_iterator, file_tag):
    current_directory = os.path.dirname(os.path.abspath(__file__))
    upper_2_dir = os.path.dirname(current_directory)
    data_file = os.path.basename(CCG_iterator.data.file_name)[:-5]
    output_file_prefix = os.path.join(upper_2_dir, 'output', data_file)

    dict_z = {str(key): round(value) for key, value in CCG_iterator.varValue.z.items() if
              value}
    with open('{}_tag_{}_z_sol.json'.format(output_file_prefix, file_tag,
                                               ), 'w') as json_file:
        json.dump(dict_z, json_file, indent=4)
    dict_x = {str(key): round(value) for key, value in CCG_iterator.varValue.x.items() if
              value}
    with open('{}_tag_{}_x_sol.json'.format(output_file_prefix, file_tag,
                                               ), 'w') as json_file:
        json.dump(dict_x, json_file, indent=4)
    dict_y = {str(key): round(value) for key, value in CCG_iterator.varValue.y.items() if
              value}
    with open('{}_tag_{}_y_sol.json'.format(output_file_prefix, file_tag,
                                               ), 'w') as json_file:
        json.dump(dict_y, json_file, indent=4)

    return dict_z, dict_x, dict_y



if __name__ == '__main__':
    result = {'node-hub-trip-passenger': [],
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

    result_kkt = {
                  'file_name': [],
                  'node-hub-trip-passenger': [],
                  'arc elim': [],
                  'Delta type': [],
                  'use kkt': [],
                  'use sos1': [],
                  'use bigM': [],
                  'use strong duality': [],
                  'time read data': [],
                  'time model creation': [],
                  'solution time': [],
                  'obj':[],
                  'gap':[],
                  'n_node_BnB': [],  # number of explored BnB nodes
    }

    current_directory = os.path.dirname(os.path.abspath(__file__))
    upper_2_dir = os.path.dirname(current_directory)
    file_path = os.path.join(upper_2_dir, 'data', 'small_network')
    file_list = os.listdir(file_path)
    file_list_original = [i for i in file_list if '.xlsx' in i]
    file_list = [os.path.join(file_path, f) for f in file_list_original]
    file_list.sort()
    console_output_file = os.path.join(upper_2_dir, 'output', 'console_info.txt')
    numerical_result_file = os.path.join(upper_2_dir, 'output', 'cal_result.xlsx')
    # file_list = ['../data/20250425/node81-hub6-tripN200-scenario20-time2025-04-25-17-41-30.xlsx']
    out_sample_file = os.path.join(upper_2_dir, 'data', 'sample', 'sample-keep6hubs.xlsx')
    with open(console_output_file, 'a') as file:
        print('files to be executed: \n', file=file)
        for file_name in file_list:
            print(file_name, file=file)

    for file_name in file_list:
        with open(console_output_file, 'a') as file:
            print('================BEGIN NEW FILE======================', file=file)
            print('execute: ' + file_name, file=file)
        df_result = pd.DataFrame()

        try:
            for arc_elimination in [False]:
                for Delta_type in ['MST']:
                    with open(console_output_file, 'a') as file:
                        print('(arc elimination={}, Delta type={}) reading data .....'.format(arc_elimination, Delta_type), file=file)
                    tmp = time.time()
                    modeldata = Modeldata(file_name=file_name, arc_elimination=arc_elimination, Delta_type=Delta_type, Delta_value=5)
                    read_data_time = time.time() - tmp  # time for data reading

                    """ccg algorithm"""
                    # CCG_timelimit = 7200
                    # list_agg_cut = [False]
                    # agg_cut_part = False  # effect only when agg_cut is True
                    # solve_MP_use_Benders = False  # CCG MP is too large, use Benders to solve it (** only disaggregated cuts are possible!!)
                    # if solve_MP_use_Benders and [True] in list_agg_cut:
                    #     raise 'when use Benders for CCG MP, only disaggregated cuts are possible'
                    # for agg_cut in list_agg_cut:
                    #     for parallel_method in ['process']:
                    #         for use_lp_basis in [False]:
                    #             for solution_appro in ['primal']:
                    #                 for primal_based_dual_to_ini in [False]:
                    #                     for average_case in [False]:
                    #                         try:
                    #                             # use aggregated cut, primal approach for CCG, solve dual to initialize, warm start, process
                    #                             CCG_iterator = Iterate(data=modeldata, agg_cut=agg_cut,
                    #                                                    solution_appro=solution_appro,
                    #                                                    primal_based_dual_to_ini=primal_based_dual_to_ini,
                    #                                                    use_lp_basis=use_lp_basis,
                    #                                                    parallel_method=parallel_method,
                    #                                                    agg_cut_part=agg_cut_part,
                    #                                                    solve_MP_use_Benders=solve_MP_use_Benders)
                    #                             CCG_iterator.start_Benders_iteration(timelimit=CCG_timelimit)
                    #                             # CCG_iterator.MP.model.optimize()
                    #                             # CCG_iterator.MP.model.write('{}.sol'.format(modeldata.file_name[:-5]))
                    #
                    #                             write_solution(CCG_iterator=CCG_iterator, file_tag='None')
                    #
                    #                             result = CCG_iterator.update_result_ccg(result=result, read_data_time=read_data_time)
                    #                             write_result()
                    #                             del CCG_iterator
                    #                         except Exception as e:
                    #                             print(file_name, e, ':calculate terminated')

                    """original KKT/strong duality reformulation"""
                    try:
                        # use KKT, use bigM
                        reform_KKT_bigM = OneStageModelFullMultiplierDepends(data=modeldata, use_KKT=True, use_sos1=False, use_bigM=True, add_strong_dual=False)
                        reform_KKT_bigM.solve_model()
                        result_kkt = reform_KKT_bigM.update_result_record(result = result_kkt, read_data_time=read_data_time)
                        write_result_KKT(result_kkt, numerical_result_file)
                        # y = {key: reform_KKT_bigM.var.y[key].X for key in reform_KKT_bigM.var.y}
                        del reform_KKT_bigM


                        # use KKT, use sos1
                        reform_KKT_sos1 = OneStageModelFullMultiplierDepends(data=modeldata, use_KKT=True, use_sos1=True, use_bigM=False, add_strong_dual=False)
                        reform_KKT_sos1.solve_model()
                        result_kkt = reform_KKT_sos1.update_result_record(result=result_kkt, read_data_time=read_data_time)
                        write_result_KKT(result_kkt, numerical_result_file)
                        del reform_KKT_sos1

                        # # use KKT, use bigM, also use strong duality condition
                        # reform_KKT_bigM = OneStageModelFullMultiplierDepends(data=modeldata, use_KKT=True, use_sos1=False, use_bigM=True, add_strong_dual=True)
                        # reform_KKT_bigM.solve_model()
                        # result_kkt = reform_KKT_bigM.update_result_record(result=result_kkt, read_data_time=read_data_time)
                        # write_result_KKT(result_kkt, numerical_result_file)
                        # del reform_KKT_bigM
                        #
                        # # use KKT, use sos1, also use strong duality condition
                        # reform_KKT_sos1_strong_dual = OneStageModelFullMultiplierDepends(data=modeldata, use_KKT=True,
                        #                                                                  use_sos1=True, use_bigM=False, add_strong_dual=True)
                        # reform_KKT_sos1_strong_dual.solve_model()
                        # result_kkt = reform_KKT_sos1_strong_dual.update_result_record(result=result_kkt, read_data_time=read_data_time)
                        # write_result_KKT(result_kkt, numerical_result_file)
                        # del reform_KKT_sos1_strong_dual

                        # use strong duality
                        reform_KKT_strongDual = OneStageModelFullMultiplierDepends(data=modeldata, use_KKT=True, use_sos1=False, use_bigM=False, add_strong_dual=True)
                        reform_KKT_strongDual.solve_model()
                        result_kkt = reform_KKT_strongDual.update_result_record(result=result_kkt, read_data_time=read_data_time)
                        write_result_KKT(result_kkt, numerical_result_file)
                        del reform_KKT_strongDual

                    # # single level, right only when fr=dr, i.e., leader and follower share the same \tau and \gamma
                    # one_layer_model = OneLayerModel(data=modeldata, use_KKT=False, use_sos1=False)
                    # one_layer_model.solve_model()
                    # result_one_layer = one_layer_model.update_result_record(result=result_kkt, read_data_time=read_data_time)
                    # write_result_KKT()
                    # del one_layer_model

                    except Exception as e:
                        with open(console_output_file, 'a') as file:
                            content = file_name + ' :calculate terminated'
                            print(content, file=file)
                            print(e, file=file)

                    """compare uncertainty through C&CG"""
                    # solution_strategy = 'process'
                    # parser = get_parser()
                    # args = parser.parse_args()
                    # CCG_timelimit = args.CCG_timelimit
                    # use_i_ccg = args.use_i_ccg
                    # solve_MP_use_Benders = False  # CCG MP is too large, use Benders to solve it (** only disaggregated cuts are possible!!)
                    #
                    # # deterministic
                    # with open(console_output_file, 'a') as file:
                    #     print('********************solve deterministic model************************', file=file)
                    # # result = {key:[] for key in result}
                    # tmp = time.time()
                    # modeldata_determ = Modeldata(file_name=file_name, arc_elimination=arc_elimination, Delta_type=Delta_type,
                    #                       Delta_value=5)
                    # time_read_determ_data = time.time() - tmp
                    # modeldata_determ.obtain_average_value()  # obtain average value and convert to a deterministic instance
                    # CCG_iterator_determ = Iterate(data=modeldata_determ, agg_cut=False,
                    #                        solution_appro='primal',
                    #                        primal_based_dual_to_ini=False,
                    #                        use_lp_basis=False,
                    #                        parallel_method=solution_strategy,
                    #                        agg_cut_part=False,
                    #                        solve_MP_use_Benders=solve_MP_use_Benders)
                    # CCG_iterator_determ.start_Benders_iteration(timelimit=CCG_timelimit, use_i_ccg=use_i_ccg)
                    # df_trip_info_determ, df_hub_info_determ, obj_info_determ = CCG_iterator_determ.collect_solution_info(
                    #     solution_type='deterministic')
                    # dict_z_determ, _, _ = write_solution(CCG_iterator=CCG_iterator_determ, file_tag='determ')
                    # result = CCG_iterator_determ.update_result_ccg(result=result,
                    #                                               read_data_time=time_read_determ_data, tag='determ')
                    # write_result(file_name='calculation_info.xlsx')
                    # del CCG_iterator_determ
                    # del modeldata_determ
                    # gc.collect()
                    #
                    # # stochastic fix z as determ
                    # with open(console_output_file, 'a') as file:
                    #     print('********************stochastic fix z as determ************************', file=file)
                    # CCG_iterator_stoch = Iterate(data=modeldata, agg_cut=False,
                    #                              solution_appro='primal',
                    #                              primal_based_dual_to_ini=False,
                    #                              use_lp_basis=False,
                    #                              parallel_method=solution_strategy,
                    #                              agg_cut_part=False,
                    #                              solve_MP_use_Benders=solve_MP_use_Benders)  # declare a new model
                    #
                    # # result = {key: [] for key in result}
                    # CCG_iterator_stoch.MP.fix_z(given_z=dict_z_determ)
                    # CCG_iterator_stoch.start_Benders_iteration(timelimit=CCG_timelimit, use_i_ccg=use_i_ccg)
                    # df_trip_info_stoch_fix_z, df_hub_info_stoch_fix_z, obj_info_stoch_fix_z = CCG_iterator_stoch.collect_solution_info(
                    #     solution_type='stochastic_fix_z')
                    # stoch_fix_obj_val = min(CCG_iterator_stoch.UB_record)
                    # write_solution(CCG_iterator=CCG_iterator_stoch, file_tag='stoch_fix')
                    # result = CCG_iterator_stoch.update_result_ccg(result=result,
                    #                                                read_data_time=read_data_time, tag='stoch_fix_z_as_determ')
                    # write_result(file_name='calculation_info.xlsx')
                    # del CCG_iterator_stoch
                    # gc.collect()
                    #
                    #
                    # # single level, consider only leader
                    # # first copy the dicts
                    # gamma_leader = modeldata.gamma_leader.copy()
                    # gamma_follower = modeldata.gamma_follower.copy()
                    # tao_leader = modeldata.tao_leader.copy()
                    # tao_follower = modeldata.tao_follower.copy()
                    # with open(console_output_file, 'a') as file:
                    #     print('********************single level, consider only leader************************', file=file)
                    # modeldata.gamma_follower = gamma_leader.copy()
                    # modeldata.tao_follower = tao_leader.copy()
                    # CCG_iterator_stoch = Iterate(data=modeldata, agg_cut=False,
                    #                              solution_appro='primal',
                    #                              primal_based_dual_to_ini=False,
                    #                              use_lp_basis=False,
                    #                              parallel_method=solution_strategy,
                    #                              agg_cut_part=False,
                    #                              solve_MP_use_Benders=solve_MP_use_Benders)
                    # CCG_iterator_stoch.start_Benders_iteration(timelimit=CCG_timelimit, use_i_ccg=use_i_ccg)
                    # df_trip_info_single_leader, df_hub_info_single_leader, obj_info_stoch_single_leader = CCG_iterator_stoch.collect_solution_info(
                    #     solution_type='single_level_leader', model_type='single_leader', real_tao=tao_follower, real_gamma=gamma_follower)
                    # single_leader_obj_val = CCG_iterator_stoch.LB_record[-1]
                    # write_solution(CCG_iterator=CCG_iterator_stoch, file_tag='single_leader')
                    # result = CCG_iterator_stoch.update_result_ccg(result=result,
                    #                                               read_data_time=read_data_time, tag='sigle_level_leader')
                    # write_result(file_name='calculation_info.xlsx')
                    # del CCG_iterator_stoch
                    # gc.collect()
                    # # recover the parameters
                    # modeldata.gamma_follower = gamma_follower.copy()
                    # modeldata.tao_follower = tao_follower.copy()
                    #
                    # # stochastic
                    # with open(console_output_file, 'a') as file:
                    #     print('********************solve stochastic model************************', file=file)
                    # CCG_iterator_stoch = Iterate(data=modeldata, agg_cut=False,
                    #                        solution_appro='primal',
                    #                        primal_based_dual_to_ini=False,
                    #                        use_lp_basis=False,
                    #                        parallel_method=solution_strategy,
                    #                        agg_cut_part=False,
                    #                        solve_MP_use_Benders=solve_MP_use_Benders)
                    # CCG_iterator_stoch.MP.restrict_bound_obj(lb=single_leader_obj_val, ub=stoch_fix_obj_val)
                    # CCG_iterator_stoch.start_Benders_iteration(timelimit=CCG_timelimit, use_i_ccg=use_i_ccg)
                    # df_trip_info_stoch, df_hub_info_stoch, obj_info_stoch = CCG_iterator_stoch.collect_solution_info(
                    #     solution_type='stochastic')
                    # dict_z_stoch, _, _ = write_solution(CCG_iterator=CCG_iterator_stoch, file_tag='stoch')
                    # result = CCG_iterator_stoch.update_result_ccg(result=result,
                    #                                         read_data_time=read_data_time, tag='stochastic')
                    # write_result(file_name='calculation_info.xlsx')
                    #
                    # del CCG_iterator_stoch
                    # gc.collect()
                    #
                    # # write result
                    # with open(console_output_file, 'a') as file:
                    #     print('*****************write model results*********************', file=file)
                    # df_trip_info = pd.concat(
                    #     [df_trip_info_determ, df_trip_info_stoch_fix_z, df_trip_info_stoch, df_trip_info_single_leader])
                    # df_hub_info = pd.concat(
                    #     [df_hub_info_determ, df_hub_info_stoch_fix_z, df_hub_info_stoch, df_hub_info_single_leader])
                    # dict_obj_info = {'determ': obj_info_determ, 'stochastic_fix_z': obj_info_stoch_fix_z,
                    #                  'stochastic': obj_info_stoch, 'single_leader': obj_info_stoch_single_leader}
                    # df_obj_info = pd.DataFrame.from_dict(dict_obj_info, orient='index')
                    #
                    # upper_2_dir = os.path.dirname(current_directory)
                    # file_path = os.path.join(upper_2_dir, 'output',
                    #                          os.path.basename(file_name)[:-5] + 'opt_solution_info.xlsx')
                    # with pd.ExcelWriter(file_path) as writer:
                    #     df_trip_info.to_excel(excel_writer=writer, sheet_name='trip_info', index=False)
                    #     df_hub_info.to_excel(excel_writer=writer, sheet_name='hub_info', index=False)
                    #     df_obj_info.to_excel(excel_writer=writer, sheet_name='obj_info_in_sample', index=True)
                    #
                    #
                    #
                    # if dict_z_stoch == dict_z_determ:
                    #     with open(console_output_file, 'a') as file:
                    #         print("Solution in stochastic and determ models are the same. Proceed to the next file.",
                    #               file=file)
                    #         print('=====================CURRENT CALCULATION EXITED============================', file=file)
                    #     continue


                    # out of sample scenarios evaluation
                    # with open(console_output_file, 'a') as file:
                    #     print('********************out of sample scenarios evaluation************************', file=file)
                    # modeldata_out_sample = Modeldata(file_name=out_sample_file, arc_elimination=arc_elimination,
                    #                                  Delta_type=Delta_type, Delta_value=5)
                    # list_df_obj_out_sample = []
                    # for sce in modeldata_out_sample.Scenarios:
                    #     modeldata_out_sample.Scenarios = [sce]
                    #     modeldata_out_sample.Scenarios_prob = {sce: 1}
                    #     modeldata_out_sample.Sce_Trip = {(s, r) for s in modeldata_out_sample.Scenarios for r in
                    #                                      modeldata_out_sample.Trips}
                    #     CCG_iterator_tmp = Iterate(data=modeldata_out_sample, agg_cut=False,
                    #                        solution_appro='primal',
                    #                        primal_based_dual_to_ini=False,
                    #                        use_lp_basis=False,
                    #                        parallel_method=solution_strategy,
                    #                        agg_cut_part=False,
                    #                        solve_MP_use_Benders=solve_MP_use_Benders)
                    #     # evaluate deterministic solution
                    #     CCG_iterator_tmp.MP.fix_z(dict_z_determ)
                    #     CCG_iterator_tmp.start_Benders_iteration(timelimit=CCG_timelimit, use_i_ccg=use_i_ccg)
                    #     df = CCG_iterator_tmp.collect_out_sample_solution_info(solution_type='determ')
                    #     obj_value_determ = df['Total = Facility cost + Transport cost (leader)'][0]
                    #     list_df_obj_out_sample.append(df.copy())
                    #     # evaluate stochastic solution
                    #     CCG_iterator_tmp.MP.fix_z(dict_z_stoch)
                    #     CCG_iterator_tmp.start_Benders_iteration(timelimit=CCG_timelimit, use_i_ccg=use_i_ccg)
                    #     df = CCG_iterator_tmp.collect_out_sample_solution_info(solution_type='stochastic')
                    #     if df['Total = Facility cost + Transport cost (leader)'][0] < obj_value_determ:
                    #         df['Stochastic is better'] = 'True'
                    #     elif df['Total = Facility cost + Transport cost (leader)'][0] == obj_value_determ:
                    #         df['Stochastic is better'] = 'Same'
                    #     else:
                    #         df['Stochastic is better'] = 'False'
                    #
                    #     list_df_obj_out_sample.append(df.copy())
                    #
                    # del CCG_iterator_tmp
                    # del modeldata_out_sample
                    # gc.collect()



                    # out of sample scenarios evaluation - use single level
                    # with open(console_output_file, 'a') as file:
                    #     print('********************out of sample scenarios evaluation single level************************', file=file)
                    # modeldata_out_sample = Modeldata(file_name=out_sample_file, arc_elimination=arc_elimination,
                    #                                  Delta_type=Delta_type, Delta_value=5)
                    # list_df_obj_out_sample_sglv = []
                    # for sce in modeldata_out_sample.Scenarios:
                    #     modeldata_out_sample.Scenarios = [sce]
                    #     modeldata_out_sample.Scenarios_prob = {sce: 1}
                    #     modeldata_out_sample.Sce_Trip = {(s, r) for s in modeldata_out_sample.Scenarios for r in
                    #                                      modeldata_out_sample.Trips}
                    #     reform_KKT_strongDual = OneStageModelFullMultiplierDepends(data=modeldata_out_sample, use_KKT=True,
                    #                                                                use_sos1=False, use_bigM=False,
                    #                                                                add_strong_dual=True)
                    #     # evaluate deterministic solution
                    #     reform_KKT_strongDual.fix_z(dict_z_determ)
                    #     reform_KKT_strongDual.solve_model()
                    #     df = reform_KKT_strongDual.collect_out_sample_solution_info(solution_type='determ')
                    #     obj_value_determ = df['Total = Facility cost + Transport cost (leader)'][0]
                    #     list_df_obj_out_sample_sglv.append(df.copy())
                    #
                    #     # evaluate stochastic solution
                    #     reform_KKT_strongDual.fix_z(dict_z_stoch)
                    #     reform_KKT_strongDual.solve_model()
                    #     df = reform_KKT_strongDual.collect_out_sample_solution_info(solution_type='stochastic')
                    #
                    #     if df['Total = Facility cost + Transport cost (leader)'][0] < obj_value_determ:
                    #         df['Stochastic is better'] = 'True'
                    #     elif df['Total = Facility cost + Transport cost (leader)'][0] == obj_value_determ:
                    #         df['Stochastic is better'] = 'Same'
                    #     else:
                    #         df['Stochastic is better'] = 'False'
                    #
                    #     list_df_obj_out_sample_sglv.append(df.copy())
                    #
                    # del reform_KKT_strongDual
                    # del modeldata_out_sample
                    # gc.collect()

                    # # single level, only follower
                    # with open(console_output_file, 'a') as file:
                    #     print('********************single level, consider only follower************************', file=file)
                    # # result = {key: [] for key in result}
                    # modeldata.gamma_leader = gamma_follower.copy()
                    # modeldata.tao_leader = tao_follower.copy()
                    # modeldata.gamma_follower = gamma_follower.copy()
                    # modeldata.tao_follower = tao_follower.copy()
                    # CCG_iterator_stoch = Iterate(data=modeldata, agg_cut=False,
                    #                              solution_appro='primal',
                    #                              primal_based_dual_to_ini=False,
                    #                              use_lp_basis=False,
                    #                              parallel_method=solution_strategy,
                    #                              agg_cut_part=False,
                    #                              solve_MP_use_Benders=solve_MP_use_Benders)
                    # CCG_iterator_stoch.start_Benders_iteration(timelimit=CCG_timelimit)
                    # df_trip_info_single_follower, df_hub_info_single_follower, obj_info_stoch_single_follower = CCG_iterator_stoch.collect_solution_info(
                    #     solution_type='single_level_follower', model_type='single_follower', real_tao=tao_leader, real_gamma=gamma_leader)
                    # write_solution(CCG_iterator=CCG_iterator_stoch, file_tag='single_follower')
                    # result = CCG_iterator_stoch.update_result_ccg(result=result,
                    #                                               read_data_time=read_data_time, tag='single_level_follower')
                    # write_result(file_name='calculation_info.xlsx')
                    # del CCG_iterator_stoch
                    # gc.collect()

                    # # write result
                    # with open(console_output_file, 'a') as file:
                    #     print('*****************write out of sample results*********************', file=file)
                    # # df_obj_out_sample = pd.concat(list_df_obj_out_sample)
                    # df_obj_out_sample_snglv = pd.concat(list_df_obj_out_sample_sglv)
                    #
                    # upper_2_dir = os.path.dirname(current_directory)
                    # file_path = os.path.join(upper_2_dir, 'output', os.path.basename(file_name)[:-5]+'out_sample.xlsx')
                    # with pd.ExcelWriter(file_path) as writer:
                    #     # df_obj_out_sample.to_excel(excel_writer=writer, sheet_name='obj_info_out_sample', index=False)
                    #     df_obj_out_sample_snglv.to_excel(excel_writer=writer, sheet_name='obj_info_out_sample_snglv', index=False)
        except Exception as e:
            with open(console_output_file, 'a') as file:
                print(file_name, e, ':calculate terminated at the end line.', file=file)






# class Iterate:
#     """Benders iterations"""
#
#     def __init__(self, data: Modeldata, agg_cut: bool, use_lp_basis=False, parallel_method='normal',
#                  solution_appro='dual', primal_based_dual_to_ini=False, agg_cut_part=False, solve_MP_use_Benders=False,
#                  use_mdl_copy=False, copy_mdl=None, revise_dual_value=False):
#         """
#         :param data:
#         :param agg_cut:
#         :param use_lp_basis: use lp basis as warm start
#         :param parallel_method: 'normal': solve in sequence; 'thread': threading; 'process': multiprocessing
#         :param solution_appro: 'dual': approach1, based on dual formulation; 'primal': approach 2, based on primal formulation
#         :param primal_based_dual_to_ini: use primal approach, but also solve dual at the initail scenario
#         """
#         current_directory = os.path.dirname(os.path.abspath(__file__))
#         upper_2_dir = os.path.dirname(current_directory)
#         self.console_output_file = os.path.join(upper_2_dir, 'output', 'console_info.txt')
#         with open(self.console_output_file, 'a') as file:
#             print('initiate data...', file=file)
#         self.data = data
#         self.agg_cut = agg_cut
#         self.use_lp_basis = use_lp_basis
#         self.parallel_method = parallel_method
#         self.solve_MP_use_Benders = solve_MP_use_Benders
#         if self.parallel_method not in ['normal', 'thread', 'process']:
#             with open(self.console_output_file, 'a') as file:
#                 print('Unknown parallel method: {}, please confirm. Using normal instead'.format(self.parallel_method), file=file)
#             self.parallel_method = 'normal'
#
#         self.solution_appro = solution_appro
#         if self.solution_appro not in ['primal', 'dual']:
#             raise 'unknown parameter solution_appro in function start_Benders_iteration()'
#         self.primal_based_dual_to_ini = primal_based_dual_to_ini
#         self.revise_dual_value = revise_dual_value
#
#         with open(self.console_output_file, 'a') as file:
#             if ':' in os.path.abspath(__file__):
#                 self.run_in_linux_server = False
#                 print('The code is running on Windows System', file=file)
#             else:
#                 self.run_in_linux_server = True
#                 print('The code is running on Linux System', file=file)
#         with open(self.console_output_file, 'a') as file:
#             print('create MP model...', file=file)
#         tmp = time.time()
#         if not self.solve_MP_use_Benders:
#             self.MP = MasterModel(data=data, agg_cut=agg_cut, solution_appro=self.solution_appro,
#                                   agg_cut_part=agg_cut_part, on_linux=self.run_in_linux_server,
#                                   use_mdl_copy=use_mdl_copy, copy_mdl=copy_mdl)
#         else:
#             with open(self.console_output_file, 'a') as file:
#                 print('\t\tSolve MP with Benders is not correct. Exit.', file=file)
#             exit()
#         self.time_MP_creation = time.time() - tmp  # time cost including adding vars and constrs
#
#         with open(self.console_output_file, 'a') as file:
#             print('\t\tdeclare empty lists and dicts...', file=file)
#         tmp_s = time.time()
#         self.s_c_cut_stat = {(s,r):0 for (s,r) in self.data.Sce_Trip}
#         self.dict_SP = {}  # for approach 1
#         self.dict_SP_dual = {}  # for approach 2
#         self.dict_SP_primal = {}  # for approach 2
#         self.dict_originalSP = {}
#         self.dict_CCGSP = {}
#         self.create_submodels()
#         self.time_SP_creation = time.time() - tmp_s  # time cost including adding vars and constrs
#         self.time_mdl_creation = time.time() - tmp  # time for MP and SP model creation
#         self.varValue = VarValue()
#         self.LB_record = []  # record of lower bound
#         self.UB_record = []  # record of upper bound
#         self.time_record = []
#         self.time_start = time.time()  # start time
#         self.time_record_solve_MP = []
#         self.time_record_solve_SP = []
#         self.SP_VBasis = {}  # VBasis of subproblems, to accelerate the parallel computing
#         self.SP_CBasis = {}
#         self.VBasis_info = {}  # VBasis information
#         self.time_retrieve_MP_solution = []
#         self.time_retrieve_SP_solution = []
#         self.time_update_MP = []
#         self.time_update_SP = []
#         self.list_n_cut_in_iter = []
#
#     def create_submodels(self):
#         if self.solution_appro == 'dual':
#             if self.parallel_method in ['normal', 'thread']:
#                 # created model in advance only in normal and thread parallel
#                 for s in self.data.Scenarios:
#                     for r in self.data.Trips:
#                         self.dict_SP[s, r] = SubModel(data=self.data, s=s, r=r)
#                         self.dict_SP[s, r].declare_model()
#             elif self.parallel_method == 'process':
#                 self.SP_template = SubModel(data=self.data, s=self.data.Scenarios[0], r=self.data.Trips[0])
#         elif self.solution_appro == 'primal':
#             if self.parallel_method in ['normal', 'thread']:
#                 # created model in advance only in normal and thread parallel
#                 for s in self.data.Scenarios:
#                     for r in self.data.Trips:
#                         self.dict_SP_primal[s, r] = SubShortestPathWithStrongDualModel(data=self.data, s=s, r=r)
#                         self.dict_SP_primal[s, r].declare_model()
#                 for r in self.data.Trips:
#                     # only create one dual-based model
#                     self.dict_SP_dual[self.data.Scenarios[0], r] = SubModel(data=self.data, s=self.data.Scenarios[0], r=r)
#                     self.dict_SP_dual[self.data.Scenarios[0], r].declare_model()
#
#             elif self.parallel_method == 'process':
#                 self.SP_template_dual = SubModel(data=self.data, s=self.data.Scenarios[0], r=self.data.Trips[0])
#                 self.SP_template_primal = SubShortestPathWithStrongDualModel(data=self.data, s=self.data.Scenarios[0],
#                                                                              r=self.data.Trips[0])
#
#     def solve_subproblems_primal(self, n_iters, use_dual_also:bool=True, need_x = False, revise_dual_value=False):
#         """solve subproblems: approach 2 based on primal formulation
#         :param use_dual_also: use dual info to warm start
#         """
#         if self.parallel_method == 'normal':
#             time_update_SP = 0
#             time_record_solve_SP = 0
#             time_retrieve_SP_solution = 0
#             # in the first iteration, solve without warmstart
#             for s in self.data.Scenarios:
#                 for r in self.data.Trips:
#                     if not use_dual_also:  # solve only primal models
#                         tmp = time.time()
#                         self.dict_SP_primal[s,r].update_objective_cons(param_z=self.varValue.z)
#                         time_update_SP += time.time() - tmp
#
#                         tmp = time.time()
#                         if self.use_lp_basis:
#                             self.dict_SP_primal[s, r].write_lp_basis_info(True, self.SP_VBasis.get(r, None),
#                                                                           self.SP_CBasis.get(r, None))
#                         else:
#                             self.dict_SP_primal[s, r].write_lp_basis_info()
#                         time_update_SP += time.time() - tmp
#                         tmp = time.time()
#                         self.dict_SP_primal[s, r].model.optimize()
#                         time_record_solve_SP += time.time() - tmp
#                         tmp = time.time()
#                         self.dict_SP_primal[s, r].change_obj()  # change obj to leader params
#                         time_update_SP += time.time() - tmp
#                         tmp = time.time()
#                         self.dict_SP_primal[s, r].model.optimize()
#                         time_record_solve_SP += time.time() - tmp
#                         tmp = time.time()
#                         result_primal = self.dict_SP_primal[s, r].obtain_solution_info(self.use_lp_basis)
#                         time_retrieve_SP_solution += time.time() - tmp
#
#                         tmp = time.time()
#                         self.varValue.update_Sub_var_primal(s, r,
#                                                             x = result_primal['x'],
#                                                             y = result_primal['y'],
#                                                             pi1=result_primal['pi1'],
#                                                             pi2=result_primal['pi2'],
#                                                             pi3=result_primal['pi3'],
#                                                             pi4=result_primal['pi4'],
#                                                             t=result_primal['t'],
#                                                             obj=result_primal['obj'])#,
#                                                             # VBasis=result_primal['VBasis'],
#                                                             # CBasis=result_primal['CBasis']
#                         time_retrieve_SP_solution += time.time() - tmp
#
#
#                     else:
#                         if n_iters > 1 and s == self.data.Scenarios[0]:
#                             tmp = time.time()
#                             # VBasis_info = {
#                             #     'pi1': [self.varValue.pi1[s, r, k] for k in self.data.Node],
#                             #     'pi2': [self.varValue.pi2[s, r, h, l] for (h, l) in self.data.Hub_pairs],
#                             #     'pi3': [self.varValue.pi3[s, r, h, l] for (h, l) in self.data.Hub_pairs],
#                             #     'pi4': [self.varValue.pi4[s, r, i, j] for (i, j) in self.data.Node_pairs_for_mdl_sp[r]],
#                             #     't': self.varValue.t[s,r]
#                             # }
#                             self.dict_SP_dual[s, r].update_objective_cons(param_z=self.varValue.z)
#                             time_update_SP += time.time() - tmp
#
#                             tmp = time.time()
#                             if self.use_lp_basis:
#                                 self.dict_SP_dual[s,r].write_lp_basis_info(True, None, None, self.VBasis_info.get(r, None))
#                             else:
#                                 self.dict_SP_dual[s,r].write_lp_basis_info()
#                             time_update_SP +=time.time() - tmp
#
#                             tmp = time.time()
#                             self.dict_SP_dual[s, r].model.optimize()
#                             time_record_solve_SP += time.time() - tmp
#
#                             tmp = time.time()
#                             result_dual = self.dict_SP_dual[s, r].obtain_solution_info(use_lp_basis=self.use_lp_basis)
#                             time_retrieve_SP_solution += time.time() - tmp
#
#                             tmp = time.time()
#                             self.varValue.update_Sub_var_primal(s, r,
#                                                                 x=result_dual['x'],
#                                                                 y=result_dual['y'],
#                                                                 pi1=result_dual['pi1'],
#                                                                 pi2=result_dual['pi2'],
#                                                                 pi3=result_dual['pi3'],
#                                                                 pi4=result_dual['pi4'],
#                                                                 t=result_dual['t'],
#                                                                 obj=result_dual['obj'])#,
#                                                                 # VBasis=list(result_dual['x'].values()) + list(result_dual['y'].values()),
#                                                                 # CBasis=None)
#                             time_retrieve_SP_solution += time.time() - tmp
#                         else:
#                             tmp = time.time()
#                             self.dict_SP_primal[s, r].update_objective_cons(param_z=self.varValue.z)
#                             time_update_SP += time.time() - tmp
#
#                             tmp = time.time()
#                             if self.use_lp_basis:
#                                 self.dict_SP_primal[s, r].write_lp_basis_info(True, self.SP_VBasis.get(r, None),
#                                                                               self.SP_CBasis.get(r, None))
#                             else:
#                                 self.dict_SP_primal[s, r].write_lp_basis_info()
#                             time_update_SP += time.time() - tmp
#                             tmp = time.time()
#                             self.dict_SP_primal[s, r].model.optimize()
#                             time_record_solve_SP += time.time() - tmp
#                             tmp = time.time()
#                             self.dict_SP_primal[s, r].change_obj()  # change obj to leader params
#                             time_update_SP += time.time() - tmp
#                             tmp = time.time()
#                             self.dict_SP_primal[s, r].model.optimize()
#                             time_record_solve_SP += time.time() - tmp
#                             tmp = time.time()
#                             result_primal = self.dict_SP_primal[s, r].obtain_solution_info(use_lp_basis=self.use_lp_basis)
#                             time_retrieve_SP_solution += time.time() - tmp
#
#                             tmp = time.time()
#                             self.varValue.update_Sub_var_primal(s, r,
#                                                                 x=result_primal['x'],
#                                                                 y=result_primal['y'],
#                                                                 pi1=result_primal['pi1'],
#                                                                 pi2=result_primal['pi2'],
#                                                                 pi3=result_primal['pi3'],
#                                                                 pi4=result_primal['pi4'],
#                                                                 t=result_primal['t'],
#                                                                 obj=result_primal['obj'])#,
#                                                                 # VBasis=result_primal['VBasis'],
#                                                                 # CBasis=result_primal['CBasis'])
#                             time_retrieve_SP_solution += time.time() - tmp
#
#             self.time_update_SP.append(time_update_SP)
#             self.time_record_solve_SP.append(time_record_solve_SP)
#             self.time_retrieve_SP_solution.append(time_retrieve_SP_solution)
#
#         elif self.parallel_method == 'thread':
#             time_update_SP = 0
#             time_record_solve_SP = 0
#             time_retrieve_SP_solution = 0
#             if (not use_dual_also) or (use_dual_also and n_iters == 1): # solve only primal problems
#                 # solve scenario[0]
#                 # pool = ThreadPool()
#                 # if self.use_lp_basis:
#                 #     result = pool.starmap(self.aux_solve_parallel_primal,
#                 #                           [(self.data.Scenarios[0], r, True, self.SP_VBasis.get(r, None),
#                 #                             self.SP_CBasis.get(r, None)) for r in self.data.Trips])
#                 # else:
#                 #     result = pool.starmap(self.aux_solve_parallel_primal,
#                 #                           [(self.data.Scenarios[0], r) for r in self.data.Trips])
#                 # pool.close()
#                 # pool.join()
#
#                 # update
#                 tmp = time.time()
#                 pool = ThreadPool(processes=ALLOW_CORE)
#                 if self.use_lp_basis:
#                     pool.starmap(self.aux_solve_parallel_primal_update,
#                                           [(self.data.Scenarios[0], r, True, self.SP_VBasis.get(r, None),
#                                             self.SP_CBasis.get(r, None)) for r in self.data.Trips])
#                 else:
#                     pool.starmap(self.aux_solve_parallel_primal_update,
#                                           [(self.data.Scenarios[0], r) for r in self.data.Trips])
#                 pool.close()
#                 pool.join()
#                 time_update_SP += time.time() - tmp
#
#                 # solve
#                 tmp = time.time()
#                 pool = ThreadPool(processes=ALLOW_CORE)
#                 pool.starmap(self.aux_solve_parallel_primal_optimize,
#                              [(self.data.Scenarios[0], r) for r in self.data.Trips])
#                 pool.close()
#                 pool.join()
#                 time_record_solve_SP += time.time() - tmp
#
#
#                 # change
#                 tmp = time.time()
#                 pool = ThreadPool(processes=ALLOW_CORE)
#                 pool.starmap(self.aux_solve_parallel_primal_change_obj,
#                              [(self.data.Scenarios[0], r) for r in self.data.Trips])
#                 pool.close()
#                 pool.join()
#                 time_update_SP += time.time() - tmp
#
#                 # solve
#                 tmp = time.time()
#                 pool = ThreadPool(processes=ALLOW_CORE)
#                 pool.starmap(self.aux_solve_parallel_primal_optimize,
#                              [(self.data.Scenarios[0], r) for r in self.data.Trips])
#                 pool.close()
#                 pool.join()
#                 time_record_solve_SP += time.time() - tmp
#
#                 # obtain info
#                 pool = ThreadPool(processes=ALLOW_CORE)
#                 result = pool.starmap(self.aux_solve_parallel_primal_retrieval,
#                              [(self.data.Scenarios[0], r, self.use_lp_basis) for r in self.data.Trips])
#                 pool.close()
#                 pool.join()
#
#                 tmp = time.time()
#                 result_dict = {list(item.keys())[0]: item[list(item.keys())[0]] for item in result}
#                 for r in self.data.Trips:
#                     item = result_dict[self.data.Scenarios[0], r]
#                     self.varValue.update_Sub_var_primal(self.data.Scenarios[0], r,
#                                                             x=item['x'],
#                                                             y=item['y'],
#                                                             pi1=item['pi1'],
#                                                             pi2=item['pi2'],
#                                                             pi3=item['pi3'],
#                                                             pi4=item['pi4'],
#                                                             t=item['t'],
#                                                             obj=item['obj'])#,
#                                                             # VBasis=item['VBasis'],
#                                                             # CBasis=item['CBasis'])
#                     if self.use_lp_basis:
#                         self.SP_VBasis[r] = item['VBasis']
#                         self.SP_CBasis[r] = item['CBasis']
#                         VBasis_info = {
#                             'pi1': [item['pi1'][k] for k in self.data.Node_of_trip[r]],
#                             'pi2': [item['pi2'][h, l] for (h, l) in self.data.Hub_pairs],
#                             'pi3': [item['pi3'][h, l] for (h, l) in self.data.Hub_pairs],
#                             'pi4': [item['pi4'][i, j] for (i, j) in self.data.Node_pairs_for_mdl_sp[r]],
#                             't': item['t']
#                         }
#                         self.VBasis_info[r] = VBasis_info.copy()  # VBasis for dual model
#                     # TODO: VBasis_info is the warm start info for dual, can we always use that from iter 1, instead of update that each iter?
#                 time_retrieve_SP_solution += time.time() - tmp
#
#
#
#                 # then solve remaining scenarios
#                 tmp = time.time()
#                 pool = ThreadPool(processes=ALLOW_CORE)
#                 if self.use_lp_basis:
#                     pool.starmap(self.aux_solve_parallel_primal_update,
#                                           [(s, r, True, self.SP_VBasis[r], self.SP_CBasis[r]) for s in
#                                            self.data.Scenarios[1:] for r in self.data.Trips])
#                 else:
#                     pool.starmap(self.aux_solve_parallel_primal_update,
#                                           [(s, r) for s in self.data.Scenarios[1:] for r in self.data.Trips])
#                 pool.close()
#                 pool.join()
#                 time_update_SP += time.time() - tmp
#
#                 # solve
#                 tmp = time.time()
#                 pool = ThreadPool(processes=ALLOW_CORE)
#                 pool.starmap(self.aux_solve_parallel_primal_optimize,
#                              [(s, r) for s in self.data.Scenarios[1:] for r in self.data.Trips])
#                 pool.close()
#                 pool.join()
#                 time_record_solve_SP += time.time() - tmp
#
#                 # change
#                 tmp = time.time()
#                 pool = ThreadPool(processes=ALLOW_CORE)
#                 pool.starmap(self.aux_solve_parallel_primal_change_obj,
#                              [(s, r) for s in self.data.Scenarios[1:] for r in self.data.Trips])
#                 pool.close()
#                 pool.join()
#                 time_update_SP += time.time() - tmp
#
#                 # solve
#                 tmp = time.time()
#                 pool = ThreadPool(processes=ALLOW_CORE)
#                 pool.starmap(self.aux_solve_parallel_primal_optimize,
#                              [(s, r) for s in self.data.Scenarios[1:] for r in self.data.Trips])
#                 pool.close()
#                 pool.join()
#                 time_record_solve_SP += time.time() - tmp
#
#                 # obtain info
#                 pool = ThreadPool(processes=ALLOW_CORE)
#                 result = pool.starmap(self.aux_solve_parallel_primal_retrieval,
#                                       [(s, r, self.use_lp_basis) for s in self.data.Scenarios[1:] for r in
#                                        self.data.Trips])
#                 pool.close()
#                 pool.join()
#
#                 tmp = time.time()
#                 result_dict = {list(item.keys())[0]: item[list(item.keys())[0]] for item in result}
#                 for r in self.data.Trips:
#                     for s in self.data.Scenarios[1:]:
#                         item = result_dict[s, r]
#                         self.varValue.update_Sub_var_primal(s, r,
#                                                             x=item['x'],
#                                                             y=item['y'],
#                                                             pi1=item['pi1'],
#                                                             pi2=item['pi2'],
#                                                             pi3=item['pi3'],
#                                                             pi4=item['pi4'],
#                                                             t=item['t'],
#                                                             obj=item['obj'])  # ,
#                                                             # VBasis=item['VBasis'],
#                                                             # CBasis=item['CBasis'])
#                 time_retrieve_SP_solution += time.time() - tmp
#
#             else:  # solve primal, but also dual for scenario 1
#                 # solve scenario 1
#                 tmp = time.time()
#                 pool = ThreadPool(processes=ALLOW_CORE)
#                 if self.use_lp_basis:
#                     # if use lp basis as warm start
#                     pool.starmap(self.aux_solve_parallel_dual_update,
#                                           [(self.data.Scenarios[0], r, True, None, None,
#                                             self.VBasis_info[r]) for r in self.data.Trips])
#                 else:
#                     pool.starmap(self.aux_solve_parallel_dual_update,
#                                           [(self.data.Scenarios[0], r) for r in
#                                            self.data.Trips])
#                 pool.close()
#                 pool.join()
#                 time_update_SP += time.time() - tmp
#
#                 # solve
#                 tmp = time.time()
#                 pool = ThreadPool(processes=ALLOW_CORE)
#                 pool.starmap(self.aux_solve_parallel_dual_optimize,
#                                       [(self.data.Scenarios[0], r) for r in
#                                        self.data.Trips])
#                 pool.close()
#                 pool.join()
#                 time_record_solve_SP += time.time() - tmp
#
#                 # obtain info
#                 tmp = time.time()
#                 pool = ThreadPool(processes=ALLOW_CORE)
#                 result = pool.starmap(self.aux_solve_parallel_dual_retrieval,
#                                       [(self.data.Scenarios[0], r, self.use_lp_basis) for r in
#                                        self.data.Trips])
#                 pool.close()
#                 pool.join()
#                 time_retrieve_SP_solution += time.time() - tmp
#
#                 tmp = time.time()
#                 result_dict = {list(item.keys())[0]: item[list(item.keys())[0]] for item in result}
#                 for r in self.data.Trips:
#                     item = result_dict[self.data.Scenarios[0], r]
#                     self.varValue.update_Sub_var_primal(self.data.Scenarios[0], r,
#                                                         x=item['x'],
#                                                         y=item['y'],
#                                                         pi1=item['pi1'],
#                                                         pi2=item['pi2'],
#                                                         pi3=item['pi3'],
#                                                         pi4=item['pi4'],
#                                                         t=item['t'],
#                                                         obj=item['obj'])  # ,
#                                                         # VBasis=item['VBasis'],
#                                                         # CBasis=item['CBasis'])
#                     if self.use_lp_basis:
#                         self.SP_VBasis[r] = list(item['x'].values()) + list(item['y'].values())
#                 time_retrieve_SP_solution += time.time() - tmp
#
#                 # solve remaining scenarios
#                 tmp = time.time()
#                 pool = ThreadPool(processes=ALLOW_CORE)
#                 if self.use_lp_basis:
#                     pool.starmap(self.aux_solve_parallel_primal_update,
#                                  [(s, r, True, self.SP_VBasis[r], self.SP_CBasis[r]) for s in
#                                   self.data.Scenarios[1:] for r in self.data.Trips])
#                 else:
#                     pool.starmap(self.aux_solve_parallel_primal_update,
#                                  [(s, r) for s in self.data.Scenarios[1:] for r in self.data.Trips])
#                 pool.close()
#                 pool.join()
#                 time_update_SP += time.time() - tmp
#
#                 # solve
#                 tmp = time.time()
#                 pool = ThreadPool(processes=ALLOW_CORE)
#                 pool.starmap(self.aux_solve_parallel_primal_optimize,
#                              [(s, r) for s in self.data.Scenarios[1:] for r in self.data.Trips])
#                 pool.close()
#                 pool.join()
#                 time_record_solve_SP += time.time() - tmp
#
#                 # change
#                 tmp = time.time()
#                 pool = ThreadPool(processes=ALLOW_CORE)
#                 pool.starmap(self.aux_solve_parallel_primal_change_obj,
#                              [(s, r) for s in self.data.Scenarios[1:] for r in self.data.Trips])
#                 pool.close()
#                 pool.join()
#                 time_update_SP += time.time() - tmp
#
#                 # solve
#                 tmp = time.time()
#                 pool = ThreadPool(processes=ALLOW_CORE)
#                 pool.starmap(self.aux_solve_parallel_primal_optimize,
#                              [(s, r) for s in self.data.Scenarios[1:] for r in self.data.Trips])
#                 pool.close()
#                 pool.join()
#                 time_record_solve_SP += time.time() - tmp
#
#                 # obtain info
#                 pool = ThreadPool(processes=ALLOW_CORE)
#                 result = pool.starmap(self.aux_solve_parallel_primal_retrieval,
#                                       [(s, r, self.use_lp_basis) for s in self.data.Scenarios[1:] for r in
#                                        self.data.Trips])
#                 pool.close()
#                 pool.join()
#
#                 tmp = time.time()
#                 result_dict = {list(item.keys())[0]: item[list(item.keys())[0]] for item in result}
#                 for r in self.data.Trips:
#                     for s in self.data.Scenarios[1:]:
#                         item = result_dict[s, r]
#                         self.varValue.update_Sub_var_primal(s, r,
#                                                             x=item['x'],
#                                                             y=item['y'],
#                                                             pi1=item['pi1'],
#                                                             pi2=item['pi2'],
#                                                             pi3=item['pi3'],
#                                                             pi4=item['pi4'],
#                                                             t=item['t'],
#                                                             obj=item['obj'])  # ,
#                                                             # VBasis=item['VBasis'],
#                                                             # CBasis=item['CBasis'])
#                 time_retrieve_SP_solution += time.time() - tmp
#
#             self.time_update_SP.append(time_update_SP)
#             self.time_record_solve_SP.append(time_record_solve_SP)
#             self.time_retrieve_SP_solution.append(time_retrieve_SP_solution)
#
#         elif self.parallel_method == 'process':
#             time_record_solve_SP = 0
#             time_retrieve_SP_solution = 0
#             tmp = time.time()
#             if (not use_dual_also) or (use_dual_also and n_iters == 1):
#             # use only primal model
#                 # solve scenario 1 and get warm start info
#                 pool = mp.Pool(processes=ALLOW_CORE)
#                 if not revise_dual_value:
#                     if self.use_lp_basis:
#                         # if use lp basis as warm start
#                         result = pool.starmap(self.SP_template_primal.create_and_solve_by_with,
#                                               [(self.data.Scenarios[0], r, self.varValue.z, True, self.SP_VBasis.get(r, None),
#                                                 self.SP_CBasis.get(r, None), True) for r in
#                                                self.data.Trips])
#                     else:
#                         result = pool.starmap(self.SP_template_primal.create_and_solve_by_with,
#                                               [(self.data.Scenarios[0], r, self.varValue.z, False, None, None, True) for r in
#                                                self.data.Trips])
#                 else:
#                     """Revise the value of dual variables"""
#                     # solve primal to get smaller dual var
#                     result = pool.starmap(self.SP_template_primal.create_and_solve_by_with_revise_dual,
#                                           [(self.data.Scenarios[0], r, self.varValue.z) for r in
#                                            self.data.Trips])
#
#                 pool.close()
#                 pool.join()
#                 time_record_solve_SP += time.time() - tmp
#
#                 tmp = time.time()
#                 result_dict = {list(item.keys())[0]: item[list(item.keys())[0]] for item in result}
#                 for r in self.data.Trips:
#                     item = result_dict[self.data.Scenarios[0], r]
#                     if self.use_lp_basis:
#                         VBasis_info = {
#                             'pi1': [item['pi1'][k] for k in self.data.Node_of_trip[r]],
#                             'pi2': [item['pi2'][h, l] for (h, l) in self.data.Hub_pairs],
#                             'pi3': [item['pi3'][h, l] for (h, l) in self.data.Hub_pairs],
#                             'pi4': [item['pi4'][i, j] for (i, j) in self.data.Node_pairs_for_mdl_sp[r]],
#                             't': item['t']
#                         }
#                         self.VBasis_info[r] = VBasis_info.copy()  # VBasis for dual model
#                         self.SP_VBasis[r] = item['VBasis']  # V and C Basis for primal model
#                         self.SP_CBasis[r] = item['CBasis']
#
#                     self.varValue.update_Sub_var_primal(self.data.Scenarios[0], r,
#                                                             x=item['x'],
#                                                             y=item['y'],
#                                                             pi1=item['pi1'],
#                                                             pi2=item['pi2'],
#                                                             pi3=item['pi3'],
#                                                             pi4=item['pi4'],
#                                                             t=item['t'],
#                                                             obj=item['obj'])#,
#                                                             # VBasis=item['VBasis'],
#                                                             # CBasis=item['CBasis'])
#                 time_retrieve_SP_solution += time.time() - tmp
#
#                 # then solve remaining scenarios
#                 tmp = time.time()
#                 pool = mp.Pool(processes=ALLOW_CORE)
#                 if not revise_dual_value:
#                     if self.use_lp_basis:
#                         # if use lp basis as warm start
#                         result = pool.starmap(self.SP_template_primal.create_and_solve_by_with,
#                                               [(s, r, self.varValue.z, True,
#                                                 self.SP_VBasis[r],
#                                                 self.SP_CBasis[r]) for s in self.data.Scenarios[1:] for r in
#                                                self.data.Trips])
#                     else:
#                         result = pool.starmap(self.SP_template_primal.create_and_solve_by_with,
#                                               [(s, r, self.varValue.z) for s in self.data.Scenarios[1:] for r in
#                                                self.data.Trips])
#                 else:
#                     """Revise the value of dual variables"""
#                     result = pool.starmap(self.SP_template_primal.create_and_solve_by_with_revise_dual,
#                                           [(s, r, self.varValue.z) for s in self.data.Scenarios[1:] for r in
#                                            self.data.Trips])
#                 pool.close()
#                 pool.join()
#                 time_record_solve_SP += time.time() - tmp
#
#
#                 tmp = time.time()
#                 result_dict = {list(item.keys())[0]: item[list(item.keys())[0]] for item in result}
#                 for r in self.data.Trips:
#                     for s in self.data.Scenarios[1:]:
#                         item = result_dict[s, r]
#                         self.varValue.update_Sub_var_primal(s, r,
#                                                             x=item['x'],
#                                                             y=item['y'],
#                                                             pi1=item['pi1'],
#                                                             pi2=item['pi2'],
#                                                             pi3=item['pi3'],
#                                                             pi4=item['pi4'],
#                                                             t=item['t'],
#                                                             obj=item['obj'])  # ,
#                                                             # VBasis=item['VBasis'],
#                                                             # CBasis=item['CBasis'])
#
#                 time_retrieve_SP_solution += time.time() - tmp
#
#             else:
#                 # solve scenario[0] first
#                 tmp = time.time()
#                 pool = mp.Pool(processes=ALLOW_CORE)
#                 if self.use_lp_basis:
#                     # if use lp basis as warm start
#                     result = pool.starmap(self.SP_template_dual.create_and_solve_by_with,
#                                           [(self.data.Scenarios[0], r, self.varValue.z, True, None, None,
#                                             self.VBasis_info[r]) for r in self.data.Trips])
#                 else:
#                     result = pool.starmap(self.SP_template_dual.create_and_solve_by_with,
#                                           [(self.data.Scenarios[0], r, self.varValue.z) for r in
#                                            self.data.Trips])
#                 pool.close()
#                 pool.join()
#                 time_record_solve_SP += time.time() - tmp
#
#                 tmp = time.time()
#                 result_dict = {list(item.keys())[0]: item[list(item.keys())[0]] for item in result}
#                 for r in self.data.Trips:
#                     item = result_dict[self.data.Scenarios[0], r]
#                     self.varValue.update_Sub_var_primal(self.data.Scenarios[0], r,
#                                                             x=item['x'],
#                                                             y=item['y'],
#                                                             pi1=item['pi1'],
#                                                             pi2=item['pi2'],
#                                                             pi3=item['pi3'],
#                                                             pi4=item['pi4'],
#                                                             t=item['t'],
#                                                             obj=item['obj'])#,
#                                                             # VBasis=None,
#                                                             # CBasis=None)
#                     if self.use_lp_basis:
#                         self.SP_VBasis[r] = list(item['x'].values()) + list(item['y'].values())
#                 time_retrieve_SP_solution += time.time() - tmp
#
#                 # then solve remaining scenarios
#                 tmp = time.time()
#                 pool = ThreadPool(processes=ALLOW_CORE)
#                 if self.use_lp_basis:
#                     # if use lp basis as warm start
#                     result = pool.starmap(self.SP_template_primal.create_and_solve_by_with,
#                                           [(s, r, self.varValue.z, True, self.SP_VBasis[r],
#                                             None) for s in self.data.Scenarios[1:] for r in
#                                            self.data.Trips])
#                 else:
#                     result = pool.starmap(self.SP_template_primal.create_and_solve_by_with,
#                                           [(s, r, self.varValue.z) for s in self.data.Scenarios[1:] for r in
#                                            self.data.Trips])
#                 pool.close()
#                 pool.join()
#                 time_record_solve_SP += time.time() - tmp
#
#                 tmp = time.time()
#                 result_dict = {list(item.keys())[0]: item[list(item.keys())[0]] for item in result}
#                 for r in self.data.Trips:
#                     for s in self.data.Scenarios[1:]:
#                         item = result_dict[s, r]
#                         self.varValue.update_Sub_var_primal(s, r,
#                                                                 x=item['x'],
#                                                                 y=item['y'],
#                                                                 pi1=item['pi1'],
#                                                                 pi2=item['pi2'],
#                                                                 pi3=item['pi3'],
#                                                                 pi4=item['pi4'],
#                                                                 t=item['t'],
#                                                                 obj=item['obj'])#,
#                                                                 # VBasis=item['VBasis'],
#                                                                 # CBasis=item['CBasis'])
#                 time_retrieve_SP_solution += time.time() - tmp
#
#             self.time_record_solve_SP.append(time_record_solve_SP)
#             self.time_retrieve_SP_solution.append(time_retrieve_SP_solution)
#             self.time_update_SP.append(0)  # no such steps
#
#
#         else:
#             raise 'unknown parallel method in function solve_subproblems_primal()'
#
#     def solve_subproblems_dual(self):
#         # solve the sub problem
#         time_update_SP = 0
#         time_record_solve_SP = 0
#         time_retrieve_SP_solution = 0
#         if self.parallel_method == 'normal':
#             for s in self.data.Scenarios:
#                 for r in self.data.Trips:
#                     tmp = time.time()
#                     self.dict_SP[s, r].update_objective_cons(param_z=self.varValue.z)
#                     time_update_SP += time.time() - tmp
#
#                     tmp = time.time()
#                     if self.use_lp_basis:
#                         self.dict_SP[s, r].write_lp_basis_info(True, self.SP_VBasis.get(r, None), self.SP_CBasis.get(r, None))
#                     else:
#                         self.dict_SP[s, r].write_lp_basis_info()
#                     time_record_solve_SP += time.time() - tmp
#
#                     tmp = time.time()
#                     self.dict_SP[s,r].model.optimize()
#                     time_record_solve_SP += time.time() - tmp
#
#                     tmp = time.time()
#                     result = self.dict_SP[s,r].obtain_solution_info(self.use_lp_basis)
#                     time_retrieve_SP_solution += time.time() - tmp
#
#                     tmp = time.time()
#                     self.varValue.update_Sub_var(s=s, r=r, pi1=result['pi1'], pi2=result['pi2'], pi3=result['pi3'],
#                                                  pi4=result['pi4'], t=result['t'], obj=result['obj'])  # update sub vars for the model under scenario r and trip t
#                     time_retrieve_SP_solution += time.time() - tmp
#
#             self.time_update_SP.append(time_update_SP)
#             self.time_record_solve_SP.append(time_record_solve_SP)
#             self.time_retrieve_SP_solution.append(time_retrieve_SP_solution)
#
#         elif self.parallel_method == 'thread':
#             time_update_SP = 0
#             time_record_solve_SP = 0
#             time_retrieve_SP_solution = 0
#             # solve scenario 1
#             # update
#             tmp = time.time()
#             pool = ThreadPool(processes=ALLOW_CORE)
#             if self.use_lp_basis:
#                 pool.starmap(self.aux_solve_parallel_update,
#                                       [(self.data.Scenarios[0], r, True, self.SP_VBasis.get(r, None),
#                                         self.SP_CBasis.get(r, None), None) for r in self.data.Trips])
#             else:
#                 pool.starmap(self.aux_solve_parallel_update,
#                                       [(self.data.Scenarios[0], r) for r in self.data.Trips])
#             pool.close()
#             pool.join()
#             time_update_SP += time.time() - tmp
#
#             # solve
#             tmp = time.time()
#             pool = ThreadPool(processes=ALLOW_CORE)
#             pool.starmap(self.aux_solve_parallel_optimize,
#                          [(self.data.Scenarios[0], r) for r in self.data.Trips])
#             time_record_solve_SP += time.time() - tmp
#
#             # obtain info
#             tmp = time.time()
#             pool = ThreadPool(processes=ALLOW_CORE)
#             result = pool.starmap(self.aux_solve_parallel_retrieval,
#                          [(self.data.Scenarios[0], r, self.use_lp_basis) for r in self.data.Trips])
#             time_retrieve_SP_solution += time.time() - tmp
#
#             tmp = time.time()
#             result_dict = {list(item.keys())[0]: item[list(item.keys())[0]] for item in result}
#             for r in self.data.Trips:
#                 item = result_dict[self.data.Scenarios[0],r]
#                 self.varValue.update_Sub_var(s=self.data.Scenarios[0], r=r, pi1=item['pi1'], pi2=item['pi2'], pi3=item['pi3'],
#                                              pi4=item['pi4'], t=item['t'], obj=item['obj'])
#                 if self.use_lp_basis:
#                     self.SP_VBasis[r] = item['VBasis']
#                     self.SP_CBasis[r] = item['CBasis']
#             time_retrieve_SP_solution += time.time() - tmp
#
#             # solve remaining scenarios
#             # update
#             tmp = time.time()
#             pool = ThreadPool(processes=ALLOW_CORE)
#             if self.use_lp_basis:
#                 pool.starmap(self.aux_solve_parallel_update,
#                              [(s, r, True, self.SP_VBasis.get(r, None),
#                                self.SP_CBasis.get(r, None), None) for s in self.data.Scenarios[1:] for r in self.data.Trips])
#             else:
#                 pool.starmap(self.aux_solve_parallel_update,
#                              [(s, r) for s in self.data.Scenarios[1:] for r in self.data.Trips])
#             pool.close()
#             pool.join()
#             time_update_SP += time.time() - tmp
#
#             # solve
#             tmp = time.time()
#             pool = ThreadPool(processes=ALLOW_CORE)
#             pool.starmap(self.aux_solve_parallel_optimize,
#                          [(s, r) for s in self.data.Scenarios[1:] for r in self.data.Trips])
#             time_record_solve_SP += time.time() - tmp
#
#             # obtain info
#             tmp = time.time()
#             pool = ThreadPool(processes=ALLOW_CORE)
#             result = pool.starmap(self.aux_solve_parallel_retrieval,
#                                   [(s, r, self.use_lp_basis) for s in self.data.Scenarios[1:] for r in self.data.Trips])
#             time_retrieve_SP_solution += time.time() - tmp
#
#             tmp = time.time()
#             result_dict = {list(item.keys())[0]: item[list(item.keys())[0]] for item in result}
#             for s in self.data.Scenarios[1:]:
#                 for r in self.data.Trips:
#                     item = result_dict[s, r]
#                     self.varValue.update_Sub_var(s=s, r=r, pi1=item['pi1'], pi2=item['pi2'], pi3=item['pi3'],
#                                                  pi4=item['pi4'], t=item['t'], obj=item['obj'])
#             time_retrieve_SP_solution += time.time() - tmp
#
#             self.time_update_SP.append(time_update_SP)
#             self.time_record_solve_SP.append(time_record_solve_SP)
#             self.time_retrieve_SP_solution.append(time_retrieve_SP_solution)
#
#
#         elif self.parallel_method == 'process':
#             # solve scenario 1 and get warm start information
#             time_record_solve_SP = 0
#             time_retrieve_SP_solution = 0
#             tmp = time.time()
#             pool = mp.Pool(processes=ALLOW_CORE)
#             with open(self.console_output_file, 'a') as file:
#                 print('\t\tstart to solve sp in parallel - the first scenario', file=file)
#             if self.use_lp_basis:
#                 # if use lp basis as warm start
#                 result = pool.starmap(self.SP_template.create_and_solve_by_with,
#                                       [(self.data.Scenarios[0], r, self.varValue.z, True, self.SP_VBasis.get(r, None),
#                                         self.SP_CBasis.get(r, None)) for r in
#                                        self.data.Trips])
#             else:
#                 result = pool.starmap(self.SP_template.create_and_solve_by_with,
#                                       [(self.data.Scenarios[0], r, self.varValue.z) for r in
#                                        self.data.Trips])
#             pool.close()
#             pool.join()
#             time_record_solve_SP += time.time() - tmp
#
#             with open(self.console_output_file, 'a') as file:
#                 print('\t\tstart to arrange results from sp', file=file)
#             tmp = time.time()
#             result_dict = {list(item.keys())[0]: item[list(item.keys())[0]] for item in result}
#
#             with open(self.console_output_file, 'a') as file:
#                 print('\t\trecord solutions', file=file)
#             for r in self.data.Trips:
#                 item = result_dict[self.data.Scenarios[0],r]
#                 self.varValue.update_Sub_var(s=self.data.Scenarios[0], r=r, pi1=item['pi1'], pi2=item['pi2'], pi3=item['pi3'],
#                                              pi4=item['pi4'], t=item['t'], obj=item['obj'])
#                 if self.use_lp_basis:
#                     self.SP_VBasis[r] = item['VBasis']
#                     self.SP_CBasis[r] = item['CBasis']
#             time_retrieve_SP_solution += time.time() - tmp
#
#             with open(self.console_output_file, 'a') as file:
#                 print('\t\tstart to solve sp in parallem - scenario >= 2', file=file)
#             # then solve remaining scenarios
#             tmp = time.time()
#             pool = mp.Pool(processes=ALLOW_CORE)
#             if self.use_lp_basis:
#                 # if use lp basis as warm start
#                 result = pool.starmap(self.SP_template.create_and_solve_by_with,
#                                       [(s, r, self.varValue.z, True, self.SP_VBasis[r],
#                                         self.SP_CBasis[r]) for s in self.data.Scenarios[1:] for r in
#                                        self.data.Trips])
#             else:
#                 result = pool.starmap(self.SP_template.create_and_solve_by_with,
#                                       [(s, r, self.varValue.z) for s in self.data.Scenarios[1:] for r in
#                                        self.data.Trips])
#             pool.close()
#             pool.join()
#             time_record_solve_SP += time.time() - tmp
#
#             with open(self.console_output_file, 'a') as file:
#                 print('\t\tarrange results - scenario>=2', file=file)
#             tmp = time.time()
#             result_dict = {list(item.keys())[0]: item[list(item.keys())[0]] for item in result}
#             for s in self.data.Scenarios[1:]:
#                 for r in self.data.Trips:
#                     item = result_dict[s, r]
#                     self.varValue.update_Sub_var(s=s, r=r, pi1=item['pi1'], pi2=item['pi2'], pi3=item['pi3'],
#                                                  pi4=item['pi4'], t=item['t'], obj=item['obj'])
#
#             time_retrieve_SP_solution += time.time() - tmp
#
#
#             self.time_record_solve_SP.append(time_record_solve_SP)
#             self.time_retrieve_SP_solution.append(time_retrieve_SP_solution)
#             self.time_update_SP.append(0)  # no such steps
#
#
#     # def aux_solve_parallel(self, s, r, use_lp_basis=False, VBasis=None, CBasis=None, VBasis_info:dict=None):
#     #     """auxilary function, will be invoked in parallel computing"""
#     #     tmp = time.time()
#     #     self.dict_SP[s, r].update_objective_cons(param_z=self.varValue.z)
#     #     self.time_update_SP[-1] += time.time() - tmp
#     #
#     #     result = self.dict_SP[s, r].solve_model(use_lp_basis=use_lp_basis, VBasis=VBasis, CBasis=CBasis, VBasis_info=VBasis_info)
#     #     self.time_record_solve_SP[-1] += result['t_optimize']
#     #     self.time_update_SP[-1] += result['t_update']
#     #
#     #     return {(s,r): result}
#
#     # def aux_solve_parallel_primal(self, s, r, use_lp_basis=False, VBasis=None, CBasis=None):
#     #     """auxilary function, will be invoked in parallel computing"""
#     #     tmp = time.time()
#     #     self.dict_SP_primal[s, r].update_objective_cons(param_z=self.varValue.z)
#     #     self.time_update_SP[-1] += time.time() - tmp
#     #
#     #     self.dict_SP_primal[s, r].write_lp_basis_info()
#     #
#     #
#     #     self.dict_SP_primal[s, r].model.optimize()
#     #     self.dict_SP_primal[s, r].change_obj()  # change obj to leader params
#     #     self.dict_SP_primal[s, r].model.optimize()
#     #     result_primal = self.dict_SP_primal[s, r].obtain_solution_info()
#     #
#     #     result = self.dict_SP_primal[s, r].solve_model(use_lp_basis=use_lp_basis, VBasis=VBasis, CBasis=CBasis)
#     #     self.time_record_solve_SP[-1] += result['t_optimize']
#     #     self.time_update_SP[-1] += result['t_update']
#     #
#     #     return {(s,r): result}
#
#     def aux_solve_parallel_primal_update(self, s, r, use_lp_basis=False, VBasis=None, CBasis=None):
#         self.dict_SP_primal[s, r].update_objective_cons(param_z=self.varValue.z)
#         self.dict_SP_primal[s, r].write_lp_basis_info(use_lp_basis=use_lp_basis, VBasis=VBasis, CBasis=CBasis)
#
#     def aux_solve_parallel_primal_retrieval(self, s, r, use_lp_basis=False):
#         result = self.dict_SP_primal[s, r].obtain_solution_info(use_lp_basis=use_lp_basis)
#         return {(s, r): result}
#
#     def aux_solve_parallel_primal_optimize(self, s, r):
#         self.dict_SP_primal[s, r].model.optimize()
#
#     def aux_solve_parallel_primal_change_obj(self, s, r):
#         self.dict_SP_primal[s, r].change_obj()
#
#
#     # def aux_solve_parallel_dual(self, s, r, use_lp_basis=False, VBasis=None, CBasis=None, VBasis_info:dict=None):
#     #     """auxilary function, will be invoked in parallel computing"""
#     #     tmp = time.time()
#     #     self.dict_SP_dual[s, r].update_objective_cons(param_z=self.varValue.z)
#     #     self.time_update_SP[-1] += time.time() - tmp
#     #
#     #     result = self.dict_SP_dual[s, r].solve_model(use_lp_basis=use_lp_basis, VBasis=VBasis, CBasis=CBasis, VBasis_info=VBasis_info)
#     #     self.time_record_solve_SP[-1] += result['t_optimize']
#     #     self.time_update_SP[-1] += result['t_update']
#     #
#     #     return {(s,r): result}
#
#     def aux_solve_parallel_dual_update(self, s, r, use_lp_basis=False, VBasis=None, CBasis=None, VBasis_info:dict=None):
#         self.dict_SP_dual[s, r].update_objective_cons(param_z=self.varValue.z)
#         self.dict_SP_dual[s, r].write_lp_basis_info(use_lp_basis=use_lp_basis, VBasis=VBasis, CBasis=CBasis, VBasis_info=VBasis_info)
#
#     def aux_solve_parallel_dual_optimize(self,s,r):
#         self.dict_SP_dual[s, r].model.optimize()
#
#     def aux_solve_parallel_dual_retrieval(self, s, r, use_lp_basis=False):
#         result = self.dict_SP_dual[s, r].obtain_solution_info(use_lp_basis=use_lp_basis)
#         return {(s, r): result}
#
#
#     def aux_solve_parallel_update(self, s, r, use_lp_basis=False, VBasis=None, CBasis=None, VBasis_info:dict=None):
#         self.dict_SP[s, r].update_objective_cons(param_z=self.varValue.z)
#         self.dict_SP[s, r].write_lp_basis_info(use_lp_basis=use_lp_basis, VBasis=VBasis, CBasis=CBasis, VBasis_info=VBasis_info)
#
#     def aux_solve_parallel_optimize(self,s,r):
#         self.dict_SP[s, r].model.optimize()
#
#     def aux_solve_parallel_retrieval(self, s, r, use_lp_basis=False):
#         result = self.dict_SP[s, r].obtain_solution_info(use_lp_basis=use_lp_basis)
#         return {(s, r): result}
#
#     def assign_auxilary_var_lb(self, lb_dict):
#         """
#         add lb for each gamma(r,omega)
#         :return:
#         """
#         for (s,r) in self.data.Sce_Trip:
#             self.MP.var.gamma[s,r].setAttr('lb', lb_dict.get((s,r), 0))
#
#     def start_Benders_iteration(self, timelimit=3600, use_i_ccg=False):
#         """
#         Benders iterations
#         :return:
#         """
#         parser = get_parser()
#         self.args = parser.parse_args()
#         MP_gap_scale = self.args.MP_gap_scale  # decrease MP_gap by a certain scale
#         inexact_gap_threshold = self.args.inexact_gap_threshold  # criteria of exploration/exploitation
#         stop_gap = self.args.stop_gap
#
#
#         n_Benders_iter = 0
#         ub = 1e20
#         best_ub = 1e20
#         lb = -1e20
#         force_exploration = False  # if MIPGap=0, force to exploration
#         force_exploitation = False  # if too much exploration executed, force to exploitation
#         n_explor_repeat = 0  # numer of continuously executed explorations
#         n_exploit_repeat = 0  # numer of continuously executed exploitations
#         start_time = time.time()
#         # MP_solved_to_optimal = False
#         while not ((best_ub - lb)/best_ub <= stop_gap):
#             loop_start_time = time.time()
#             with open(self.console_output_file, 'a') as file:
#                 print(
#                     '\n--------Iteration: {}--------current lb={:.4f}, ub={:.4f}, best_ub={:.4f}, gap={:.4f}%, cal_time={:.2f}/{}--------'.format(
#                         n_Benders_iter, lb, ub, best_ub, (best_ub - lb) / best_ub*100, time.time() - start_time,
#                         self.args.CCG_timelimit), file=file)
#                 # print('mistake caused by numerical issue: {:.4f}'.format(sum(self.varValue.gamma_round.values()) - sum(self.varValue.gamma.values())), file=file)
#             n_Benders_iter += 1
#
#             if self.solve_MP_use_Benders:
#                 self.MP_Benders.update_varValue(varValue=self.varValue)  # input varValue info
#
#             if time.time() - start_time > timelimit:
#                 with open(self.console_output_file, 'a') as file:
#                     print('time limit reached, terminate.', file=file)
#                 break
#
#             with open(self.console_output_file, 'a') as file:
#                 print('solve master...', file=file)
#             tmp = time.time()
#             self.MP.solve_model(use_i_ccg=use_i_ccg, varValue=self.varValue)  # todo: delete the parameter varValue 20250102
#             # MP_solved_to_optimal = self.MP.solved_to_optimal
#             self.time_record_solve_MP.append(time.time() - tmp)
#             # i_ub = sum(self.data.Beta_hl[h, l] * round(self.MP.var.z[h, l].X) for (h, l) in self.data.Hub_pairs) + sum(
#             #     self.data.Scenarios_prob[s] * sum(
#             #         self.data.Trip_p_amount[s,r] * self.MP.var.gamma[s, r].X for r in self.data.Trips) for s in
#             #     self.data.Scenarios)  # inexact upper bound  # inexact upper bound
#             i_ub = self.MP.model.ObjBound  # inexact upper bound  # inexact upper bound
#             # i_lb = self.MP.model.ObjBound  # inexact lower bound
#
#             if self.MP.obj_lb_is_0:  # in exploitation
#                 if i_ub > lb:
#                     lb = i_ub
#             self.LB_record.append(lb)
#
#             tmp = time.time()
#             self.varValue.update_Master_var(var=self.MP.var)  # update z
#             self.time_retrieve_MP_solution.append(time.time() - tmp)
#
#             with open(self.console_output_file, 'a') as file:
#                 print('solve sub...', file=file)
#             if self.solution_appro == 'dual':
#                 self.solve_subproblems_dual()
#             elif self.solution_appro == 'primal':
#                 self.solve_subproblems_primal(n_iters=n_Benders_iter, use_dual_also=self.primal_based_dual_to_ini, revise_dual_value=self.revise_dual_value)
#
#             # TODO: delete 20241227
#             for s,r in self.data.Sce_Trip:
#                 if self.MP.var.gamma[s,r].X > self.varValue.latest_sub_obj[s,r] + 0.1:
#                     with open(self.console_output_file, 'a') as file:
#                         print('>>>>s={}, r={}, gamma={:.4f} is greater than sub_obj={:.4f}, the bigM value is too small!'.format(s,r,self.MP.var.gamma[s, r].X,
#                                                                     self.varValue.latest_sub_obj[s, r]), file=file)
#             # ii=0
#             # for (s, r) in self.data.Sce_Trip:
#             #     print('***s={}, r={}, subobj={:.4f}'.format(s, r, self.varValue.latest_sub_obj[s, r]))
#             #     ii += 1
#             #     if ii > 10:
#             #         break
#             # print('max_pi1={:.4f}'.format(max(self.varValue.pi1.values())))
#             # a = self.varValue.pi1.values()
#             # a = [-i for i in a]
#             # print('min_pi1=-{:.4f}'.format(max(a)))
#             # for (h,l) in self.data.Hub_pairs:
#             #     print('h={}, l={}, pi2={:.4f}'.format(h, l, self.varValue.pi2[h,l]))
#             # print('sum_pi2={:.4f}'.format(sum(self.varValue.pi2.values())))
#
#             with open(self.console_output_file, 'a') as file:
#                 print('calculate ub...', file=file)
#             tmp = time.time()
#             ub = 0
#             for s in self.data.Scenarios:
#                 for r in self.data.Trips:
#                     # update upper bound
#                     ub += self.data.Scenarios_prob[s] * self.data.Trip_p_amount[s,r] * self.varValue.latest_sub_obj[s, r]
#             ub += sum(self.data.Beta_hl[h, l] * round(self.varValue.z[h, l]) for (h, l) in self.data.Hub_pairs)
#             if ub < best_ub:
#                 best_ub = ub
#             self.UB_record.append(best_ub)
#
#             if use_i_ccg:
#                 with open(self.console_output_file, 'a') as file:
#                     print('add cut in i-CCG criteria...', file=file)
#
#                 with open(self.console_output_file, 'a') as file:
#                     print('***********inexact gap = {:.4f}'.format((best_ub - i_ub) / best_ub), file=file)
#                 if ((best_ub - i_ub) / best_ub > inexact_gap_threshold or force_exploration) and not force_exploitation:
#                     force_exploration = False  # if MP is already optimal, go to exploration in the next iter
#                     n_explor_repeat += 1
#                     with open(self.console_output_file, 'a') as file:
#                         print('\t\t i-CCG: exploration', file=file)
#                     # exploration
#                     self.n_cut_in_iter = 0
#                     self.n_combi_cut = 0
#                     for s in self.data.Scenarios:
#                         for r in self.data.Trips:
#                             if not self.MP.agg_cut:
#                                 if self.varValue.gamma[s, r] - self.varValue.latest_sub_obj[s, r] >= -1e-5:
#                                     continue  # when use disaggregated cut and the last cut is not violated, do not add cut again
#
#                             if self.varValue.gamma[s, r] > self.varValue.latest_sub_obj[s, r]+0.1:
#                                 with open(self.console_output_file, 'a') as file:
#                                     print('\t\t\t >>>>s={}, r={}, gamma={:.4f} >= subobj={:.4f}'.format(s,r,self.varValue.gamma[s, r], self.varValue.latest_sub_obj[s, r]), file=file)
#
#                             # if self.varValue.gamma[s,r] < self.varValue.gamma_round[s,r]-0.1:
#                             #     with open(self.console_output_file, 'a') as file:
#                             #         print('\t\t\t <<<<s={}, r={}, gamma={:.4f} <= gamma_round={:.4f}'.format(s, r,
#                             #             self.varValue.gamma[s, r],self.varValue.gamma_round[s, r]),file=file)
#                             #         z_val = list(self.varValue.z.values())
#                             #         z_val = [i for i in z_val if i > 0]
#                             #         print('\t\t\t\t z=', z_val, file=file)
#
#                             self.n_cut_in_iter += 1
#                             self.s_c_cut_stat[s,r] += 1
#                             with open(self.console_output_file, 'a') as file:
#                                 print('\t\t\t s={},r={}, deviation betwwen $zeta$ and sub_obj={}'.format(s, r,
#                                             self.varValue.latest_sub_obj[s, r] - self.varValue.gamma[s, r]), file=file)
#                             # (add cut only when necessary) generate c & c in the master problem
#                             if self.solution_appro == 'dual':
#                                 self.MP.add_var_and_cons_ccg_dual(varValue=self.varValue, s=s, r=r)
#                             elif self.solution_appro == 'primal':
#                                 self.MP.add_var_and_cons_ccg_primal(varValue=self.varValue, s=s, r=r)
#                             else:
#                                 raise 'unknown param solution_appro in function start_Benders_iteration()'
#                             if self.s_c_cut_stat[s,r] > 6:
#                                 # this (s,r) pair probably added cut with too large coefficient, add combinatorial cut for assistance
#                                 self.MP.add_combinatorial_cut(varValue=self.varValue, s=s, r=r)
#                                 self.n_cut_in_iter += 1
#                                 self.n_combi_cut += 1
#                     with open(self.console_output_file, 'a') as file:
#                         print('\t\t\t {} cuts are added, {} of them are combinatorial cuts'.format(self.n_cut_in_iter, self.n_combi_cut), file=file)
#                     self.list_n_cut_in_iter.append(self.n_cut_in_iter)
#                     # provide a lower bound for obj
#                     self.MP.update_obj_lb(obj_lb=i_ub)
#                     self.MP.obj_lb_is_0 = False
#                     # let MIP_gap be larger, recover to the initial value
#                     self.MP.update_MIP_gap(scale=1, to_original=True)
#                     self.MP.MIP_gap = self.MP.original_MIP_gap
#
#                     if n_explor_repeat >= 5:
#                         with open(self.console_output_file, 'a') as file:
#                             print('\t\t There are already 5 exploration, force to exploitation', file=file)
#                         force_exploitation = True
#                         n_explor_repeat = 0
#                     n_exploit_repeat = 0  # set exploitation repeat counter to 0
#                 else:
#                     with open(self.console_output_file, 'a') as file:
#                         print('\t\t i-CCG: exploitation', file=file)
#                     n_exploit_repeat += 1
#                     # exploitation
#                     if force_exploitation:
#                         self.MP.update_MIP_gap(scale=MP_gap_scale * 0.1)  #forcuse more on the MP search,
#                     else:
#                         self.MP.update_MIP_gap(scale=MP_gap_scale)
#                     # self.MP.update_MP_timelimit(dt=MP_timelimit_dt)
#                     # delete the inexact lower bound on obj
#                     self.MP.update_obj_lb(obj_lb=lb)
#                     self.MP.obj_lb_is_0 = True
#                     if self.MP.solved_to_optimal:
#                         with open(self.console_output_file, 'a') as file:
#                             print('\t\t MP is solved to optimal, force next iter to exploration', file=file)
#                         force_exploration = True
#                     force_exploitation = False
#                     n_explor_repeat = 0
#                     # add a condition to force exploration
#                     if n_exploit_repeat >= 5:
#                         with open(self.console_output_file, 'a') as file:
#                             print('\t\t There are already 5 exploitations, force to exploration', file=file)
#                         force_exploration = True
#                         n_exploit_repeat = 0
#
#             else:
#                 with open(self.console_output_file, 'a') as file:
#                     print('add cut in exact CCG criteria...', file=file)
#                 # add cuts
#                 for s in self.data.Scenarios:
#                     for r in self.data.Trips:
#                         if not self.MP.agg_cut:
#                             if self.varValue.gamma[s, r] >= self.varValue.latest_sub_obj[s, r]:
#                                 continue  # when use disaggregated cut and the last cut is not violated, do not add cut again
#                         # (add cut only when necessary) generate c & c in the master problem
#                         if self.solution_appro == 'dual':
#                             self.MP.add_var_and_cons_ccg_dual(varValue=self.varValue, s=s, r=r)
#                         elif self.solution_appro == 'primal':
#                             self.MP.add_var_and_cons_ccg_primal(varValue=self.varValue, s=s, r=r)
#                         else:
#                             raise 'unknown param solution_appro in function start_Benders_iteration()'
#
#             self.MP.update_var_index()  # update the index of master vars
#             self.time_update_MP.append(time.time() - tmp)
#
#             # record time consumption
#             self.time_record.append(time.time() - loop_start_time)
#
#         # TODO 20241107: This calculation seems not necessary, delete it.
#         # if self.solution_appro == 'primal':
#         #     self.solve_subproblems_primal(n_iters=n_Benders_iter, use_dual_also=self.primal_based_dual_to_ini, need_x=True)
#
#         self.record = {'time model creation':self.time_mdl_creation,
#                 'time cost': self.time_record,
#                 'total time cost': sum(self.time_record),
#                 'time cost MP': self.time_record_solve_MP,
#                 'total time cost MP': sum(self.time_record_solve_MP),
#                 'time cost SP': self.time_record_solve_SP,
#                 'total time cost SP': sum(self.time_record_solve_SP),
#                 'time retrieve MP info': self.time_retrieve_MP_solution,
#                 'total time retrieve MP info': sum(self.time_retrieve_MP_solution),
#                 'time retrieve SP info': self.time_retrieve_SP_solution,
#                 'total time retrieve SP info': sum(self.time_retrieve_SP_solution),
#                 'time update MP': self.time_update_MP,
#                 'total time update MP': sum(self.time_update_MP),
#                 'time update SP': self.time_update_SP,
#                 'total time update SP': sum(self.time_update_SP),
#                 'LB record': self.LB_record,
#                 'UB record': self.UB_record,
#                 'gap': (min(self.UB_record) - self.LB_record[-1]) / (min(self.UB_record)+1e-8),
#                 'pure solve time': sum(self.time_record_solve_MP)+sum(self.time_record_solve_SP),
#                 'n iters': n_Benders_iter,
#                 'list_n_cut_in_iter': self.list_n_cut_in_iter
#                        }
#
#     def update_result_ccg(self, result, read_data_time, tag=None):
#         """
#         collect results
#         :param result: the result dictionary
#         :return:
#         """
#         record = self.record
#         result['node-hub-trip-passenger'].append(re.findall(r'\d+', self.data.file_name)[:4])
#         result['tag'].append(tag)
#         result['agg_cut'].append(self.agg_cut)
#         result['arc elim'].append(self.data.arc_elimination)
#         result['Delta type'].append(self.data.Delta_type)
#         result['parallel method'].append(self.parallel_method)
#         result['lp warm start'].append(self.use_lp_basis)
#         result['solution approach'].append(self.solution_appro)
#         result['primal also dual'].append(self.primal_based_dual_to_ini)
#         result['time read data'].append(read_data_time)
#
#         result['time model creation'].append(record['time model creation'])
#         result['solution time'].append(record['time cost'])
#         result['total solution time'].append(record['total time cost'])
#         result['solution time MP'].append(record['time cost MP'])
#         result['total solution time MP'].append(record['total time cost MP'])
#         result['solution time SP'].append(record['time cost SP'])
#         result['total solution time SP'].append(record['total time cost SP'])
#
#         result['time retrieve MP info'].append(record['time retrieve MP info'])
#         result['total time retrieve MP info'].append(record['total time retrieve MP info'])
#         result['time retrieve SP info'].append(record['time retrieve SP info'])
#         result['total time retrieve SP info'].append(record['total time retrieve SP info'])
#         result['time update MP'].append(record['time update MP'])
#         result['total time update MP'].append(record['total time update MP'])
#         result['time update SP'].append(record['time update SP'])
#         result['total time update SP'].append(record['total time update SP'])
#
#         result['LB record'].append(record['LB record'])
#         result['UB record'].append(record['UB record'])
#         result['final_lb'].append(max(record['LB record']))
#         result['final_ub'].append(min(record['UB record']))
#         result['gap'].append(record['gap'])
#         result['pure solve time'].append(record['pure solve time'])
#         result['n iters'].append(record['n iters'])
#         result['n_cut_each_iter'].append(record['list_n_cut_in_iter'])
#
#         return result
#
#     def plot_result(self):
#         plt.plot(self.LB_record, 'b-.')
#         plt.plot(self.UB_record, 'r-*')
#         plt.show()
#
#
#
#     def collect_solution_info(self, solution_type, model_type='normal', real_tao=None, real_gamma=None):
#         """
#         collect the solution information
#         @:param solution_type: 'determ', 'stochastic', 'stochastic_fixed'
#         :return: the information of leader and follower decisions
#         """
#         """Collect Trip Information"""
#         trip_info = {'trip name': [],
#                      'scenario': [],
#                      'solution type': [],
#                      'origination': [],
#                      'destination': [],
#                      'amounts': [],
#                      'income_level': [],
#                      'scenario_prob': [],
#                      'stops': [],
#                      'hubs': [],
#                      'follower cost': [],
#                      'leader cost': []
#                      }
#
#         trip_info['trip name'] = [r for r in self.data.Trips for s in self.data.Scenarios]
#         trip_info['solution type'] = [solution_type for r in self.data.Trips for s in self.data.Scenarios]
#         trip_info['origination'] = [self.data.Trip_origin[r] for r in self.data.Trips for s in self.data.Scenarios]
#         trip_info['destination'] = [self.data.Trip_destination[r] for r in self.data.Trips for s in self.data.Scenarios]
#         trip_info['amounts'] = [self.data.Trip_p_amount[s,r] for r in self.data.Trips for s in self.data.Scenarios]
#         trip_info['income_level'] = [self.data.node_income_level[self.data.Trip_destination[r]] for r in self.data.Trips
#                                      for s in self.data.Scenarios]
#         trip_info['scenario'] = [s for r in self.data.Trips for s in self.data.Scenarios]
#         trip_info['scenario_prob'] = [self.data.Scenarios_prob[s] for r in self.data.Trips for s in self.data.Scenarios]
#
#         trip_info['stops'] = [
#             [[i, j] for (i, j) in self.data.Node_pairs_for_mdl_sp[r] if self.varValue.y.get((s, r, i, j), 0) > 0] for r in
#             self.data.Trips for s in self.data.Scenarios]
#         trip_info['hubs'] = [[[h, l] for (h, l) in self.data.Hub_pairs if self.varValue.x.get((s,r,h,l),0)>0] for r in
#                              self.data.Trips for s in self.data.Scenarios]
#         trip_info['follower cost'] = [self.data.Trip_p_amount[s,r] * np.sum([
#             self.data.tao_follower[h, l, s] * self.varValue.x.get((s, r, h, l),0) for (h, l) in self.data.Hub_pairs]) +
#                                     self.data.Trip_p_amount[s,r] * np.sum([
#             self.data.gamma_follower[i, j, s] * self.varValue.y.get((s, r, i, j),0) for (i, j) in self.data.Node_pairs_for_mdl_sp[r]]) for r in
#                                     self.data.Trips for s in self.data.Scenarios]
#         trip_info['leader cost'] = [self.data.Trip_p_amount[s,r] * np.sum([
#             self.data.tao_leader[h, l, s] * self.varValue.x.get((s, r, h, l),0) for (h, l) in self.data.Hub_pairs]) +
#                                     self.data.Trip_p_amount[s,r] * np.sum([
#             self.data.gamma_leader[i, j, s] * self.varValue.y.get((s, r, i, j),0) for (i, j) in self.data.Node_pairs_for_mdl_sp[r]]) for r in
#                                     self.data.Trips for s in self.data.Scenarios]
#         df_trip_info = pd.DataFrame.from_dict(trip_info, orient='columns')
#
#         """Collect hub Information"""
#         hub_info = {'hub leg': [],
#                     'solution type': [],
#                     'opening cost': [],
#                     'used times_low': [],
#                     'used times_middle': [],
#                     'used times_high': [],
#                     'used times_total': [],
#                     'served customers_low': [],
#                     'served customers_middle': [],
#                     'served customers_high': [],
#                     'served customers_total': [],
#                      }
#         opened_hl_pairs = [(h,l) for (h,l) in self.data.Hub_pairs if self.varValue.z[h, l] > 0.1]
#         hub_info['hub leg'] = [[h, l] for (h, l) in opened_hl_pairs]
#         hub_info['solution type'] = [solution_type for (h, l) in opened_hl_pairs]
#         hub_info['opening cost'] = [self.data.Beta_hl[h, l] for (h, l) in opened_hl_pairs]
#         hub_info['used times_total'] = [
#             np.sum([1 for r in self.data.Trips for s in self.data.Scenarios if self.varValue.x.get((s, r, h, l),0) > 0.1]) for
#             (h, l) in opened_hl_pairs]  # how many times is the hub pari used
#         hub_info['used times_low'] = [np.sum(
#                 [1 for r in self.data.Trips for s in self.data.Scenarios if self.varValue.x.get((s, r, h, l), 0) > 0.1
#                  if self.data.node_income_level[self.data.Trip_destination[r]] == 'Low'])
#             for (h, l) in opened_hl_pairs]  # how many times is the hub pari used
#         hub_info['used times_high'] = [np.sum(
#             [1 for r in self.data.Trips for s in self.data.Scenarios if self.varValue.x.get((s, r, h, l), 0) > 0.1
#              if self.data.node_income_level[self.data.Trip_destination[r]] == 'High'])
#             for (h, l) in opened_hl_pairs]  # how many times is the hub pari used
#         hub_info['used times_middle'] = [np.sum(
#             [1 for r in self.data.Trips for s in self.data.Scenarios if self.varValue.x.get((s, r, h, l), 0) > 0.1
#              if self.data.node_income_level[self.data.Trip_destination[r]] == 'Middle'])
#             for (h, l) in opened_hl_pairs]  # how many times is the hub pari used
#         hub_info['served customers_total'] = [np.sum(
#             [self.data.Trip_p_amount[s,r] for r in self.data.Trips for s in self.data.Scenarios if self.varValue.x.get((s, r, h, l), 0) > 0.1])
#             for (h, l) in opened_hl_pairs]  # how many times is the hub pari used
#         hub_info['served customers_high'] = [np.sum(
#             [self.data.Trip_p_amount[s,r] for r in self.data.Trips for s in self.data.Scenarios if self.varValue.x.get((s, r, h, l), 0) > 0.1
#              if self.data.node_income_level[self.data.Trip_destination[r]] == 'High'])
#             for (h, l) in opened_hl_pairs]  # how many times is the hub pari used
#         hub_info['served customers_low'] = [np.sum(
#             [self.data.Trip_p_amount[s,r] for r in self.data.Trips for s in self.data.Scenarios if
#              self.varValue.x.get((s, r, h, l), 0) > 0.1
#              if self.data.node_income_level[self.data.Trip_destination[r]] == 'Low'])
#             for (h, l) in opened_hl_pairs]  # how many times is the hub pari used
#         hub_info['served customers_middle'] = [np.sum(
#             [self.data.Trip_p_amount[s,r] for r in self.data.Trips for s in self.data.Scenarios if
#              self.varValue.x.get((s, r, h, l), 0) > 0.1
#              if self.data.node_income_level[self.data.Trip_destination[r]] == 'Middle'])
#             for (h, l) in opened_hl_pairs]  # how many times is the hub pari used
#         df_hub_info = pd.DataFrame.from_dict(hub_info, orient='columns')
#
#
#
#
#         # objective information
#         obj_info = {'Facility cost': sum(self.data.Beta_hl[h, l] * self.varValue.z[h, l] for (h, l) in self.data.Hub_pairs),
#                     'Transport cost (leader)':sum(
#             self.data.Scenarios_prob[s] * sum(
#                 self.data.Trip_p_amount[s,r] * (sum(
#                     self.data.tao_leader[h, l, s] * self.varValue.x.get((s, r, h, l),0) for (h, l) in
#                     self.data.Hub_pairs) + sum(
#                     self.data.gamma_leader[i, j, s] * self.varValue.y.get((s, r, i, j),0) for (i, j) in self.data.Node_pairs_for_mdl_sp[r])) for r
#                 in self.data.Trips) for s in
#             self.data.Scenarios),
#                     'Transport cost (follower)':sum(
#             self.data.Scenarios_prob[s] * sum(
#                 self.data.Trip_p_amount[s,r] * (sum(
#                     self.data.tao_follower[h, l, s] * self.varValue.x.get((s, r, h, l),0) for (h, l) in
#                     self.data.Hub_pairs) + sum(
#                     self.data.gamma_follower[i, j, s] * self.varValue.y.get((s, r, i, j), 0) for (i, j) in
#                     self.data.Node_pairs_for_mdl_sp[r])) for r in self.data.Trips) for s in
#             self.data.Scenarios)
#                     }
#         if model_type == 'single_leader':
#             # use the correct single level tao and gamma, otherwise it's the same as the leader
#             obj_info['Transport cost (follower)'] = sum(
#             self.data.Scenarios_prob[s] * sum(
#                 self.data.Trip_p_amount[s,r] * (sum(
#                     real_tao[h, l, s] * self.varValue.x.get((s, r, h, l),0) for (h, l) in
#                     self.data.Hub_pairs) + sum(
#                     real_gamma[i, j, s] * self.varValue.y.get((s, r, i, j), 0) for (i, j) in
#                     self.data.Node_pairs_for_mdl_sp[r])) for r in self.data.Trips) for s in
#             self.data.Scenarios)
#         if model_type == 'single_follower':
#             obj_info['Transport cost (leader)'] = sum(
#             self.data.Scenarios_prob[s] * sum(
#                 self.data.Trip_p_amount[s,r] * (sum(
#                     real_tao[h, l, s] * self.varValue.x.get((s, r, h, l),0) for (h, l) in
#                     self.data.Hub_pairs) + sum(
#                     real_gamma[i, j, s] * self.varValue.y.get((s, r, i, j),0) for (i, j) in self.data.Node_pairs_for_mdl_sp[r])) for r
#                 in self.data.Trips) for s in
#             self.data.Scenarios)
#         obj_info['Total = Facility cost + Transport cost (leader)'] = obj_info['Facility cost'] + obj_info[
#             'Transport cost (leader)']
#
#         return df_trip_info.copy(), df_hub_info.copy(), obj_info.copy()
#
#     def collect_out_sample_solution_info(self, solution_type):
#         # out of sample cost
#         obj_out_sample_info = {'Scenario': self.data.Scenarios,
#                                'solution type':solution_type,
#                                'Facility cost': sum(
#             self.data.Beta_hl[h, l] * self.varValue.z[h, l] for (h, l) in self.data.Hub_pairs),
#                                'Transport cost (leader)': sum(self.data.Scenarios_prob[s]*sum(
#                 self.data.Trip_p_amount[s,r] * (sum(
#                     self.data.tao_leader[h, l, s] * self.varValue.x.get((s, r, h, l),0) for (h, l) in
#                     self.data.Hub_pairs) + sum(
#                     self.data.gamma_leader[i, j, s] * self.varValue.y.get((s, r, i, j), 0) for (i, j) in self.data.Node_pairs_for_mdl_sp[r])) for r
#                 in self.data.Trips) for s in
#             self.data.Scenarios),
#                                'Transport cost (follower)': sum(self.data.Scenarios_prob[s] * sum(
#                                        self.data.Trip_p_amount[s,r] * (sum(
#                                            self.data.tao_follower[h, l, s] * self.varValue.x.get((s, r, h, l),0) for (h, l) in
#                                            self.data.Hub_pairs) + sum(
#                                            self.data.gamma_follower[i, j, s] * self.varValue.y.get((s, r, i, j), 0) for (i, j) in
#                                            self.data.Node_pairs_for_mdl_sp[r])) for r
#                                        in self.data.Trips) for s in
#                                    self.data.Scenarios)
#                                }
#         df_obj_out_sample = pd.DataFrame.from_dict(obj_out_sample_info)
#         df_obj_out_sample['Total = Facility cost + Transport cost (leader)'] = df_obj_out_sample['Facility cost'] + \
#                                                                                df_obj_out_sample[
#                                                                                    'Transport cost (leader)']
#         df_obj_out_sample['Stochastic is better'] = None
#
#         return df_obj_out_sample
