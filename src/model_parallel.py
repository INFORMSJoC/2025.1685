# -*- coding: utf-8 -*-
"""
  Author: Author
  Email : liusuri@mail.dlut.edu.cn
   Time : 2024/11/01 17:22
Function: Model with subproblems solved in parallel, restrict the set N to reduce model size
Remark: 20250827: The following classes are used in other files:
                    a. VarValue
                    b. SubShortestPathWithStrongDualModel
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
import json
import numpy as np
import gc
import psutil
import sys
from pympler import asizeof
import argparse

ALLOW_CORE = 16


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--CCG_timelimit", type=int, help="time limit of CCG algo", default=3600*12)
    parser.add_argument("--MP_timelimit", type=int, help="time limit of MP", default=720)  # spend less time on MP. The i-ccg algo will solve MP repeatedly when required.
    parser.add_argument("--use_i_ccg", type=int, help="whether use i-CCG algo", default=True)
    parser.add_argument("--MP_gap_scale", type=float, help="MIPGap = MIP * scale", default=0.15)
    parser.add_argument("--inexact_gap_threshold", type=float, help="inexact_gap_threshold", default=0.05)
    parser.add_argument("--stop_gap", type=float, help="stop gap CCG", default=1e-5)
    parser.add_argument("--MP_ini_gap", type=float, help="initial gap of MP", default=0.1)


    return parser






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
            model.addConstrs(max_var >= pi2[h,l] for (h,l) in self.data.Hub_pairs)
            model.addConstr(max_var >= t)

            model.setObjective(max_var, sense=GRB.MINIMIZE)
            model.optimize()


            if need_x:
                x = {(h, l): - c_tau.pi for (h, l) in self.data.Hub_pairs}
                x = {key: (0 if value < 0.1 else value) for key,value in x.items()}
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

    def update_Master_var(self, var):
        self.z = {key: round(var.z[key].X) for key in var.z}
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

    def update_Master_theta(self, var):
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

        if y is not None:
            self.x.update({(s, r,) + key: x[key] for key in x})
            self.y.update({(s, r,) + key: y[key] for key in y})

        self.latest_sub_obj.update({(s, r,): obj})

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
        """Fix recorded open arcs to one and all unrecorded arcs to zero.

        ``given_z`` uses string representations of the arc tuples as keys.
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
        if self.add_strong_dual:
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
        if self.add_why_no_need_M:
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

    def solve_model(self, time_limit=3600):
        self.model.setParam('TimeLimit', time_limit)
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

                    """original KKT/strong duality reformulation"""
                    try:
                        # use KKT, use bigM
                        reform_KKT_bigM = OneStageModelFullMultiplierDepends(data=modeldata, use_KKT=True, use_sos1=False, use_bigM=True, add_strong_dual=False)
                        reform_KKT_bigM.solve_model()
                        result_kkt = reform_KKT_bigM.update_result_record(result = result_kkt, read_data_time=read_data_time)
                        write_result_KKT(result_kkt, numerical_result_file)
                        del reform_KKT_bigM


                        # use KKT, use sos1
                        reform_KKT_sos1 = OneStageModelFullMultiplierDepends(data=modeldata, use_KKT=True, use_sos1=True, use_bigM=False, add_strong_dual=False)
                        reform_KKT_sos1.solve_model()
                        result_kkt = reform_KKT_sos1.update_result_record(result=result_kkt, read_data_time=read_data_time)
                        write_result_KKT(result_kkt, numerical_result_file)
                        del reform_KKT_sos1

                        # use strong duality
                        reform_KKT_strongDual = OneStageModelFullMultiplierDepends(data=modeldata, use_KKT=True, use_sos1=False, use_bigM=False, add_strong_dual=True)
                        reform_KKT_strongDual.solve_model()
                        result_kkt = reform_KKT_strongDual.update_result_record(result=result_kkt, read_data_time=read_data_time)
                        write_result_KKT(result_kkt, numerical_result_file)
                        del reform_KKT_strongDual

                    except Exception as e:
                        with open(console_output_file, 'a') as file:
                            content = file_name + ' :calculate terminated'
                            print(content, file=file)
                            print(e, file=file)

        except Exception as e:
            with open(console_output_file, 'a') as file:
                print(file_name, e, ':calculate terminated at the end line.', file=file)
