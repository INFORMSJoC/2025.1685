# -*- coding: utf-8 -*-
"""
  Author: Author
  Email : liusuri@mail.dlut.edu.cn
   Time : 2025/6/10 17:20
Function: 1) obtain the tables in practical analysis: trip duration and that under different \theta
          2) use the raw data and output .xlsx file
          3) output the detailed information of each trip when needed: df_trip_info_raw. Otherwise, output only the aggregated (average) information
          4) change the range of \theta when required
"""
import numpy as np
import pandas as pd
from geopy.distance import geodesic
import os
import ast
import openpyxl
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from read_data import Modeldata

RESULT_FOLDER = ROOT / 'results' / 'raw' / 'case_study'
OUTPUT_FOLDER = ROOT / 'output'


def cal_dist(i, j):
    """
    distance between node i and node j
    :param i:
    :param j:
    :return:
    """
    dist = geodesic((node_lat[i], node_lon[i]), (node_lat[j], node_lon[j])).miles
    return dist

def cal_travel_time(i,j,s):
    dist_ij = cal_dist(i,j)
    speed = data.speed[i,j,s]

    return dist_ij / speed * 60


n_scenario = 10

for theta_in_file in [0.0001, 0.002, 0.005]:
    file_inner_name = 'node725-hub10-tripNfull-scenario10-passenger2586-time2025-07-19-16-54-46'
    stop_cor_file = ROOT / 'data' / 'network' / 'cluster_node_area.csv'
    result_file = RESULT_FOLDER / '{}-theta_{}opt_solution_info_Delta_5.xlsx'.format(file_inner_name, theta_in_file)
    input_file = ROOT / 'data' / 'case_study' / '{}-theta_{}.xlsx'.format(file_inner_name, theta_in_file)

    data = Modeldata(file_name=input_file, Delta_value=5)
    df_stop_cor = pd.read_csv(stop_cor_file)
    node_lat = dict(zip(df_stop_cor['node_id'], df_stop_cor['node_lat']))
    node_lon = dict(zip(df_stop_cor['node_id'], df_stop_cor['node_lon']))
    list_node = list(df_stop_cor['node_id'])


    df_trip_info_raw = pd.read_excel(result_file, sheet_name='trip_info')
    df_trip_info_raw['direct_distance'] = df_trip_info_raw.apply(lambda x: cal_dist(x['origination'], x['destination']), axis=1)
    df_trip_info_raw['amounts'] = df_trip_info_raw.apply(lambda x: x['amounts']*50, axis=1)
    df_trip_info_raw['direct_travel_time'] = df_trip_info_raw.apply(
        lambda x: cal_travel_time(x['origination'], x['destination'], x['scenario']), axis=1)
    x_sol = df_trip_info_raw['hubs'].to_list()
    x_sol = [ast.literal_eval(s) for s in x_sol]
    y_sol = df_trip_info_raw['stops'].to_list()
    y_sol = [ast.literal_eval(s) for s in y_sol]
    record_s = df_trip_info_raw['scenario'].to_list()
    df_trip_info_raw['bus_transfer'] = [len(i) for i in x_sol]

    list_time_cost = []
    list_dist_cost = []

    for item_id in range(len(x_sol)):
        s = record_s[item_id]
        bus_legs = x_sol[item_id]
        shuttle_legs = y_sol[item_id]
        time_cost = 0
        dist_cost = 0
        for arc in bus_legs:
            time_cost += data.Travel_time[arc[0], arc[1], s] + data.S_wait_bus
        for arc in shuttle_legs:
            time_cost += data.Travel_time[arc[0], arc[1], s]
            dist_cost += data.g_cost_shuttle * data.Dist[arc[0], arc[1]]
        list_time_cost.append(time_cost)
        list_dist_cost.append(dist_cost)

    df_trip_info_raw['time_cost'] = list_time_cost
    df_trip_info_raw['dist_cost'] = list_dist_cost


    obj_detail_info = {
        'file_name': [],
        'solution_type': [],
        'income': [],
        'transport_mode': [],
        'n_trips': [],
        'n_amount': [],
        'model_distance': [],
        'direct_distance': [],
        'model_travel_time': [],
        'direct_travel_time': [],
        'bus_transfer': [],
    }

    for sol_type in ['stochastic', 'determi', 'stochastic_fix_z', 'stochastic_z_single', 'single_level_leader']:
        df_trip_info = df_trip_info_raw[df_trip_info_raw['solution type'] == sol_type]
        for income in ['Low', 'Middle', 'High', 'all']:
            for trans_mode in ['shuttle', 'shuttle+bus', 'all']:
                print(trans_mode)
                obj_detail_info['file_name'].append(os.path.basename(result_file)[:-5])
                obj_detail_info['solution_type'].append(sol_type)
                obj_detail_info['income'].append(income)
                obj_detail_info['transport_mode'].append(trans_mode)
                # select the targeted income level
                if income != 'all':
                    df = df_trip_info[df_trip_info['income_level'] == income]
                else:
                    df = df_trip_info
                # select the targeted transportation mode
                if trans_mode == 'shuttle':
                    df = df[df['hubs'] == '[]']
                elif trans_mode == 'shuttle+bus':
                    df = df[df['hubs'] != '[]']
                else:
                    df = df

                obj_detail_info['n_trips'].append(sum(df['scenario_prob']))
                obj_detail_info['n_amount'].append((df['amounts']*df['scenario_prob']).sum())
                obj_detail_info['direct_distance'].append((df['amounts']*df['direct_distance']*df['scenario_prob']).sum() / (df['scenario_prob'] * df['amounts']).sum())
                obj_detail_info['direct_travel_time'].append((df['amounts']*df['direct_travel_time']*df['scenario_prob']).sum() / (df['scenario_prob'] * df['amounts']).sum())
                obj_detail_info['model_distance'].append((df['amounts']*df['follower travel distance']*df['scenario_prob']).sum() / (df['scenario_prob'] * df['amounts']).sum())
                obj_detail_info['model_travel_time'].append((df['amounts']*df['follower travel time']*df['scenario_prob']).sum() / (df['scenario_prob'] * df['amounts']).sum() / 60)
                obj_detail_info['bus_transfer'].append((df['amounts']*df['bus_transfer']*df['scenario_prob']).sum() / (df['scenario_prob'] * df['amounts']).sum())


    df_obj_info = pd.DataFrame.from_dict(obj_detail_info)

    obj_dist_time = {
        'model': [],
        'time_cost': [],
        'dist_cost': [],
    }

    for sol_type in ['stochastic', 'determi', 'stochastic_fix_z', 'stochastic_z_single', 'single_level_leader']:
        obj_dist_time['model'].append(sol_type)
        df = df_trip_info_raw[df_trip_info_raw['solution type'] == sol_type]
        obj_dist_time['time_cost'].append(
            (df['amounts'] * df['time_cost'] * df['scenario_prob']).sum())
        obj_dist_time['dist_cost'].append(
            (df['amounts'] * df['dist_cost'] * df['scenario_prob']).sum())
    df_obj_dist_time = pd.DataFrame.from_dict(obj_dist_time)

    output_path = OUTPUT_FOLDER / '{}dist_time_info.xlsx'.format(result_file.stem)
    workbook = openpyxl.Workbook()  # create such a file
    workbook.save(output_path)  # save file
    writer = pd.ExcelWriter(output_path, mode='a', engine='openpyxl', if_sheet_exists='replace')  # excel writer
    df_obj_info.to_excel(excel_writer=writer, sheet_name='obj_info', index=False)
    df_obj_dist_time.to_excel(excel_writer=writer, sheet_name='obj_dist_time', index=False)
    writer.close()





