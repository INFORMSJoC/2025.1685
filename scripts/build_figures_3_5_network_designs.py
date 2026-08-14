# -*- coding: utf-8 -*-
"""
  Author: Author
  Email : liusuri@mail.dlut.edu.cn
   Time : 2024/9/19 11:59
Function: 1) Show the network design on a map or the income patches, optional.
          2) The followers are present in the same color as the hub station that they used.
          3) The direction of hubs are also presented.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import transbigdata as tbd
if os.environ.get('MAPBOX_TOKEN'):
    tbd.set_mapboxtoken(os.environ['MAPBOX_TOKEN'])
tbd.set_imgsavepath(r'./')
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
import json
import re
import matplotlib.image as mpimg
import numpy as np

def plot_map():
    """
    plot the map
    """
    long_min = -83.830515
    long_max = -83.565814
    lat_min = 42.181508
    lat_max = 42.316971
    # 坐标格式为：[左上角纬度，左上角经度，右下角纬度，右下角经度]
    bounds = [lat_max + 0.02, long_min, lat_min, long_max]
    bounds_plot = [bounds[1], bounds[2], bounds[3], bounds[0]]
    zoom_num = 13

    plt.clf()
    tbd.plot_map(plt, bounds=bounds_plot, zoom=zoom_num, style=1)

    cor_offset = 0.04
    x_rect = [long_min-cor_offset, long_max+cor_offset, long_max+cor_offset, long_min-cor_offset]
    y_rect = [lat_min-cor_offset, lat_min-cor_offset, lat_max+cor_offset, lat_max+cor_offset]

    # 使用fill函数填充灰色透明矩形
    plt.fill(x_rect, y_rect, color='whitesmoke', alpha=0.4)

def plot_income_map():
    img_file = '../income_img/income-30color.png'
    image = mpimg.imread(img_file)
    plt.clf()
    plt.imshow(image)


def plot_bus_routine():
    plt.rcParams['font.sans-serif'] = ['times new roman']  # font name
    plt.rcParams['font.size'] = 10  # font size

    """plot opend leg"""
    for key in z_sol:
        if z_sol[key] > 0.1:
            # width = bus_use_time.get(key, 0)/max(bus_use_time.values()) + 0.1
            width = 1  # TODO: change here
            plt.plot([dict_node_lon[key[0]], dict_node_lon[key[1]]],
                     [dict_node_lat[key[0]], dict_node_lat[key[1]]], linewidth=width, color='black',
                     alpha=0.5, linestyle='-')
            x_arrow_start = dict_node_lon[key[0]]
            y_arrow_start = dict_node_lat[key[0]]
            x_arrow_end = dict_node_lon[key[1]]
            y_arrow_end = dict_node_lat[key[1]]

            x_arrow_mid = x_arrow_start + 0.999 * (x_arrow_end - x_arrow_start)
            y_arrow_mid = y_arrow_start + 0.999 * (y_arrow_end - y_arrow_start)
            plt.annotate(
                '', xy=(dict_node_lon[key[1]], dict_node_lat[key[1]]), xytext=(x_arrow_mid, y_arrow_mid),
                arrowprops=dict(arrowstyle='->', color='black', linewidth=1,
                                mutation_scale=12,  # Bigger arrowhead 7 in v1 submission. v2: 11 in large fig, 13 in small fig
                                alpha=1
                                )
            )

    """plot closed bus stops"""
    # for h in list_hub:
    #     if h not in opened_hub:
    #         plt.scatter(dict_node_lon[h], dict_node_lat[h], linewidth=0.5, color='gray',
    #                     marker='o', s=8, alpha=0.95)

    # plot opened bus stops
    for i in range(len(opened_hub)):
        h = opened_hub[i]
        plt.scatter(dict_node_lon[h], dict_node_lat[h], linewidth=0.5, color=hub_set_color[i],
                    marker='^', s=45, alpha=0.95)  # s=10 in v1 submission. V2: 25 in large fig, 30 in small fig.



def plot_trips(scenario=0, income='All'):
    """
    plot trip routines used bus
    :return:
    """
    if income == 'All':
        trip_by_bus = list(set([k[1] for k in x_sol.keys()]))  # trips used bus
    else:
        trip_by_bus = list(set([k[1] for k in x_sol.keys() if dict_node_income[dict_trip_de[k[1]]]==income]))  # trips used bus
    trip_by_bus.sort()


    # use shuttles
    for key in y_sol:
        if key[0]==scenario and key[1] in trip_by_bus:
            if key[2] in opened_hub or key[3] in opened_hub:
                # connect the node to the hub
                # plt.plot([dict_node_lon[key[2]], dict_node_lon[key[3]]],
                #          [dict_node_lat[key[2]], dict_node_lat[key[3]]], linewidth=0.2, color='salmon',
                #          alpha=0.55, linestyle='-.')
                # plot the point that connects hub
                if key[2] not in opened_hub:
                    stop_p = key[2]
                    target_hub = key[3]
                else:
                    stop_p = key[3]
                    target_hub = key[2]
                plt.scatter(dict_node_lon[stop_p], dict_node_lat[stop_p], color=hub_set_color[opened_hub.index(target_hub)],
                            marker='o', s=3.5, linewidths=0.1, alpha=0.95)

    # # plot all stops
    # for n in list_node:
    #     if n not in list_hub:
    #         plt.scatter(dict_node_lon[n], dict_node_lat[n], color='salmon',
    #                     marker='o', s=2, linewidths=0.1, alpha=0.95)


def plot_shuttle_trips(income='All'):
    """
    plot the trips use shuttle only
    :return:
    """
    trip_by_bus = list(set(list_trip) - set([k[1] for k in x_sol.keys()]))  # trips used bus
    if income != 'All':
        trip_by_bus = [r for r in trip_by_bus if dict_node_income[dict_trip_de[r]]==income]  # keep only the asked income level
    trip_by_bus.sort()
    # if len(trip_by_bus) > 30:
    #     trip_by_bus = trip_by_bus[:30]


    colors = plt.get_cmap('tab10').colors  # + plt.get_cmap('tab20b').colors + plt.get_cmap('tab20c').colors
    colors = list(colors) * (700 // len(colors) + 1)  # 确保有足够多的颜色


    x1_cor = [dict_node_lon[dict_trip_ori[r]] for r in trip_by_bus]
    y1_cor = [dict_node_lat[dict_trip_ori[r]] for r in trip_by_bus]

    x3_cor = [dict_node_lon[dict_trip_de[r]] for r in trip_by_bus]
    y3_cor = [dict_node_lat[dict_trip_de[r]] for r in trip_by_bus]

    x2_cor = [i for i in x3_cor]
    y2_cor = [i for i in y1_cor]

    for i in range(len(x1_cor)):
        plt.scatter(x1_cor[i], y1_cor[i], color=colors[i], marker='o', s=2.5, linewidths=0.1, alpha=0.95)
        plt.scatter(x3_cor[i], y3_cor[i], color=colors[i], marker='o', s=2.5, linewidths=0.1, alpha=0.95)
        # plt.plot([x1_cor[i], x3_cor[i]], [y1_cor[i], y3_cor[i]], linewidth=0.2, color=colors[i],
        #          alpha=0.95, linestyle='-.')
        # plt.plot([x2_cor[i], x3_cor[i]], [y2_cor[i], y3_cor[i]], linewidth=0.2, color=colors[i],
        #          alpha=0.95, linestyle='-.')


def plot_one_y_routine(trip_id=None, scenario=0):
    """
    plot trip routines used bus
    :return:
    """
    trip_by_bus = list(set([k[1] for k in x_sol.keys()]))  # trips used bus
    trip_by_bus.sort()
    if trip_id is None:
        trip_id = trip_by_bus[4]
        print(trip_id)

    routine = {k: y_sol[k] for k in y_sol if k[1]==trip_id and k[0]==scenario}
    for key in routine:
        plt.plot([dict_node_lon[key[2]], dict_node_lon[key[3]]],
                 [dict_node_lat[key[2]], dict_node_lat[key[3]]], linewidth=1, color='salmon',
                 alpha=0.55, linestyle='-.')

def save_fig(figure_name):
    fontdict = {'family': 'Times New Roman',
                'size': 8,
                'style': 'italic',
                'color': 'black'
                }
    # title_font = font_manager.FontProperties(family="Times New Roman", size=15)
    # plt.title('Routine', fontproperties=title_font)

    plt.xlabel('Longitude', fontdict=fontdict)
    plt.ylabel('Latitude', fontdict=fontdict)
    plt.axis('off')
    plt.savefig(figure_name, dpi=800, bbox_inches='tight', pad_inches=0)


def convert_cor(lat, lon):
    y_max = 1014
    p1_lat = 42.300887
    p1_lon = -83.781120
    p1_x = 560
    p1_y = y_max - 344

    p2_lat = 42.260959
    p2_lon = -83.603092
    p2_x = 1079
    p2_y = y_max - 502

    x_target = (lon - p1_lon) / (p2_lon - p1_lon) * (p2_x - p1_x) + p1_x
    y_target = (lat - p1_lat) / (p2_lat - p1_lat) * (p2_y - p1_y) + p1_y

    return int(x_target), int(y_target)



if __name__ == '__main__':
    data_folder = ROOT / 'data' / 'case_study'
    result_folder = ROOT / 'results' / 'raw' / 'case_study'
    file_hub = ROOT / 'data' / 'network' / 'cluster-hubs.csv'
    file_trip = ROOT / 'data' / 'network' / 'cluster_trip_area_raw.csv'
    file_node = ROOT / 'data' / 'network' / 'cluster_node_area.csv'

    df_hub = pd.read_csv(file_hub)
    list_hub = df_hub['hub_id'].to_list()  # list of hub

    df_trip = pd.read_csv(file_trip)
    list_trip = df_trip['trip_id'].to_list()  # list of trip
    dict_trip_ori = dict(zip(df_trip['trip_id'], df_trip['start_area_point']))
    dict_trip_de = dict(zip(df_trip['trip_id'], df_trip['end_area_point']))

    df_node = pd.read_csv(file_node)
    list_node = df_node['node_id'].to_list()  # list of node
    dict_node_lat = dict(zip(df_node['node_id'], df_node['node_lat']))
    dict_node_lon = dict(zip(df_node['node_id'], df_node['node_lon']))
    dict_node_income = dict(zip(df_node['node_id'], df_node['node_income_level']))
    dict_node_x = {i: convert_cor(dict_node_lat[i], dict_node_lon[i])[0] for i in list_node}
    dict_node_y = {i: convert_cor(dict_node_lat[i], dict_node_lon[i])[1] for i in list_node}

    for data_file in os.listdir(data_folder):
        for theta in [None]:  # [0.0002, 0.0005, 0.001, 0.002, 0.005]:  # TODO, normally, use None only
            if theta is None:
                data_file_use = data_file
            else:
                data_file_use = data_file[:-5] + str(theta) + '.xlsx'
            file_name = os.path.join(data_folder, data_file)
            out_sample_eval_file = data_file_use[:-5] + 'opt_solution_info_Delta_5.xlsx'
            if not out_sample_eval_file in os.listdir(result_folder):
                continue
            print('plot for file: {}'.format(data_file_use))

            theta_match = re.search(r'theta_([0-9.]+)\.xlsx$', data_file_use)
            theta_value = theta_match.group(1) if theta_match else 'unknown'
            sol_types = ['stoch', 'determ', 'single_leader'] if theta_value == '0.0001' else ['stoch']
            for sol_type in sol_types:
                file_x_sol = os.path.join(result_folder, data_file_use)[:-5]+'_tag_{}_Delta_5_x_sol.json'.format(sol_type)
                file_y_sol = os.path.join(result_folder, data_file_use)[:-5]+'_tag_{}_Delta_5_y_sol.json'.format(sol_type)
                file_z_sol = os.path.join(result_folder, data_file_use)[:-5]+'_tag_{}_Delta_5_z_sol.json'.format(sol_type)
                for use_map in [True]:  # use map or use income map
                    for income_req in ['All']:  # , 'High', 'Low', 'Middle']:
                        os.makedirs(os.path.join(result_folder, 'images'), exist_ok=True)  # output path of images
                        # figure_folder = os.path.join(result_folder, 'images')
                        figure_folder = ROOT / 'output' / 'network_designs'
                        os.makedirs(figure_folder, exist_ok=True)
                        with open(file_x_sol) as f:
                            x_sol = json.load(f)
                            x_sol = {eval(k): v for k,v in x_sol.items() if v > 0.1}
                        with open(file_y_sol) as f:
                            y_sol = json.load(f)
                            y_sol = {eval(k): v for k, v in y_sol.items() if v > 0.1}
                        with open(file_z_sol) as f:
                            z_sol = json.load(f)
                            z_sol = {eval(k): v for k, v in z_sol.items() if v > 0.1}

                        df_samples = pd.read_excel(file_name, sheet_name='SampleSet')  # read file
                        list_Scenarios = list(df_samples['Sample_id'])
                        Scenarios_prob = {i: float(df_samples[df_samples['Sample_id'] == i]['Probability'].iloc[0]) for i in
                                               list_Scenarios}
                        trip_df =  pd.read_excel(file_name, sheet_name='Trip')  # read file
                        Trip_p_amount = dict(zip(trip_df['trip_id'], trip_df['amount']))
                        # calculate the utilization time of each hub
                        bus_use_time = {}
                        for bus_leg in z_sol.keys():
                            used_trips = [(key[0], key[1]) for key in x_sol if key[2]==bus_leg[0] and key[3]==bus_leg[1]]
                            amount = [Scenarios_prob[s]*Trip_p_amount[r] for s,r in used_trips]
                            bus_use_time[bus_leg] = sum(amount)

                        opened_hub = list(set([key[0] for key in z_sol] + [key[1] for key in z_sol]))
                        opened_hub.sort()
                        hub_set_color = ['salmon', 'forestgreen', 'navy', 'deepskyblue', 'darkorange', 'purple', 'deeppink', 'black', 'gold', 'brown', 'green']

                        # if sol_type in ['stoch', 'single_leader']:
                        #     list_sce = list(set([key[0] for key in y_sol.keys()]))
                        # else:
                        #     list_sce = list(set([key[0] for key in y_sol.keys()]))[0]
                        # list_sce.sort()
                        list_sce = [1]  # TODO: delete when results are all correct

                        if use_map:
                            for s in list_sce:
                                # figure_name = '{}-scenario_{}-usemap_{}-soltype_{}-income_{}.jpg'.format(data_file_use[-12:-5], s,
                                #                                                               use_map, sol_type, income_req)
                                figure_name = 'theta_{}-{}.jpg'.format(theta_value, sol_type)
                                figure_name = os.path.join(figure_folder, figure_name)
                                print('plot... ',s)
                                plot_map()
                                plot_bus_routine()
                                plot_trips(scenario=s, income=income_req)
                                # plot_shuttle_trips(income=income_req)
                                # plot_one_y_routine(scenario=s)
                                save_fig(figure_name=figure_name)
                        else:
                            dict_node_lon = dict_node_x
                            dict_node_lat = {i: 1014-dict_node_y[i] for i in dict_node_y}
                            for s in list_sce:
                                figure_name = '{}-scenario_{}-usemap_{}-soltype_{}-income_{}.jpg'.format(data_file_use[-11:-5], s,
                                                                                                       use_map, sol_type,
                                                                                                       income_req)
                                figure_name = os.path.join(figure_folder, figure_name)
                                print('plot... ',s)
                                plot_income_map()
                                plot_bus_routine()
                                # plot_trips(scenario=s, income=income_req)
                                # plot_one_y_routine(scenario=s)
                                save_fig(figure_name=figure_name)





