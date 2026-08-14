# -*- coding: utf-8 -*-
"""
  Author: Author
  Email : liusuri@mail.dlut.edu.cn
   Time : 2025/1/6 14:11
Function: This code redefines the cutting plan algorithm in model_parallel.py. Combinatorial cuts are included.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import pandas as pd
from read_data import Modeldata
import time
import json
import argparse
from model_parallel import OneLayerModel
from model_purely_combinatorial import IterateComb


ALLOW_CORE = 16

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--CCG_timelimit", type=int, help="time limit of CCG algo", default=3600*12)
    parser.add_argument("--MP_timelimit", type=int, help="time limit of MP", default=1800)  # spend less time on MP. The i-ccg algo will solve MP repeatedly when required.
    parser.add_argument("--use_i_ccg", type=int, help="whether use i-CCG algo", default=True)
    parser.add_argument("--MP_gap_scale", type=int, help="MIPGap = MIP * scale", default=0.15)
    parser.add_argument("--inexact_gap_threshold", type=int, help="inexact_gap_threshold", default=0.05)
    parser.add_argument("--stop_gap", type=int, help="stop gap CCG", default=1e-5)
    parser.add_argument("--MP_ini_gap", type=int, help="initial gap of MP", default=0.1)

    parser.add_argument("--Single_Level_timelimit", type=int, help="timelimit of sinvle level algo rithm", default=3600)


    return parser


def write_result(file_name=None, result=None):
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


def write_solution(CCG_iterator, file_tag):
    current_directory = os.path.dirname(os.path.abspath(__file__))
    upper_2_dir = os.path.dirname(current_directory)
    data_file = os.path.basename(CCG_iterator.data.file_name)[:-5]
    output_file_prefix = os.path.join(upper_2_dir, 'output', data_file)

    dict_z = {str(key): round(value) for key, value in CCG_iterator.varValue.best_z.items() if
              value}
    with open('{}_tag_{}_z_sol.json'.format(output_file_prefix, file_tag,
                                               ), 'w') as json_file:
        json.dump(dict_z, json_file, indent=4)
    dict_x = {str(key): round(value) for key, value in CCG_iterator.varValue.best_x.items() if
              value}

    with open('{}_tag_{}_x_sol.json'.format(output_file_prefix, file_tag,
                                               ), 'w') as json_file:
        json.dump(dict_x, json_file, indent=4)
    dict_y = {str(key): round(value) for key, value in CCG_iterator.varValue.best_y.items() if
              value}
    with open('{}_tag_{}_y_sol.json'.format(output_file_prefix, file_tag,
                                               ), 'w') as json_file:
        json.dump(dict_y, json_file, indent=4)

    return dict_z, dict_x, dict_y


def write_solution_single_level(model, file_tag):
    current_directory = os.path.dirname(os.path.abspath(__file__))
    upper_2_dir = os.path.dirname(current_directory)
    data_file = os.path.basename(model.data.file_name)[:-5]
    output_file_prefix = os.path.join(upper_2_dir, 'output', data_file)

    dict_z = {str(key): round(value) for key, value in model.varValue.z.items() if
              value}
    with open('{}_tag_{}_z_sol.json'.format(output_file_prefix, file_tag,
                                               ), 'w') as json_file:
        json.dump(dict_z, json_file, indent=4)
    dict_x = {str(key): round(value) for key, value in model.varValue.x.items() if
              value}
    with open('{}_tag_{}_x_sol.json'.format(output_file_prefix, file_tag,
                                               ), 'w') as json_file:
        json.dump(dict_x, json_file, indent=4)
    dict_y = {str(key): round(value) for key, value in model.varValue.y.items() if
              value}
    with open('{}_tag_{}_y_sol.json'.format(output_file_prefix, file_tag,
                                               ), 'w') as json_file:
        json.dump(dict_y, json_file, indent=4)

    return dict_z, dict_x, dict_y

def get_empty_result():
    result = {'file_name': [],
              'node-hub-trip-passenger': [],
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
              'time preprocess': [],
              'min time preprocess': [],
              'max time preprocess': [],
              'min iter preprocess': [],
              'max iter preprocess': [],
              'ave iter preprocess': [],
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
              'MILPGap': [],
              'pure solve time': [],
              'n iters': [],
              'n_cut_each_iter': [],
              'n_node_BnB': [],
              'unexplored_sub_prob': [],
              }
    return result



if __name__ == '__main__':
    result1 = get_empty_result()
    result2 = get_empty_result()
    current_directory = os.path.dirname(os.path.abspath(__file__))
    upper_2_dir = os.path.dirname(current_directory)
    file_path = os.environ.get('ODMTS_DATA_DIR', os.path.join(upper_2_dir, 'data', 'case_study'))
    file_list = os.listdir(file_path)
    file_list_original = [i for i in file_list if '.xlsx' in i]
    file_list = [os.path.join(file_path, f) for f in file_list_original]
    file_list.sort()

    out_sample_folder = os.path.join(upper_2_dir, 'data', 'sample')  # todo

    for file_name in file_list:
        console_output_file = os.path.join(upper_2_dir, 'output', 'console_info_{}.txt'.format(os.path.basename(file_name)[:-5]))
        with open(console_output_file, 'a') as file:
            print('================BEGIN NEW FILE======================', file=file)
            print('execute: ' + file_name, file=file)
        # try:
        for arc_elimination in [True]:
            # for Delta_type in ['MST']:
            Delta_value = 5
            Delta_type = 'MST'
            for theta_given in [None]:  #, 0.0002, 0.0005, 0.001, 0.002, 0.005]:  # TODO: change here! Fix the value to 0.0001 or None. 20250715

                with open(console_output_file, 'a') as file:
                    print('(arc elimination={}, Delta type={}) reading data .....'.format(arc_elimination, Delta_type), file=file)
                tmp = time.time()
                modeldata = Modeldata(file_name=file_name, arc_elimination=arc_elimination, Delta_type=Delta_type, Delta_value=Delta_value, leader_theta_given=theta_given)
                read_data_time = time.time() - tmp  # time for data reading

                """compare uncertainty through C&CG"""
                solution_strategy = 'process'
                parser = get_parser()
                args = parser.parse_args()
                CCG_timelimit = args.CCG_timelimit
                use_i_ccg = args.use_i_ccg
                solve_MP_use_Benders = False  # CCG MP is too large, use Benders to solve it (** only disaggregated cuts are possible!!)

                with open(console_output_file, 'a') as file:
                    print('********************single level MILP model by Gurobi************************', file=file)
                one_lay_mdl = OneLayerModel(data=modeldata, use_KKT=False, use_sos1=False, add_strong_dual=False)
                one_lay_mdl.solve_model()
                df_trip_info_single_leader, df_hub_info_single_leader, obj_info_stoch_single_leader, obj_detail_info_stoch_single_leader = one_lay_mdl.collect_solution_info(
                    solution_type='single_level_leader')
                dict_z_single_leader, _, _ = write_solution_single_level(model=one_lay_mdl, file_tag='single_leader_Delta_{}'.format(Delta_value))
                result1 = one_lay_mdl.update_result_record_ccg_form(result=result1, read_data_time=read_data_time, tag='sigle_level_MILP_Delta_{}'.format(Delta_value))
                one_lay_mdl.calculate_sub_obj()  # calculate leader obj
                write_result(file_name='calculation_info.xlsx', result=result1)

                # deterministic
                with open(console_output_file, 'a') as file:
                    print('********************Solve deterministic model************************', file=file)
                tmp = time.time()
                modeldata_determ = Modeldata(file_name=file_name, arc_elimination=arc_elimination, Delta_type=Delta_type,
                                      Delta_value=Delta_value)
                time_read_determ_data = time.time() - tmp
                modeldata_determ.obtain_average_value()  # obtain average value and convert to a deterministic instance

                CCG_iterator_determ = IterateComb(data=modeldata_determ, agg_cut=False,
                                             solution_appro='primal',
                                             primal_based_dual_to_ini=False,
                                             use_lp_basis=False,
                                             parallel_method='process',
                                             agg_cut_part=False,
                                             solve_MP_use_Benders=False,
                                             revise_dual_value=True,
                                                  n_tree_iter_max=250)
                CCG_iterator_determ.start_Benders_iteration(timelimit=CCG_timelimit, use_i_ccg=True)
                df_trip_info_determ, df_hub_info_determ, obj_info_determ, obj_detail_info_determ = CCG_iterator_determ.collect_solution_info(
                    solution_type='determi')
                dict_z_determ, _, _ = write_solution(CCG_iterator=CCG_iterator_determ, file_tag='determ_Delta_{}'.format(Delta_value))
                result1 = CCG_iterator_determ.update_result_ccg(result=result1,
                                                                  read_data_time=0, tag='deterministic_Delta_{}'.format(Delta_value))
                write_result(file_name='calculation_info.xlsx', result=result1)
                del CCG_iterator_determ

                # stochastic fix z as determ
                with open(console_output_file, 'a') as file:
                    print('********************stochastic fix z as determ************************', file=file)
                CCG_iterator_stoch = IterateComb(data=modeldata, agg_cut=False,
                                             solution_appro='primal',
                                             primal_based_dual_to_ini=False,
                                             use_lp_basis=False,
                                             parallel_method='process',
                                             agg_cut_part=False,
                                             solve_MP_use_Benders=False,
                                             revise_dual_value=True,
                                                 n_tree_iter_max=250)

                # result = {key: [] for key in result}
                CCG_iterator_stoch.MP.fix_z(given_z=dict_z_determ)
                CCG_iterator_stoch.start_Benders_iteration(timelimit=CCG_timelimit, use_i_ccg=use_i_ccg)
                df_trip_info_stoch_fix_z, df_hub_info_stoch_fix_z, obj_info_stoch_fix_z ,obj_detail_info_stoch_fix_z = CCG_iterator_stoch.collect_solution_info(
                    solution_type='stochastic_fix_z')
                dict_z_stoch_fix_z, _, _ = write_solution(CCG_iterator=CCG_iterator_stoch, file_tag='stoch_fix_Delta_{}'.format(Delta_value))
                result1 = CCG_iterator_stoch.update_result_ccg(result=result1,
                                                               read_data_time=read_data_time, tag='stoch_fix_z_as_determ_Delta_{}'.format(Delta_value))
                write_result(file_name='calculation_info.xlsx', result=result1)
                del CCG_iterator_stoch

                # stochastic fix z as single-level
                with open(console_output_file, 'a') as file:
                    print('********************stochastic fix z as single level************************', file=file)
                CCG_iterator_stoch_z_single = IterateComb(data=modeldata, agg_cut=False,
                                                 solution_appro='primal',
                                                 primal_based_dual_to_ini=False,
                                                 use_lp_basis=False,
                                                 parallel_method='process',
                                                 agg_cut_part=False,
                                                 solve_MP_use_Benders=False,
                                                 revise_dual_value=True,
                                                          n_tree_iter_max=250)

                # result = {key: [] for key in result}
                CCG_iterator_stoch_z_single.MP.fix_z(given_z=dict_z_single_leader)
                CCG_iterator_stoch_z_single.start_Benders_iteration(timelimit=CCG_timelimit, use_i_ccg=use_i_ccg)
                df_trip_info_stoch_z_single, df_hub_info_stoch_z_single, obj_info_stoch_z_single, obj_detail_info_stoch_z_single = CCG_iterator_stoch_z_single.collect_solution_info(
                    solution_type='stochastic_z_single')
                dict_z_stoch_z_single, _, _ = write_solution(CCG_iterator=CCG_iterator_stoch_z_single, file_tag='stoch_z_single_Delta_{}'.format(Delta_value))
                result1 = CCG_iterator_stoch_z_single.update_result_ccg(result=result1,
                                                              read_data_time=read_data_time,
                                                              tag='stoch_fix_z_as_single_Delta_{}'.format(Delta_value))
                write_result(file_name='calculation_info.xlsx', result=result1)
                del CCG_iterator_stoch_z_single

                # stochastic
                with open(console_output_file, 'a') as file:
                    print('********************solve stochastic model************************', file=file)
                CCG_iterator_stoch = IterateComb(data=modeldata, agg_cut=False,
                                             solution_appro='primal',
                                             primal_based_dual_to_ini=False,
                                             use_lp_basis=False,
                                             parallel_method='process',
                                             agg_cut_part=False,
                                             solve_MP_use_Benders=False,
                                             revise_dual_value=True,
                                                 n_tree_iter_max=250)

                CCG_iterator_stoch.start_Benders_iteration(timelimit=CCG_timelimit, use_i_ccg=use_i_ccg)
                df_trip_info_stoch, df_hub_info_stoch, obj_info_stoch, obj_detail_info_stoch = CCG_iterator_stoch.collect_solution_info(
                    solution_type='stochastic')
                dict_z_stoch, _, _ = write_solution(CCG_iterator=CCG_iterator_stoch, file_tag='stoch_Delta_{}'.format(Delta_value))
                result1 = CCG_iterator_stoch.update_result_ccg(result=result1,
                                                        read_data_time=read_data_time, tag='stochastic_Delta_{}'.format(Delta_value))
                write_result(file_name='calculation_info.xlsx', result=result1)
                CCG_iterator_stoch.MP.model.write(os.path.join(upper_2_dir, 'output', 'stochastic_master.lp'))
                del CCG_iterator_stoch

                # write result
                with open(console_output_file, 'a') as file:
                    print('*****************write model results*********************', file=file)
                df_trip_info = pd.concat(
                    [df_trip_info_determ, df_trip_info_stoch_fix_z, df_trip_info_stoch_z_single, df_trip_info_stoch, df_trip_info_single_leader])
                df_hub_info = pd.concat(
                    [df_hub_info_determ, df_hub_info_stoch_fix_z, df_hub_info_stoch_z_single, df_hub_info_stoch, df_hub_info_single_leader])
                dict_obj_info = {'determ': obj_info_determ, 'stochastic_fix_z': obj_info_stoch_fix_z, 'stochastic_z_as_single': obj_info_stoch_z_single,
                                 'stochastic': obj_info_stoch, 'single_leader': obj_info_stoch_single_leader}
                df_obj_info = pd.DataFrame.from_dict(dict_obj_info, orient='index')
                df_obj_detail_info = pd.concat([obj_detail_info_determ, obj_detail_info_stoch_fix_z, obj_detail_info_stoch_z_single, obj_detail_info_stoch, obj_detail_info_stoch_single_leader])

                upper_2_dir = os.path.dirname(current_directory)
                file_path = os.path.join(upper_2_dir, 'output',
                                         os.path.basename(file_name)[:-5] + 'opt_solution_info_Delta_{}.xlsx'.format(Delta_value))
                with pd.ExcelWriter(file_path) as writer:
                    df_trip_info.to_excel(excel_writer=writer, sheet_name='trip_info', index=False)
                    df_hub_info.to_excel(excel_writer=writer, sheet_name='hub_info', index=False)
                    df_obj_info.to_excel(excel_writer=writer, sheet_name='obj_info_in_sample', index=True)
                    df_obj_detail_info.to_excel(excel_writer=writer, sheet_name='obj_detail_info_in_sample', index=False)

                if dict_z_stoch == dict_z_determ:
                    with open(console_output_file, 'a') as file:
                        print("Solution in stochastic and determ models are the same. Proceed to the next file.",
                              file=file)
                        print('=====================CURRENT CALCULATION EXITED============================', file=file)
                    continue

                """out of sample scenarios evaluation - use single level"""
                # with open(console_output_file, 'a') as file:
                #     print('********************out of sample scenarios evaluation single level************************', file=file)
                # # out of sample evaluation: use determ
                # with open(console_output_file, 'a') as file:
                #     print('********************out of sample evaluation: use determ************************', file=file)
                #
                # out_sample_file_id = 0
                # df_trip_info_list = []
                # df_hub_info_list = []
                # dict_obj_info = {}
                # df_obj_detail_info_out_sample = pd.DataFrame()
                # for out_sample_file in os.listdir(out_sample_folder):
                #
                #     out_sample_file_full = os.path.join(out_sample_folder, out_sample_file)
                #
                #     modeldata_out_sample = Modeldata(file_name=out_sample_file_full, arc_elimination=arc_elimination,
                #                                      Delta_type=Delta_type, Delta_value=5)
                #     CCG_iterator_out_sample_determ = IterateComb(data=modeldata_out_sample, agg_cut=False,
                #                                  solution_appro='primal',
                #                                  primal_based_dual_to_ini=False,
                #                                  use_lp_basis=False,
                #                                  parallel_method='process',
                #                                  agg_cut_part=False,
                #                                  solve_MP_use_Benders=False,
                #                                  revise_dual_value=True)
                #     CCG_iterator_out_sample_determ.MP.fix_z(given_z=dict_z_determ)
                #     CCG_iterator_out_sample_determ.start_Benders_iteration(timelimit=CCG_timelimit, use_i_ccg=use_i_ccg)
                #     df_trip_info_out_sample_determ, df_hub_info_out_sample_determ, obj_info_out_sample_determ, obj_detail_info_out_sample_determ = CCG_iterator_out_sample_determ.collect_solution_info(
                #         solution_type='out_sample_determ_{}'.format(out_sample_file[:-5]))
                #     result2 = CCG_iterator_out_sample_determ.update_result_ccg(result=result2,
                #                                                    read_data_time=read_data_time, tag='out_sample_determ_{}'.format(out_sample_file[:-5]))
                #     write_result(file_name='calculation_info_out_sample.xlsx', result=result2)
                #     # out of sample evaluation: use stoch
                #     with open(console_output_file, 'a') as file:
                #         print('********************out of sample evaluation: use determ************************', file=file)
                #
                #     CCG_iterator_out_sample_stoch = IterateComb(data=modeldata_out_sample, agg_cut=False,
                #                                  solution_appro='primal',
                #                                  primal_based_dual_to_ini=False,
                #                                  use_lp_basis=False,
                #                                  parallel_method='process',
                #                                  agg_cut_part=False,
                #                                  solve_MP_use_Benders=False,
                #                                  revise_dual_value=True)
                #     CCG_iterator_out_sample_stoch.MP.fix_z(given_z=dict_z_stoch)
                #     CCG_iterator_out_sample_stoch.start_Benders_iteration(timelimit=CCG_timelimit, use_i_ccg=use_i_ccg)
                #     df_trip_info_out_sample_stoch, df_hub_info_out_sample_stoch, obj_info_out_sample_stoch, obj_detail_info_out_sample_stoch = CCG_iterator_out_sample_stoch.collect_solution_info(
                #         solution_type='out_sample_stoch_{}'.format(out_sample_file[:-5]))
                #     write_solution(CCG_iterator=CCG_iterator_out_sample_stoch, file_tag='out_sample_stoch_{}'.format(out_sample_file[:-5]))
                #     result2 = CCG_iterator_out_sample_stoch.update_result_ccg(result=result2,
                #                                                    read_data_time=read_data_time, tag='out_sample_stoch_{}'.format(out_sample_file[:-5]))
                #     write_result(file_name='calculation_info_out_sample.xlsx', result=result2)
                #
                #     # df_trip_info_list.append(df_trip_info_out_sample_determ.copy())
                #     # df_trip_info_list.append(df_trip_info_out_sample_stoch.copy())
                #     # df_hub_info_list.append(df_hub_info_out_sample_determ)
                #     # df_hub_info_list.append(df_hub_info_out_sample_stoch)
                #     dict_obj_info['out_sample_determ_{}'.format(out_sample_file[:-5])] = obj_info_out_sample_determ.copy()
                #     dict_obj_info['out_sample_stoch_{}'.format(out_sample_file[:-5])] = obj_info_out_sample_stoch.copy()
                #
                #     out_sample_file_id += 1
                #
                #     # write result
                #     with open(console_output_file, 'a') as file:
                #         print('*****************write out of sample results*********************', file=file)
                #     upper_2_dir = os.path.dirname(current_directory)
                #     file_path = os.path.join(upper_2_dir, 'output', os.path.basename(file_name)[:-5]+'out_sample.xlsx')
                #     # df_trip_info = pd.concat(df_trip_info_list)
                #     # df_hub_info = pd.concat(df_hub_info_list)
                #     df_obj_info = pd.DataFrame.from_dict(dict_obj_info, orient='index')
                #     df_obj_detail_info_out_sample = pd.concat([df_obj_detail_info_out_sample, obj_detail_info_out_sample_determ, obj_detail_info_out_sample_stoch])
                #     with pd.ExcelWriter(file_path) as writer:
                #         # df_trip_info.to_excel(excel_writer=writer, sheet_name='trip_info', index=False)
                #         # df_hub_info.to_excel(excel_writer=writer, sheet_name='hub_info', index=False)
                #         df_obj_info.to_excel(excel_writer=writer, sheet_name='obj_info_out_sample', index=True)
                #         df_obj_detail_info_out_sample.to_excel(excel_writer=writer, sheet_name='obj_detail_info_out_sample', index=False)

        # except Exception as e:
        #     with open(console_output_file, 'a') as file:
        #         content = file_name + ' :calculate terminated'
        #         print(content, file=file)
        #         print(e, file=file)







