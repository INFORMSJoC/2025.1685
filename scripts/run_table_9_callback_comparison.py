# -*- coding: utf-8 -*-
"""
  Author: Author
  Email : liusuri@mail.dlut.edu.cn
   Time : 2026/2/14 15:33
Function: 
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from run_tables_6_8_case_study import *


if __name__ == '__main__':

    current_directory = os.path.dirname(os.path.abspath(__file__))
    upper_2_dir = os.path.dirname(current_directory)
    file_path = os.environ.get('ODMTS_DATA_DIR', os.path.join(upper_2_dir, 'data', 'table_5_large_instances'))
    file_list = os.listdir(file_path)
    file_list_original = [i for i in file_list if '.xlsx' in i]
    file_list = [os.path.join(file_path, f) for f in file_list_original]
    file_list.sort()

    for file_name in file_list:
        print('calculate:', file_name)
        console_output_file = os.path.join(upper_2_dir, 'output', 'console_info_{}.txt'.format(os.path.basename(file_name)[:-5]))
        if os.path.basename(console_output_file) in os.listdir(os.path.join(upper_2_dir, 'output')):
            print('this file is calculated: ', file_name)
            continue
        result1 = get_empty_result()
        with open(console_output_file, 'a') as file:
            print('================BEGIN NEW FILE======================', file=file)
            print('execute: ' + file_name, file=file)
        current_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        out_xlsx_file = 'calculation_info_{}.xlsx'.format(current_time)
        # try:
        for arc_elimination in [False]:
            for Delta_type in ['MST']:
                with open(console_output_file, 'a') as file:
                    print('(arc elimination={}, Delta type={}) reading data .....'.format(arc_elimination, Delta_type), file=file)
                run_largest_ccg = False
                tmp = time.time()
                modeldata = Modeldata(file_name=file_name, arc_elimination=arc_elimination, Delta_type=Delta_type, Delta_value=5)
                read_data_time = time.time() - tmp  # time for data reading
                """compare uncertainty through C&CG"""
                solution_strategy = 'process'
                parser = get_parser()
                args = parser.parse_args()
                CCG_timelimit = 3600
                use_i_ccg = args.use_i_ccg
                solve_MP_use_Benders = False  # CCG MP is too large, use Benders to solve it (** only disaggregated cuts are possible!!)
                # Compare C&CG
                # with open(console_output_file, 'a') as file:
                #     print('********************inexact C&CG************************', file=file)
                # CCG_iterator_stoch = IterateComb(data=modeldata, agg_cut=False,
                #                                  solution_appro='primal',
                #                                  primal_based_dual_to_ini=False,
                #                                  use_lp_basis=False,
                #                                  parallel_method='process_1_core',
                #                                  agg_cut_part=False,
                #                                  solve_MP_use_Benders=False,
                #                                  revise_dual_value=False,
                #                                  n_tree_iter_max=0,
                #                                  )
                # CCG_iterator_stoch.start_Benders_iteration(timelimit=CCG_timelimit, use_i_ccg=use_i_ccg)
                # CCG_iterator_stoch.update_cal_infor(n_Benders_iter=None)
                # result1 = CCG_iterator_stoch.update_result_ccg(result=result1,
                #                                                read_data_time=read_data_time, tag='inexactCCG')
                # write_result(file_name=out_xlsx_file, result=result1)
                # del CCG_iterator_stoch
                #
                # # Compare C&CG
                # with open(console_output_file, 'a') as file:
                #     print('********************inexact C&CG-tree100************************', file=file)
                # CCG_iterator_stoch = IterateComb(data=modeldata, agg_cut=False,
                #                                  solution_appro='primal',
                #                                  primal_based_dual_to_ini=False,
                #                                  use_lp_basis=False,
                #                                  parallel_method='process',
                #                                  agg_cut_part=False,
                #                                  solve_MP_use_Benders=False,
                #                                  revise_dual_value=False,
                #                                  n_tree_iter_max=100,
                #                                  )
                # CCG_iterator_stoch.start_Benders_iteration(timelimit=CCG_timelimit, use_i_ccg=use_i_ccg)
                # CCG_iterator_stoch.update_cal_infor(n_Benders_iter=None)
                # result1 = CCG_iterator_stoch.update_result_ccg(result=result1,
                #                                                read_data_time=read_data_time, tag='inexactCCG-tree100')
                # write_result(file_name=out_xlsx_file, result=result1)
                # del CCG_iterator_stoch
                #
                # # Compare C&CG
                # with open(console_output_file, 'a') as file:
                #     print('********************inexact C&CG-tree100-revise dual************************', file=file)
                # CCG_iterator_stoch = IterateComb(data=modeldata, agg_cut=False,
                #                                  solution_appro='primal',
                #                                  primal_based_dual_to_ini=False,
                #                                  use_lp_basis=False,
                #                                  parallel_method='process',
                #                                  agg_cut_part=False,
                #                                  solve_MP_use_Benders=False,
                #                                  revise_dual_value=True,
                #                                  n_tree_iter_max=100,
                #                                  )
                # CCG_iterator_stoch.start_Benders_iteration(timelimit=CCG_timelimit, use_i_ccg=use_i_ccg)
                # CCG_iterator_stoch.update_cal_infor(n_Benders_iter=None)
                # result1 = CCG_iterator_stoch.update_result_ccg(result=result1,
                #                                                read_data_time=read_data_time, tag='inexactCCG-tree100-revdual')
                # write_result(file_name=out_xlsx_file, result=result1)
                # del CCG_iterator_stoch



                # # stochastic - tree search (1000 iters) with CCG, no enhancement strategies
                # with open(console_output_file, 'a') as file:
                #     print('********************Classi C&CG************************', file=file)
                # CCG_iterator_stoch = IterateComb(data=modeldata,
                #                                  solution_appro='primal',
                #                                  parallel_method='process_1_core',
                #                                  revise_dual_value=False,  # need to determine
                #                                  n_tree_iter_max=0,  # need to determine
                #                                  )
                # CCG_iterator_stoch.start_Benders_iteration_classic(timelimit=CCG_timelimit, use_i_ccg=False)
                # result1 = CCG_iterator_stoch.update_result_ccg(result=result1,
                #                                         read_data_time=read_data_time, tag='classicCCG')
                # write_result(file_name=out_xlsx_file, result=result1)
                # del CCG_iterator_stoch

                # # stochastic - tree search (1000 iters) with CCG, no enhancement strategies
                # with open(console_output_file, 'a') as file:
                #     print('********************Classi C&CG-tree100************************', file=file)
                # CCG_iterator_stoch = IterateComb(data=modeldata,
                #                                  solution_appro='primal',
                #                                  parallel_method='process',
                #                                  revise_dual_value=False,  # need to determine
                #                                  n_tree_iter_max=100,  # need to determine
                #                                  )
                # CCG_iterator_stoch.start_Benders_iteration_classic(timelimit=CCG_timelimit, use_i_ccg=False)
                # result1 = CCG_iterator_stoch.update_result_ccg(result=result1,
                #                                         read_data_time=read_data_time, tag='classicCCG-tree100')
                # write_result(file_name=out_xlsx_file, result=result1)
                # del CCG_iterator_stoch
                #
                # # stochastic - tree search (1000 iters) with CCG, no enhancement strategies
                # with open(console_output_file, 'a') as file:
                #     print('********************Classi C&CG-tree100 revise dual************************', file=file)
                # CCG_iterator_stoch = IterateComb(data=modeldata,
                #                                  solution_appro='primal',
                #                                  parallel_method='process',
                #                                  revise_dual_value=True,  # need to determine
                #                                  n_tree_iter_max=100,  # need to determine
                #                                  )
                # CCG_iterator_stoch.start_Benders_iteration_classic(timelimit=CCG_timelimit, use_i_ccg=False)
                # result1 = CCG_iterator_stoch.update_result_ccg(result=result1,
                #                                         read_data_time=read_data_time, tag='classicCCG-tree100-revdual')
                # write_result(file_name=out_xlsx_file, result=result1)
                # del CCG_iterator_stoch

                # stochastic - tree search (1000 iters) with CCG, no enhancement strategies
                with open(console_output_file, 'a') as file:
                    print('********************Callback C&CG************************', file=file)
                CCG_iterator_stoch = IterateComb(data=modeldata,
                                                 solution_appro='primal',
                                                 parallel_method='process_1_core',
                                                 revise_dual_value=False,  # need to determine
                                                 n_tree_iter_max=0,  # need to determine
                                                 )
                CCG_iterator_stoch.start_Benders_iteration_callback(timelimit=CCG_timelimit, use_i_ccg=False)
                result1 = CCG_iterator_stoch.update_result_ccg(result=result1,
                                                        read_data_time=read_data_time, tag='callbackCCG')
                write_result(file_name=out_xlsx_file, result=result1)
                del CCG_iterator_stoch

                # stochastic - tree search (1000 iters) with CCG, no enhancement strategies
                with open(console_output_file, 'a') as file:
                    print('********************Callback C&CG-tree100************************', file=file)
                CCG_iterator_stoch = IterateComb(data=modeldata,
                                                 solution_appro='primal',
                                                 parallel_method='process_1_core',
                                                 revise_dual_value=False,  # need to determine
                                                 n_tree_iter_max=100,  # need to determine
                                                 )
                CCG_iterator_stoch.start_Benders_iteration_callback(timelimit=CCG_timelimit, use_i_ccg=False)
                result1 = CCG_iterator_stoch.update_result_ccg(result=result1,
                                                        read_data_time=read_data_time, tag='callbackCCG-tree100')
                write_result(file_name=out_xlsx_file, result=result1)
                del CCG_iterator_stoch

                # # stochastic - tree search (1000 iters) with CCG, no enhancement strategies
                # with open(console_output_file, 'a') as file:
                #     print('********************Callback C&CG-tree100 revise dual************************', file=file)
                # CCG_iterator_stoch = IterateComb(data=modeldata,
                #                                  solution_appro='primal',
                #                                  parallel_method='process_1_core',
                #                                  revise_dual_value=True,  # need to determine
                #                                  n_tree_iter_max=100,  # need to determine
                #                                  )
                # CCG_iterator_stoch.start_Benders_iteration_callback(timelimit=CCG_timelimit, use_i_ccg=False)
                # result1 = CCG_iterator_stoch.update_result_ccg(result=result1,
                #                                         read_data_time=read_data_time, tag='callbackCCG-tree100-revisedual')
                # write_result(file_name=out_xlsx_file, result=result1)
                # del CCG_iterator_stoch




        # except Exception as e:
        #     with open(console_output_file, 'a') as file:
        #         content = file_name + ' :calculate terminated'
        #         print(content, file=file)
        #         print(e, file=file)
