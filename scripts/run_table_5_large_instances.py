# -*- coding: utf-8 -*-
"""
  Author: Author
  Email : liusuri@mail.dlut.edu.cn
   Time : 2025/6/10 12:07
Function: Run to compare the performance of tree+ccg on large-sized instances
"""


import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from run_tables_6_8_case_study import *

if __name__ == '__main__':
    result1 = get_empty_result()
    current_directory = os.path.dirname(os.path.abspath(__file__))
    upper_2_dir = os.path.dirname(current_directory)
    file_path = os.environ.get('ODMTS_DATA_DIR', os.path.join(upper_2_dir, 'data', 'table_5_large_instances'))
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
        try:
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

                    # Cuts: Algorithm 3 initialized without response-search constraints.
                    cuts_solver = IterateComb(
                        data=modeldata,
                        agg_cut=False,
                        solution_appro='primal',
                        primal_based_dual_to_ini=False,
                        use_lp_basis=False,
                        parallel_method='process_1_core',
                        agg_cut_part=False,
                        solve_MP_use_Benders=False,
                        revise_dual_value=False,
                        n_tree_iter_max=0,
                    )
                    cuts_solver.start_Benders_iteration(timelimit=CCG_timelimit, use_i_ccg=False)
                    result1 = cuts_solver.update_result_ccg(
                        result=result1, read_data_time=read_data_time, tag='Cuts')
                    write_result(file_name='calculation_info.xlsx', result=result1)
                    del cuts_solver

                    # stochastic - tree search (20 iters) with CCG
                    with open(console_output_file, 'a') as file:
                        print('********************tree search with 10000 iterations************************', file=file)
                    CCG_iterator_stoch = IterateComb(data=modeldata, agg_cut=False,
                                                 solution_appro='primal',
                                                 primal_based_dual_to_ini=False,
                                                 use_lp_basis=False,
                                                 parallel_method='process',
                                                 agg_cut_part=False,
                                                 solve_MP_use_Benders=False,
                                                 revise_dual_value=True,
                                                 n_tree_iter_max=10000,
                                                     )

                    sol_info = CCG_iterator_stoch.solve_uncompleted_MILP()  # solve the incompleted MILP and obtain UB and LB for the original problem
                    CCG_iterator_stoch.update_cal_infor(n_Benders_iter=0)
                    CCG_iterator_stoch.record.update(sol_info)

                    result1 = CCG_iterator_stoch.update_result_ccg(result=result1,
                                                                   read_data_time=read_data_time,
                                                                   tag='tree_search-10000-MILP')
                    write_result(file_name='calculation_info.xlsx', result=result1)
                    if result1['gap'][-1] < 0.1:
                        continue
                    del CCG_iterator_stoch

                    # stochastic - tree search (1000 iters) with CCG, no enhancement strategies
                    with open(console_output_file, 'a') as file:
                        print('********************tree search with 100 iterations************************', file=file)
                    CCG_iterator_stoch = IterateComb(data=modeldata, agg_cut=False,
                                                     solution_appro='primal',
                                                     primal_based_dual_to_ini=False,
                                                     use_lp_basis=False,
                                                     parallel_method='process',
                                                     agg_cut_part=False,
                                                     solve_MP_use_Benders=False,
                                                     revise_dual_value=False,
                                                     n_tree_iter_max=100,
                                                     )

                    CCG_iterator_stoch.start_Benders_iteration(timelimit=CCG_timelimit, use_i_ccg=False)
                    result1 = CCG_iterator_stoch.update_result_ccg(result=result1,
                                                            read_data_time=read_data_time, tag='tree-100-exactCCG')
                    write_result(file_name='calculation_info.xlsx', result=result1)
                    del CCG_iterator_stoch

                    # stochastic - tree search (100 iters) with CCG
                    with open(console_output_file, 'a') as file:
                        print('********************tree search with 100 iterations************************', file=file)
                    CCG_iterator_stoch = IterateComb(data=modeldata, agg_cut=False,
                                                 solution_appro='primal',
                                                 primal_based_dual_to_ini=False,
                                                 use_lp_basis=False,
                                                 parallel_method='process',
                                                 agg_cut_part=False,
                                                 solve_MP_use_Benders=False,
                                                 revise_dual_value=True,
                                                 n_tree_iter_max=100,
                                                     )

                    CCG_iterator_stoch.start_Benders_iteration(timelimit=CCG_timelimit, use_i_ccg=False)
                    result1 = CCG_iterator_stoch.update_result_ccg(result=result1,
                                                            read_data_time=read_data_time, tag='tree-100-dual_revise')
                    write_result(file_name='calculation_info.xlsx', result=result1)
                    del CCG_iterator_stoch

                    # # stochastic - tree search (100 iters) with CCG
                    # with open(console_output_file, 'a') as file:
                    #     print('********************tree search with 100 iterations************************', file=file)
                    # CCG_iterator_stoch = IterateComb(data=modeldata, agg_cut=False,
                    #                              solution_appro='primal',
                    #                              primal_based_dual_to_ini=False,
                    #                              use_lp_basis=False,
                    #                              parallel_method='process',
                    #                              agg_cut_part=False,
                    #                              solve_MP_use_Benders=False,
                    #                              revise_dual_value=True,
                    #                              n_tree_iter_max=100,
                    #                                  )
                    #
                    # CCG_iterator_stoch.start_Benders_iteration(timelimit=CCG_timelimit, use_i_ccg=True)
                    # result1 = CCG_iterator_stoch.update_result_ccg(result=result1,
                    #                                         read_data_time=read_data_time, tag='tree-100-dual_revise_inexact_ccg')
                    # write_result(file_name='calculation_info.xlsx', result=result1)
                    # del CCG_iterator_stoch

                    # # stochastic - tree search (1000 iters) with CCG
                    # with open(console_output_file, 'a') as file:
                    #     print('********************tree search with 1000 iterations************************', file=file)
                    # CCG_iterator_stoch = IterateComb(data=modeldata, agg_cut=False,
                    #                              solution_appro='primal',
                    #                              primal_based_dual_to_ini=False,
                    #                              use_lp_basis=False,
                    #                              parallel_method='process',
                    #                              agg_cut_part=False,
                    #                              solve_MP_use_Benders=False,
                    #                              revise_dual_value=True,
                    #                              n_tree_iter_max=1000,
                    #                                  )
                    #
                    # CCG_iterator_stoch.start_Benders_iteration(timelimit=CCG_timelimit, use_i_ccg=use_i_ccg)
                    # result1 = CCG_iterator_stoch.update_result_ccg(result=result1,
                    #                                         read_data_time=read_data_time, tag='tree-1000-inexactCCG')
                    # write_result(file_name='calculation_info.xlsx', result=result1)
                    # del CCG_iterator_stoch
                    #
                    # # stochastic - tree search (100 iters) with CCG, no enhancement strategies
                    # with open(console_output_file, 'a') as file:
                    #     print('********************tree search with 1000 iterations************************', file=file)
                    # CCG_iterator_stoch = IterateComb(data=modeldata, agg_cut=False,
                    #                                  solution_appro='primal',
                    #                                  primal_based_dual_to_ini=False,
                    #                                  use_lp_basis=False,
                    #                                  parallel_method='process',
                    #                                  agg_cut_part=False,
                    #                                  solve_MP_use_Benders=False,
                    #                                  revise_dual_value=False,
                    #                                  n_tree_iter_max=1000,
                    #                                  )
                    #
                    # CCG_iterator_stoch.start_Benders_iteration(timelimit=CCG_timelimit, use_i_ccg=False)
                    # result1 = CCG_iterator_stoch.update_result_ccg(result=result1,
                    #                                         read_data_time=read_data_time, tag='tree-1000-exactCCG')
                    # write_result(file_name='calculation_info.xlsx', result=result1)
                    # del CCG_iterator_stoch

                    # # stochastic - tree search (100 iters) with CCG
                    # with open(console_output_file, 'a') as file:
                    #     print('********************tree search with 10000 iterations************************', file=file)
                    # CCG_iterator_stoch = IterateComb(data=modeldata, agg_cut=False,
                    #                              solution_appro='primal',
                    #                              primal_based_dual_to_ini=False,
                    #                              use_lp_basis=False,
                    #                              parallel_method='process',
                    #                              agg_cut_part=False,
                    #                              solve_MP_use_Benders=False,
                    #                              revise_dual_value=True,
                    #                              n_tree_iter_max=10000,
                    #                                  )
                    #
                    # CCG_iterator_stoch.start_Benders_iteration(timelimit=CCG_timelimit, use_i_ccg=use_i_ccg)
                    # result1 = CCG_iterator_stoch.update_result_ccg(result=result1,
                    #                                         read_data_time=read_data_time, tag='tree_search_CCG-10000')
                    # write_result(file_name='calculation_info.xlsx', result=result1)
                    # del CCG_iterator_stoch
                    #
                    # # stochastic - tree search (100 iters) with CCG
                    # if run_largest_ccg:
                    #     with open(console_output_file, 'a') as file:
                    #         print('********************tree search with 100000 iterations************************', file=file)
                    #     CCG_iterator_stoch = IterateComb(data=modeldata, agg_cut=False,
                    #                                  solution_appro='primal',
                    #                                  primal_based_dual_to_ini=False,
                    #                                  use_lp_basis=False,
                    #                                  parallel_method='process',
                    #                                  agg_cut_part=False,
                    #                                  solve_MP_use_Benders=False,
                    #                                  revise_dual_value=True,
                    #                                  n_tree_iter_max=100000,
                    #                                      )
                    #
                    #     CCG_iterator_stoch.start_Benders_iteration(timelimit=CCG_timelimit, use_i_ccg=use_i_ccg)
                    #     result1 = CCG_iterator_stoch.update_result_ccg(result=result1,
                    #                                             read_data_time=read_data_time, tag='tree_search_CCG-100000')
                    #     write_result(file_name='calculation_info.xlsx', result=result1)
                    #     del CCG_iterator_stoch





        except Exception as e:
            with open(console_output_file, 'a') as file:
                content = file_name + ' :calculate terminated'
                print(content, file=file)
                print(e, file=file)




