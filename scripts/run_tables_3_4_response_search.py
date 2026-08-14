"""Run the response-search (RS) reformulation reported in Tables 3 and 4."""

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from read_data import Modeldata
from model_purely_combinatorial import IterateComb
from run_tables_6_8_case_study import get_empty_result, write_result


def main():
    data_dir = Path(os.environ.get("ODMTS_DATA_DIR", ROOT / "data" / "table_3_small_instances"))
    parallel_method = os.environ.get("ODMTS_PARALLEL_METHOD", "process")
    results = get_empty_result()

    for input_file in sorted(data_dir.glob("*.xlsx")):
        start = time.time()
        data = Modeldata(file_name=str(input_file), arc_elimination=False,
                         Delta_type="MST", Delta_value=5)
        read_time = time.time() - start
        solver = IterateComb(
            data=data,
            agg_cut=False,
            solution_appro="primal",
            primal_based_dual_to_ini=False,
            use_lp_basis=False,
            parallel_method=parallel_method,
            agg_cut_part=False,
            solve_MP_use_Benders=False,
            revise_dual_value=True,
            n_tree_iter_max=10_000,
        )
        solve_info = solver.solve_uncompleted_MILP()
        solver.update_cal_infor(n_Benders_iter=0)
        solver.record.update(solve_info)
        results = solver.update_result_ccg(results, read_data_time=read_time, tag="RS")
        write_result(file_name="tables_3_4_response_search.xlsx", result=results)


if __name__ == "__main__":
    main()
