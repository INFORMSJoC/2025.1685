"""Run the response-search (RS) reformulation reported in Tables 3 and 4."""

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from read_data import Modeldata
from model_purely_combinatorial import IterateComb
from run_tables_6_8_case_study import get_empty_result, write_result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", type=int, choices=(3, 4), default=3,
                        help="Paper table to reproduce (default: 3)")
    parser.add_argument("--time-limit", type=int, default=None,
                        help="MILP time limit in seconds (default: 600 for Table 3; 3600 for Table 4)")
    parser.add_argument("--preprocessing-cores", type=int, choices=(1, 2, 4, 8, 16), default=8,
                        help="Response-search workers; use 1 for serial preprocessing (default: 8)")
    args = parser.parse_args()
    if args.time_limit is None:
        args.time_limit = {3: 600, 4: 3600}[args.table]
    return args


def main():
    args = parse_args()
    default_data_dir = {
        3: ROOT / "data" / "table_3_small_instances",
        4: ROOT / "data" / "table_4_medium_instances",
    }[args.table]
    data_dir = Path(os.environ.get("ODMTS_DATA_DIR", default_data_dir))
    parallel_method = os.environ.get("ODMTS_PARALLEL_METHOD", "process")
    results = get_empty_result()

    for input_file in sorted(data_dir.glob("*.xlsx")):
        start = time.time()
        data = Modeldata(file_name=str(input_file), arc_elimination=False,
                         Delta_type="MST", Delta_value=5)
        read_time = time.time() - start
        solver = IterateComb(
            data=data,
            parallel_method=parallel_method,
            revise_dual_value=True,
            n_tree_iter_max=10000,
            preprocessing_cores=args.preprocessing_cores,
        )
        solve_info = solver.solve_uncompleted_MILP(time_limit=args.time_limit)
        solver.update_cal_infor(n_Benders_iter=0)
        solver.record.update(solve_info)
        results = solver.update_result_ccg(results, read_data_time=read_time, tag="RS")
        write_result(
            file_name=f"tables_3_4_response_search_{args.preprocessing_cores}_cores.xlsx",
            result=results,
        )


if __name__ == "__main__":
    main()
