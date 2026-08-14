"""
Run the KKT and strong-duality benchmarks reported in Tables 3 and 4.
add_why_no_need_M = True denotes to add the valid inequality.
"""

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model_parallel import Modeldata, OneStageModelFullMultiplierDepends, write_result_KKT


def empty_results():
    return {
        "file_name": [],
        "node-hub-trip-passenger": [],
        "arc elim": [],
        "Delta type": [],
        "use kkt": [],
        "use sos1": [],
        "use bigM": [],
        "use strong duality": [],
        "use why no need": [],
        "time read data": [],
        "time model creation": [],
        "solution time": [],
        "obj": [],
        "gap": [],
        "n_node_BnB": [],
    }


METHODS = {
    "KKT": dict(use_KKT=True, use_sos1=True, use_bigM=False, add_strong_dual=False,
                add_why_no_need_M=False),
    "KKT+VI": dict(use_KKT=True, use_sos1=True, use_bigM=False, add_strong_dual=False,
                   add_why_no_need_M=True),
    "SD": dict(use_KKT=True, use_sos1=False, use_bigM=False, add_strong_dual=True,
               add_why_no_need_M=False),
    "SD+VI": dict(use_KKT=True, use_sos1=False, use_bigM=False, add_strong_dual=True,
                  add_why_no_need_M=True),
}

TABLE_METHODS = {
    3: ("KKT", "KKT+VI", "SD", "SD+VI"),
    4: ("SD", "SD+VI"),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", type=int, choices=(3, 4), default=3,
                        help="Paper table to reproduce (default: 3)")
    parser.add_argument("--time-limit", type=int, default=None,
                        help="Optimization time limit in seconds (default: 600 for Table 3; 3600 for Table 4)")
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
    output_file = ROOT / "output" / "tables_3_4_single_level.xlsx"
    results = empty_results()

    for input_file in sorted(data_dir.glob("*.xlsx")):
        start = time.time()
        data = Modeldata(file_name=str(input_file), arc_elimination=False,
                         Delta_type="MST", Delta_value=5)
        read_time = time.time() - start
        for method in TABLE_METHODS[args.table]:
            options = METHODS[method]
            print(f"{input_file.name}: {method}")
            model = OneStageModelFullMultiplierDepends(data=data, **options)
            model.solve_model(time_limit=args.time_limit)
            model.update_result_record(results, read_data_time=read_time)
            write_result_KKT(results, str(output_file))


if __name__ == "__main__":
    main()
