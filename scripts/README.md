# Scripts

| Script | Paper output |
| --- | --- |
| `run_tables_3_4_response_search.py` | RS preprocessing and MILP results in Tables 3 and 4 |
| `run_tables_3_4_single_level_benchmarks.py` | KKT, KKT+VI, SD, and SD+VI results in Tables 3 and 4 |
| `run_table_5_large_instances.py` | Large-instance comparison in Table 5 |
| `run_tables_6_8_case_study.py` | Case-study designs and raw values for Tables 6-8 |
| `build_tables_6_7_case_study.py` | Investment, disutility, and trip-duration summaries in Tables 6 and 7 |
| `build_table_8_theta_sensitivity.py` | Leader-preference sensitivity summary in Table 8 |
| `build_figures_3_5_network_designs.py` | Case-study network designs in Figures 3 and 5 |
| `build_figure_4_response_iterations.py` | RS iteration distributions in Figure 4 |
| `run_table_9_callback_comparison.py` | Branch-and-cut callback comparison in Table 9 |

All scripts resolve paths relative to the repository root. Set
`ODMTS_DATA_DIR` only when selecting the Table 4 dataset for a script whose
default is the Table 3 dataset.

