[![INFORMS Journal on Computing Logo](https://INFORMSJoC.github.io/logos/INFORMS_Journal_on_Computing_Header.jpg)](https://pubsonline.informs.org/journal/ijoc)

# Stochastic Bilevel Network Design

This archive contains the software, instances, and results used for the paper
"Stochastic Bilevel Optimization for the Network Design of Multimodal Transit
Systems with Heterogeneous Rider Preferences under Uncertain Travel Times and
Demand" by Suri Liu, Yiling Zhang, Beste Basciftci, and Wenyuan Wang.

The code implements the response-search reformulation and the cutting-plane
algorithms described in the paper. The archive follows the INFORMS Journal on
Computing submission layout and contains only the files used for Tables 3-9 and
Figures 3-5.

## Requirements

- Linux (the reported experiments used Ubuntu 20.04 LTS)
- Python 3.9
- Gurobi 12.0 with a valid local license
- Up to 16 CPU cores for the parallel response-search experiments

Install the Python dependencies with:

```bash
python3.9 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The Gurobi license is intentionally not included. Configure it using the
standard Gurobi licensing procedure before running an optimization script.

## Repository layout

- `src/`: optimization models, data reader, response search, and evaluation code.
- `scripts/`: experiment and postprocessing scripts named by paper output.
- `data/`: the exact instances used for Tables 3-5 and the case study.
- `results/`: reported workbooks, figures, and the raw files needed by plotting/postprocessing scripts.
- `output/`: destination for newly generated solver output.

See the README files inside `data/`, `scripts/`, and `results/` for the precise
mapping between files and paper outputs.

## Paper output coverage

The archive includes every computational result reported in Tables 3-9 and
Figures 3-5. The earlier numbered items are intentionally excluded:

- Figure 1 is a conceptual multimodal-transit schematic used to introduce the
  problem; it is not produced from an experiment or dataset.
- Figure 2 is the illustrative four-hub network for Example 1, not an
  experimental output.
- Table 1 is the hand-worked output of Algorithm 1 on that four-hub example.
- Table 2 sorts and deduplicates the same illustrative responses with
  Algorithm 2.

Consequently, Figures 1-2 and Tables 1-2 have no corresponding experiment
scripts or result files to include. There are no other numbered tables or
figures in the paper beyond Table 9 and Figure 5.

## Data provenance and verification

The data selection was checked against the files that generated the final
reported results, rather than inferred only from directory names:

1. The instance names recorded in the `file_name` column of the final Table 3,
   Table 4, and Table 5 result workbooks were extracted, deduplicated, and
   compared with the packaged workbook names. The comparisons were exact:
   4/4 small, 6/6 medium, and 6/6 large instances, with no missing or extra
   files. Table 9 records a four-instance subset of the six Table 5 instances.
2. The case-study result archive identifies its final Section 6.3 run as the
   batch with timestamp `2025-07-19-16-54-46`. The three packaged inputs use
   that exact batch identifier and the three reported values
   `theta = 0.0001`, `0.002`, and `0.005`.
3. SHA-256 hashes were compared after copying. All 19 packaged instance
   workbooks and all three network CSV files are byte-for-byte identical to
   their selected source files.
4. The network CSV files are plotting metadata, not optimization instances.
   Their three names are referenced directly by
   `scripts/build_figures_3_5_network_designs.py`.

This establishes an auditable link from the final result records to the
packaged inputs. Additional development instances were not included.

## Replicating the experiments

Run commands from the repository root. Optimization times depend on hardware,
solver configuration, and available CPU cores. The paper used time limits of
600 seconds for Table 3 and 3600 seconds for Tables 4, 5, and 9.

### Tables 3 and 4

Response-search reformulation:

```bash
python scripts/run_tables_3_4_response_search.py
python scripts/run_tables_3_4_response_search.py --table 4
```

KKT, KKT+VI, SD, and SD+VI reformulations for Table 3; only SD and SD+VI
for Table 4:

```bash
python scripts/run_tables_3_4_single_level_benchmarks.py
python scripts/run_tables_3_4_single_level_benchmarks.py --table 4
```

The runners default to Table 3 and 600 seconds. Selecting `--table 4`
automatically selects the Table 4 inputs and the 3600-second limit. Use
`--time-limit` only to override the paper's limit.

For the preprocessing columns with 2, 4, 8, and 16 cores, run the RS command
under the corresponding Linux CPU affinity or job-scheduler allocation. Use
`ODMTS_PARALLEL_METHOD=normal` for the serial column.

### Table 5

```bash
python scripts/run_table_5_large_instances.py
```

This script runs Cuts, RS, RS+Cuts, and RS+Cuts+Stab configurations on the six
large instances.

### Tables 6-8

```bash
python scripts/run_tables_6_8_case_study.py
python scripts/build_tables_6_7_case_study.py
python scripts/build_table_8_theta_sensitivity.py
```

### Figures 3 and 5

The final figures are included in `results/figures`. To regenerate the network
plots from the saved solution JSON files, set a Mapbox token and run:

```bash
export MAPBOX_TOKEN=your_token
python scripts/build_figures_3_5_network_designs.py
```

### Figure 4

```bash
python scripts/build_figure_4_response_iterations.py
```

### Table 9

```bash
python scripts/run_table_9_callback_comparison.py
```

## Results

The final computational tables are in `results/tables`, and the computational
figures are in `results/figures`. Raw case-study solutions are limited to the five model
configurations and three leader preference values used in Tables 6-8 and
Figures 3 and 5.

## Cite

Please cite both the paper and this software archive. The repository identifier
is `2025.1685`; add the final journal and archive DOIs when they are assigned.

## License

The software is distributed under the MIT License. See `LICENSE`.
