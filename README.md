[![INFORMS Journal on Computing Logo](https://INFORMSJoC.github.io/logos/INFORMS_Journal_on_Computing_Header.jpg)](https://pubsonline.informs.org/journal/ijoc)

# Stochastic Bilevel Network Design

This archive is distributed in association with the [INFORMS Journal on
Computing](https://pubsonline.informs.org/journal/ijoc) under the [MIT
License](LICENSE).

This archive contains the software, instances, and results used for the paper
["Stochastic Bilevel Optimization for the Network Design of Multimodal Transit
Systems with Heterogeneous Rider Preferences under Uncertain Travel Times and
Demand"](https://doi.org/10.1287/ijoc.2025.1685) by Suri Liu, Yiling Zhang,
Beste Basciftci, and Wenyuan Wang.

The code implements the response-search reformulation and the cutting-plane
algorithms described in the paper. The archive follows the INFORMS Journal on
Computing submission layout and contains the files used for Tables and
Figures for numerical results.

## Requirements

- Linux (the reported experiments used Ubuntu 20.04 LTS)
- Python 3.9
- Gurobi 12.0 with a valid local license
- Up to 16 CPU cores for the parallel response-search experiments

The code can also run on Windows system with a valid Gurobi license.

## Repository layout

- `src/`: optimization models, data reader, response search, and evaluation code.
- `scripts/`: experiment and postprocessing scripts named by paper output. The scripts starting with "run_" execute algorithms. Then run the scripts starting with "build_" to build tables using original outputs.
- `data/`: the exact instances used for Tables 3-5 and the case study.
- `results/`: reported workbooks, figures, and the raw files needed by plotting/postprocessing scripts.
- `output/`: destination for newly generated solver output.



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

The RS runner uses 8 preprocessing workers by default. Set
`--preprocessing-cores` explicitly to reproduce each preprocessing column:

```bash
python scripts/run_tables_3_4_response_search.py --preprocessing-cores 1
python scripts/run_tables_3_4_response_search.py --preprocessing-cores 2
python scripts/run_tables_3_4_response_search.py --preprocessing-cores 4
python scripts/run_tables_3_4_response_search.py --preprocessing-cores 8
python scripts/run_tables_3_4_response_search.py --preprocessing-cores 16
```

Use the same options together with `--table 4` for Table 4. A value of 1 runs
the response search serially; larger values create exactly that many worker
processes in `PotentialHubFinder.find_hub_leg_all`. Each run is written to a
core-specific workbook so the timing results are not overwritten. Use
`--preprocessing-cores 16` for the 16-core preprocessing time included in the
paper's RS total-time column.

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

To cite the contents of this repository, please cite both the paper and this
software archive using their respective DOIs:

https://doi.org/10.1287/ijoc.2025.1685

https://doi.org/10.1287/ijoc.2025.1685.cd

BibTeX for citing this software archive:

```bibtex
@misc{LiuEtAl2025Code,
  author =        {Suri Liu and Yiling Zhang and Beste Basciftci and Wenyuan Wang},
  publisher =     {INFORMS Journal on Computing},
  title =         {{Stochastic Bilevel Network Design}},
  year =          {2025},
  doi =           {10.1287/ijoc.2025.1685.cd},
  url =           {https://github.com/INFORMSJoC/2025.1685},
  note =          {Available for download at https://github.com/INFORMSJoC/2025.1685},
}
```

## License

The software is distributed under the MIT License. See `LICENSE`.
