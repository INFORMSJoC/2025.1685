# Data

- `table_3_small_instances/`: four 81-node, 4-hub instances used in Table 3.
- `table_4_medium_instances/`: six 81-node, 4-hub instances used in Table 4.
- `table_5_large_instances/`: six 81-node, 6-hub instances used in Table 5;
  Table 9 uses four of these six instances.
- `case_study/`: the three final Ann Arbor/Ypsilanti instances used for
  Tables 6-8 and Figures 3 and 5 (`theta = 0.0001`, `0.002`, and `0.005`).
- `network/`: hub, trip-area, and node-area files used to draw the case-study
  network designs.

The instance workbooks already contain the generated scenarios used in the
reported experiments. No additional random scenario generation is needed.

The instance filenames were matched to the `file_name` records in the final
Table 3-5 and Table 9 workbooks. The case-study timestamp and parameter values
were matched to the final Section 6.3 result batch. SHA-256 comparison then
confirmed that every file here is byte-identical to the selected original.
