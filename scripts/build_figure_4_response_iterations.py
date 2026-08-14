# -*- coding: utf-8 -*-
"""
  Author: Author
  Email : liusuri@mail.dlut.edu.cn
   Time : 2025/7/8 16:57
Function: draw the distribution of the number of iterations in the tree-search algorithm. The results are in ../results\figures\figure_4
"""


import json
import matplotlib.pyplot as plt
import numpy as np
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
folder = os.path.join(ROOT, 'results', 'raw', 'figure_4')
figure_folder = os.path.join(ROOT, 'results', 'figures', 'figure_4')

list_file = [os.path.join(folder, i) for i in os.listdir(folder) if 'search_iter' in i]

for data_path in list_file:
    find_int_num = re.findall(r'\d+', os.path.basename(data_path))

    # Convert them to integers
    find_int_num = [int(num) for num in find_int_num]
    n_trip = find_int_num[2]
    n_sce = find_int_num[3]
    # plot_title = '{} trips, {} scenarios'.format(n_trip, n_sce)
    plot_title = '{}-{}'.format(n_trip, n_sce)

    with open(data_path, "r") as f:
        data: dict[str, float] = json.load(f)

    data = list(data.values())


    # Set global font properties
    plt.rcParams.update({
        "font.family": "Times New Roman",      # "sans-serif", "serif", "monospace"
        "font.size": 12,
        "axes.titlesize": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

    # Create figure
    fig, ax = plt.subplots(figsize=(5, 4))

    counts, bins, patches = ax.hist(
        data,
        bins=30,
        weights=np.ones_like(data) / len(data),
        color='dimgray',
        edgecolor='dimgray',
        alpha=0.65,           # transparency (0 = fully transparent, 1 = opaque)
        linewidth=1.0        # edge line width in points
    )

    # Axis labels and title
    ax.set_xlabel("RS Iters")
    ax.set_ylabel("Relative Frequency")
    # ax.set_title(plot_title)

    # Set y-axis range if desired (e.g., from 0 to 0.2)
    ax.set_ylim(0, 1)

    # Remove top and right spines for a clean look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Optional: add gridlines only in y-direction
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax.xaxis.grid(True, linestyle='--', alpha=0.5)

    # Tight layout to avoid clipping
    plt.tight_layout()

    # Save high-resolution figure
    plt.savefig(os.path.join(figure_folder, "{}.png".format(plot_title)), dpi=500)

    # Show the plot
    # plt.show()

