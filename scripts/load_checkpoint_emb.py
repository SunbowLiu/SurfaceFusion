#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import collections
import torch
import os
import re
from collections import Counter
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import font_manager
import sys


def drawweight(args, ta, vanilla):
    # fontP = font_manager.FontProperties()

    # plt.gca().xaxis.set_major_formatter(ticker.EngFormatter())
    plt.plot(vanilla, linewidth='5', label='Vanilla')
    plt.plot(ta, linewidth='5', label='Lexical Fusion')
    plt.legend(fontsize='large')
    plt.grid()
    plt.ylabel('Log Eigenvalues', fontsize='large')
    plt.xlabel('Index of Singular Value', fontsize='large')
    # plt.tick_params(labelsize='xx-large')
    # plt.locator_params(nbins=5)

    # save figure
    matplotlib.rcParams['font.family'] = "Times New Roman"
    plt.savefig(args.output + '.svd.png', format="png", bbox_inches='tight', dpi=300)
    plt.show()


def getweight(args, input):
    state = torch.load(input,
                       map_location=(
                           lambda s, _: torch.serialization.default_restore_location(s, 'cpu')
                       ))

    if args.emb == 'target':
        emb = state['model']['decoder.embed_tokens.weight'].float()
    else:
        emb = state['model']['encoder.embed_tokens.weight'].float()

    U, sigma, VT = np.linalg.svd(emb)

    sigma = sigma / np.max(sigma)

    return np.log(sigma)


def main():
    parser = argparse.ArgumentParser(
        description='Tool to average the params of input checkpoints to '
                    'produce a new checkpoint',
    )
    parser.add_argument('--input-ta', type=str, metavar='STR',
                        help='Input ta checkpoint file path.')
    parser.add_argument('--input-vanilla', type=str, metavar='STR',
                        help='Input vanilla checkpoint file path.')
    parser.add_argument('--emb', default='source', type=str, metavar='STR',
                        help='source or target.')
    parser.add_argument('--output', type=str, metavar='STR',
                        help='output file path.')
    args = parser.parse_args()

    drawweight(args, getweight(args, args.input_ta), getweight(args, args.input_vanilla))


if __name__ == '__main__':
    main()
