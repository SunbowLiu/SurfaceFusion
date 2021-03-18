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


def drawweight(args, svd_list):
    # fontP = font_manager.FontProperties()

    # plt.gca().xaxis.set_major_formatter(ticker.EngFormatter())
    plt.plot(svd_list[0], label='Random')
    plt.plot(svd_list[1], label='TA_Important')
    plt.plot(svd_list[2], label='TA_Useless')
    plt.legend()
    # plt.locator_params(nbins=5)

    # save figure
    matplotlib.rcParams['font.family'] = "Times New Roman"
    if args.path:
        plt.savefig(args.path, format="png")
    else:
        plt.savefig(args.input + '.bestworse.svd.png', format="png")

    plt.show()


def getweight(args, input):
    state = torch.load(input,
                       map_location=(
                           lambda s, _: torch.serialization.default_restore_location(s, 'cpu')
                       ))
    dims = getattr(state['args'], 'encoder_embed_dim') // 2

    if args.feature == 'fgta':
        # soft_weight=torch.nn.functional.softmax(state['model']['decoder.ta_weight'][5].float(),dim=0)
        # soft_weight = np.argsort(-soft_weight[0])
        mean_weight = torch.mean(state['model']['decoder.ta_weight'][-2:].float(), dim=0)
        mean_weight = torch.nn.functional.softmax(mean_weight, dim=0)
        mean_weight = np.argsort(-mean_weight[0])

        # print(soft_weight)
        print(mean_weight)

    emb = state['model']['encoder.embed_tokens.weight'].float()

    ori = emb[:, np.random.permutation(dims * 2)]
    # now=emb[:,soft_weight]
    mean = emb[:, mean_weight]
    print(ori[0])
    print(mean[0][:dims])

    return calsvd(ori[:, :dims]), calsvd(mean[:, :dims]), calsvd(mean[:, -dims:])


def calsvd(emb):
    U, sigma, VT = np.linalg.svd(emb)

    sigma = sigma / np.max(sigma)

    print(np.linalg.norm(emb), np.linalg.norm(emb, ord=1), np.linalg.norm(emb, ord=2), np.linalg.norm(emb, ord=np.inf))

    return sigma


def main():
    parser = argparse.ArgumentParser(
        description='Tool to average the params of input checkpoints to '
                    'produce a new checkpoint',
    )
    parser.add_argument('--input', type=str, metavar='STR',
                        help='Input checkpoint file path.')
    parser.add_argument('--feature', default='ta', type=str, metavar='STR',
                        help='fgta or vanilla ta.')
    parser.add_argument('--path', default='', type=str, metavar='STR',
                        help='output path.')

    args = parser.parse_args()

    drawweight(args, getweight(args, args.input))


if __name__ == '__main__':
    main()
