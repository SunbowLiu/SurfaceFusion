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


def drawweight(args, weight, enc_layer, dec_layer):
    ig, axs = plt.subplots()
    rlv = np.array(weight)[::-1]
    print(weight)

    maximum = np.max(rlv)
    minimum = np.min(rlv)

    # axs.matshow(rlv, cmap="gray_r", vmin=minimum, vmax=maximum)
    axs.matshow(rlv, cmap="Blues", vmin=0.097, vmax=0.264)

    plt.gca()
    for i in range(dec_layer):
        for j in range(enc_layer):
            axs.text(j, 6 - 1 - i, '{:.2f}'.format(weight[i][j]).replace("0.", "."), ha="center", va="center",
                     color='black', fontsize=20)
    axs.set_xlabel('Encoder Layer', fontsize=20)
    # axs.set_ylabel('Decoder Layer',fontsize=20)
    axs.set_xticks(np.arange(enc_layer))
    axs.set_yticks(np.arange(dec_layer))
    axs.set_xticklabels(range(enc_layer), fontsize=20)
    axs.set_yticklabels(range(dec_layer, 0, -1), fontsize=20)
    axs.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True)

    matplotlib.rcParams['font.family'] = "Times New Roman"

    if args.path:
        plt.savefig(args.path, format="pdf", bbox_inches='tight')
    else:
        plt.savefig(args.input + 'taweight.pdf', format="pdf", bbox_inches='tight')


def getweight(args):
    state = torch.load(args.input,
                       map_location=(
                           lambda s, _: torch.serialization.default_restore_location(s, 'cpu')
                       ))

    weight = []
    addlayer = 0
    for x in range(6):
        if args.feature == 'fgta':
            soft_weight = torch.nn.functional.softmax(state['model']['decoder.ta_weight'][x].float(), dim=0)
            soft_weight = torch.mean(soft_weight, dim=1).numpy().tolist()
            weight.append(soft_weight)
            addlayer = 1
            # soft_weight=torch.argmax(soft_weight,dim=0).numpy().tolist()
            # print(sorted(Counter(soft_weight).items()))
        else:
            weight = torch.nn.functional.softmax(state['model']['decoder.ta_weight'].float())
            break

    return weight, getattr(state['args'], 'encoder_layers') + addlayer, getattr(state['args'], 'decoder_layers')


def main():
    parser = argparse.ArgumentParser(
        description='Tool to average the params of input checkpoints to '
                    'produce a new checkpoint',
    )
    parser.add_argument('--input', type=str, metavar='STR',
                        help='Input checkpoint file path.')
    parser.add_argument('--feature', default='ta', type=str, metavar='STR',
                        help='Input checkpoint file path.')
    parser.add_argument('--path', default='', type=str, metavar='STR',
                        help='output path.')

    args = parser.parse_args()

    weight, enc_layer, dec_layer = getweight(args)

    drawweight(args, weight, enc_layer, dec_layer)


if __name__ == '__main__':
    main()
