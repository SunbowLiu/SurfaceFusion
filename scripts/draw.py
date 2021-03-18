# -*- coding: utf-8 -*-
#python draw.py dict src tgt checkpoint1 checkpoint2 ...
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
from sklearn.decomposition import PCA

def getweight(args):
	state = torch.load(args.checkpoint,
                       map_location=(
                           lambda s, _: torch.serialization.default_restore_location(s, 'cpu')
                       ))
	emb = state['model']['encoder.embed_tokens.weight'].float()
	pca=PCA(n_components=2)
	emb_pca=pca.fit_transform(emb)
	return emb_pca

def draw(emb,args):
	with open(args.dict) as f1:
		dictionay_all = [x.split()[0] for x in f1]
	with open(args.src) as f2:
		dictionary_src =set([word for line in f2 for word in line.split()])
	with open(args.tgt) as f3:
		dictionary_tgt =set([word for line in f3 for word in line.split()])

	final=[0,0,0,0] #bos,eos,pad,unk
	for x in dictionay_all:
		if x in dictionary_src and x in dictionary_tgt:
			final.append('0')
		elif x in dictionary_src:
			final.append('1')
		else:
			final.append('2')

	plt.scatter(emb[:,0],emb[:,1],c=final,s=1)
	plt.xlabel('X', fontsize='large')
	plt.ylabel('Y', fontsize='large')
	plt.xticks([-0.4,-0.2,0,0.2,0.4])
	plt.yticks([-0.4,-0.2,0,0.2,0.4])
	plt.savefig(args.output+".2d.emb.png", format="png", bbox_inches='tight')

def main():
	parser = argparse.ArgumentParser(
        description='Tool to draw 2d embedding visualization',
    )
	parser.add_argument('--dict', type=str, metavar='STR',
                        help='Dict path.')
	parser.add_argument('--src', type=str, metavar='STR',
                        help='src path.')
	parser.add_argument('--tgt', default='source', type=str, metavar='STR',
                        help='tgt path.')
	parser.add_argument('--checkpoint', default='source', type=str, metavar='STR',
						help='Input checkpoint file path.')
	parser.add_argument('--output', default='', type=str, metavar='STR',
						help='output path.')
	args = parser.parse_args()

	emb = getweight(args)
	draw(emb, args)

if __name__ == '__main__':
	main()




