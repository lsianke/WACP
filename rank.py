import os
import numpy as np
import utils
import math
import random
import torch
import argparse
from scipy.stats import norm, entropy

from utils import *
from models import *
from mask import *
from rank import *

def _state_dict(state_dict):
	rst = []
	for n, p in state_dict.items():
		if "total_ops" not in n and "total_params" not in n:
			rst.append((n.replace('module.', ''), p))
	rst = dict(rst)
	return rst

#closeness centrality 
def get_closeness(Graph,N):
	close = []
	for i in range(N):
		sum = 0.0
		for _,w in Graph[i]:
			# sum += w #
			sum += 1024 - w 
		if sum <= 0: 
			close.append(0)
			continue
		close.append(len(Graph[i])/sum) # JS 越小越中心
	return close

def in_scc(F,rots,x):
	rt = find(F,x)
	if rt in rots:
		return True
	else :
		return False

def del_filter(G,closeness,F,N,cdel = -1):
	oris = set(range(N))
	savs = set()
	dels = set()
	rots = set() 
	ind = np.argsort(closeness)
	cnt = 0 
	delegate = [[] for x in range(N)] # 卷积核替代图

	for i in range(N):
		if closeness[ind[i]] > 0:
			if ind[i] not in dels and ind[i] not in savs and not in_scc(F,rots,ind[i]):
				savs.add(ind[i])
				rots.add(find(F,ind[i]))
			for _,w in G[ind[i]]:
				if _ not in dels and _ not in savs: 
					cnt += 1
					if cdel > 0 and cnt > cdel: 
						return list(oris-dels),delegate
					dels.add(_)
					delegate[ind[i]].append(_)
					
	return list(oris-dels),delegate

def find(F,x):
	if F[x] == x: 
		return x
	else:
		F[x] = find(F,F[x])
	return F[x]

def merge(F,x,y):
	if find(F,x) != find(F,y): F[find(F,y)] = find(F,x)

def get_dels(G,N,m):
	dels = set()
	sccs = set()
	F = list(range(N + 1))

	for i in range(N):
		for j,w in enumerate(G[i]):
			if w < m and i != j:
				dels.add(i)
				dels.add(j)
				merge(F,i,j)
	del_num = len(dels)
	for i in dels:
		sccs.add(find(F,i))
	scc_num = len(sccs)
	return del_num,scc_num,F

def get_threshold(G,ratio,N,X=None):
	# X = math.ceil(N * ratio) # 要删去的个数
	# X = int(N * ratio)
	if X is None: X = N - math.ceil(N * (1-ratio))
	eps = 1e-7
	l = np.array(G).min()-eps
	r = np.array(G).max()+eps
	rst = r #
	F = None

	# print('----->',l,r)

	while l<r:
		m = (l+r)/2
		# print('--',l,r,m,X)
		del_num,scc_num,F = get_dels(G,N,m)
		if del_num - scc_num > X:
			rst = r
			r = m-eps
		elif del_num - scc_num < X:
			l = m+eps
		else :
			return m,del_num,scc_num,F
		
	del_num,scc_num,F = get_dels(G,N,rst)
	return rst,del_num,scc_num,F

# Jensen-Shannon 散度
def get_distance_JS(arch, model, layer_id):
	bn_names    = get_bn_names(model)
	state_dict  = model.state_dict()
	norm_weight = None
	norm_bias   = None
	if arch == 'vgg_16_bn':
		cfg = [0,1,3,4,6,7,8,10,11,12,14,15,16]
		norm_weight = state_dict['features.norm'+str(cfg[layer_id])+'.weight']
		norm_bias   = state_dict['features.norm'+str(cfg[layer_id])+'.bias']
	elif arch in ['resnet_56','resnet_50','googlenet','densenet_40','mobilenet_v2']:
		name        = bn_names[layer_id]
		norm_weight = state_dict[name+'.weight']
		norm_bias   = state_dict[name+'.bias']

	std = norm_weight.cpu().numpy().tolist()
	m   = norm_bias.cpu().numpy().tolist()
	num = len(norm_bias)
	distance = []
	p = []
	for i in range(num):
		if m[i] < 0. : m[i] = 0.
		if std[i] <= 0. : std[i] = 1e-5
		p.append(np.array([norm.pdf(x/20,m[i],std[i]) for x in range(1,100,1)]))
		if sum(p[-1])==0: p[-1][0] = 1e-5
	for i in range(num): # num
		_distance = []
		for j in range(num):
			if i == j: 
				_distance.append(0.)
				continue
			M = (p[i]+p[j])/2
			JS = 0.5 * entropy(p[i],M,base=2) + 0.5 * entropy(p[j],M,base=2)
			if JS > 3: JS = 3
			_distance.append(JS)
		distance.append(_distance)
	return distance

# 
#
#rest_num 要丢掉的通道的数量；指定的情况下ratio不起作用
def get_rank_onelayer(arch, model, layer_id, ratio, rest_num = None):

	if os.path.exists('distance/'+arch+'/'+str(layer_id)+'.npy'):
		D = np.load('distance/'+arch+'/'+str(layer_id)+'.npy')
	else :
		D = get_distance_JS(arch, model, layer_id)
		if not os.path.isdir('distance/'+arch):
			os.makedirs('distance/'+arch)
		np.save('distance/'+arch+'/'+str(layer_id)+'.npy', D)

	num = len(D)
	G = []
	delegate = []

	if ratio > 0.:
		if rest_num is None:
			threshold,del_num,scc_num,F = get_threshold(D,ratio,num)
			cdel = num - math.ceil(num * (1-ratio))
		else :
			threshold,del_num,scc_num,F = get_threshold(D,ratio,num,X=rest_num)
			cdel = rest_num
		ccnt = 0
		for i in range(num):
			edges = []
			for j,w in enumerate(D[i]):
				if w <= threshold and i != j:
					edges.append((j,w))
					ccnt += 1
			G.append(edges)
		closeness = get_closeness(G,num)
		rank,delegate = del_filter(G,closeness,F,num,cdel = cdel)
	else :
		rank = list(range(num))
		delegate = []

	return rank,delegate

def get_conv_names(model = None):
	conv_names = []
	for name, module in model.named_modules():
		if isinstance(module,nn.Conv2d):
			conv_names.append(name+'.weight')
	return conv_names

def get_bn_names(model = None):
	conv_names = []
	for name, module in model.named_modules():
		if isinstance(module,nn.BatchNorm2d):
			conv_names.append(name)
	return conv_names
