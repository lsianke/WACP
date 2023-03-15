import torch.nn as nn
import numpy as np
import torch
import math
import argparse
import torch.optim as optim
import time
import os
import copy
import pickle
import random
from PIL import Image
from copy import deepcopy
from collections import OrderedDict
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torchvision import datasets, models, transforms
from scipy.stats import norm, entropy

from utils import *
from models import *
from mask import *
from rank import *
from tricks import *

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 & ImageNet Pruning')

parser.add_argument(
	'--data_dir',	
	default='G:\\data',	
	type=str,   
	metavar='DIR',				 
	help='path to dataset')
parser.add_argument(
	'--dataset',	 
	default='CIFAR10',   
	type=str,   
	choices=('CIFAR10','ImageNet'),
	help='dataset')
parser.add_argument(
	'--num_workers', 
	default=0,		   
	type=int,   
	metavar='N',				   
	help='number of data loading workers (default: 0)')
parser.add_argument(
	'--epochs',
	type=int,
	default=15,
	help='The num of epochs to train.')
parser.add_argument(
	'--lr',		 
	default=0.01,		
	type=float,								
	help='initial learning rate')
parser.add_argument(
	'--lr_decay_step',
	default='5,10',
	type=str,
	metavar='LR',
	help='learning rate decay step')
parser.add_argument(
	'--resume',
	type=str,
	default=None,
	metavar='PATH',
	help='load the model from the specified checkpoint')
parser.add_argument(
	'--batch_size', 
	default=128, 
	type=int,
	metavar='N',
	help='mini-batch size')
parser.add_argument(
	'--momentum', 
	default=0.9, 
	type=float, 
	metavar='M',
	help='momentum')
parser.add_argument(
	'--weight_decay', 
	default=0., 
	type=float,
	metavar='W', 
	help='weight decay',
	dest='weight_decay')
parser.add_argument(
	'--bn_weight_decay', 
	default=0.,
	type=float, 
	# action="store_true",
	help='bn_weight_decay')
parser.add_argument(
	"--warmup", 
	default=0, 
	type=int, 
	metavar="E", 
	help="number of warmup epochs"
	)
parser.add_argument(
	'--gpu', 
	default='0', 
	type=str,
	help='GPU id to use.')
parser.add_argument(
	'--job_dir',
	type=str,
	default='',
	help='The directory where the summaries will be stored.')
parser.add_argument(
	'--compress_rate',
	type=str,
	default=None,
	help='compress rate of each conv')
parser.add_argument(
	'--arch',
	type=str,
	default='vgg_16_bn',
	choices=('resnet_50','vgg_16_bn','resnet_56','densenet_40','googlenet','mobilenet_v2'),
	help='The architecture to prune')
parser.add_argument(
	'--input_size',
	type=int,
	default=32,
	help='The num of input size')
parser.add_argument(
	'--save_id',
	type=int,
	default=0,
	help='save_id')
parser.add_argument(
	'--from_scratch',
	type=bool,
	default=False,
	help='train from scratch')
parser.add_argument(
	"--label_smoothing",
	default=0.0,
	type=float,
	metavar="S",
	help="label smoothing",)
parser.add_argument(
	"--mixup", 
	default=0.2, 
	type=float, 
	metavar="ALPHA", 
	help="mixup alpha")

args		   = None
lr_decay_step  = None
logger		   = None
compress_rate  = None
trainloader	   = None
testloader	   = None
criterion	   = None
device		   = None
model		   = None
mask		   = None
checkpoint_dir = None
save_dir	   = None
best_acc	   = 0.
best_accs	   = []

def init():
	global args,lr_decay_step,logger,compress_rate,trainloader,testloader,criterion,device,model,mask,best_acc,best_accs,checkpoint_dir,save_dir
	args				   = parser.parse_args()
	logger				 = get_logger(os.path.join(args.job_dir, 'log/log'))
	compress_rate		  = format_compress_rate(args.compress_rate)
	trainloader,testloader = load_data(data_name = args.dataset, data_dir = args.data_dir, batch_size = args.batch_size, num_workers = args.num_workers)
	criterion			  = nn.CrossEntropyLoss()
	device				 = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
	model				  = eval(args.arch)().to(device)
	mask				   = eval('mask_'+args.arch)(model=model, job_dir=args.job_dir, device=device)
	best_acc			   = 0.  # best test accuracy

	if args.lr_decay_step != 'cos':
		lr_decay_step = list(map(int, args.lr_decay_step.split(',')))
	else :
		lr_decay_step = args.lr_decay_step

	if len(args.job_dir) > 0  and args.job_dir[-1] != '\\':
		args.job_dir += '/'

	if len(args.gpu) > 1:
		gpus = args.gpu.split(',')
		device_id = []
		for i in gpus:
			device_id.append(int(i))
		print('device_ids:',device_id)
		model = nn.DataParallel(model, device_ids=device_id).cuda()
	else :
		model = model.to(device)
	
	if not os.path.isdir(args.job_dir + 'final_pruned_model'):
		os.makedirs(args.job_dir + 'final_pruned_model')
	if not os.path.isdir(args.job_dir + 'pruned_checkpoint'):
		os.makedirs(args.job_dir + 'pruned_checkpoint')
	if not os.path.isdir(args.job_dir + 'mask'):
		os.makedirs(args.job_dir + 'mask')

	checkpoint_dir = args.job_dir + 'pruned_checkpoint' + '/'
	save_dir	   = args.job_dir + 'final_pruned_model' + '/'

	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
	logger.info('args:{}'.format(args))

def train(epoch,model,cov_id,trainloader,optimizer,criterion,mask = None):
	losses = AverageMeter('Loss', ':.4f')
	top1   = AverageMeter('Acc@1', ':.2f')
	top5   = AverageMeter('Acc@5', ':.2f')

	model.train()
	num	= len(trainloader)
	since  = time.time()
	_since = time.time()
	for i, (inputs,labels) in enumerate(trainloader, 0):

		# if i > 1 : break

		inputs = inputs.to(device)
		labels = labels.to(device)

		if args.from_scratch is True: adjust_learning_rate(optimizer, epoch, i, num)

		optimizer.zero_grad()
		outputs = model(inputs)
		loss = criterion(outputs, labels)

		#0.1
		# if args.label_smoothing > 0.0: loss = lambda: LabelSmoothing(args.label_smoothing)

		loss.backward()
		optimizer.step()

		# if mask is not None : mask.grad_mask(cov_id)

		acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
		losses.update(loss.item(), inputs.size(0))
		top1.update(acc1[0], inputs.size(0))
		top5.update(acc5[0], inputs.size(0))

		if i!=0 and i%2000 == 0:
			_end = time.time()
			logger.info('epoch[{}]({}/{}) Loss: {:.4f} Acc@1: {:.2f} Acc@5: {:.2f} time: {:.4f}'.format(epoch,i,int(1280000/args.batch_size),losses.avg,top1.avg,top5.avg,_end - _since))
			_since = time.time()

	end = time.time()
	logger.info('train	Loss: {:.4f} Acc@1: {:.2f} Acc@5: {:.2f} time: {:.4f}'.format(losses.avg,top1.avg,top5.avg,end - since))

def validate(epoch,model,cov_id,testloader,criterion,save = True):
	losses = AverageMeter('Loss', ':.4f')
	top1   = AverageMeter('Acc@1', ':.2f')
	top5   = AverageMeter('Acc@5', ':.2f')

	model.eval()
	with torch.no_grad():
		since = time.time()
		for i, data in enumerate(testloader, 0):

			# if i > 1 : break

			inputs, labels = data
			inputs = inputs.to(device)
			labels = labels.to(device)

			outputs = model(inputs)
			loss = criterion(outputs, labels)

			acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
			losses.update(loss.item(), inputs.size(0))
			top1.update(acc1[0], inputs.size(0))
			top5.update(acc5[0], inputs.size(0))

		end = time.time()
		logger.info('validate Loss: {:.4f} Acc@1: {:.2f} Acc@5: {:.2f} time: {:.4f}'.format(losses.avg,top1.avg,top5.avg,end - since))

	global best_acc
	if save and best_acc <= top1.avg:
		best_acc = top1.avg
		state = {
			'state_dict': model.state_dict(),
			'best_prec1': best_acc,
			'epoch': epoch,
		}
		cov_name = '_cov' + str(cov_id)
		if cov_id == -1: cov_name = ''
		torch.save(state,checkpoint_dir+args.arch+cov_name + '.pt')
		logger.info('storing checkpoint:'+'pruned_checkpoint/'+args.arch+cov_name + '.pt')

	return top1.avg,top5.avg

def prune_vgg16bn():

	last_layer = 12
	cfg = [0,1,3,4,6,7,8,10,11,12,14,15,16]
	ranks = []
	optimizer  = None 
	scheduler  = None
	conv_names = get_conv_names(model)
	bn_names   = get_bn_names(model)

	for layer_id in range(last_layer,-1,-1):

		logger.info("===> pruning layer {}".format(layer_id))
		optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,weight_decay = args.weight_decay)
		scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_decay_step, gamma=0.1)
		
		if layer_id == last_layer: 
			pruned_checkpoint = torch.load(args.resume, map_location='cuda:0')
			logger.info('loading checkpoint:' + args.resume)
			model.load_state_dict(_state_dict(pruned_checkpoint['state_dict']))
		else :
			pruned_checkpoint = torch.load(checkpoint_dir + args.arch+"_cov" + str(layer_id+1) + '.pt', map_location=device)
			logger.info('loading checkpoint:' + "pruned_checkpoint/"+args.arch+"_cov" + str(layer_id+1) + '.pt')
			model.load_state_dict(pruned_checkpoint['state_dict'])

		conv_name  = conv_names[layer_id]
		state_dict = model.state_dict()
		rank, delegate = get_rank_onelayer(args.arch, model, layer_id, compress_rate[layer_id])
		ranks.insert(0,rank)
		# print(len(rank),rank)
		mask.layer_mask(layer_id, param_per_cov=4, arch=args.arch, rank = rank) 

		if layer_id != last_layer:
			# before_acc1,_ = validate(0,model,0,testloader,criterion,save = False)
			incs = adjust_filter(conv_names[layer_id+1],delegate,model = model)
			after_acc1,_ = validate(0,model,0,testloader,criterion,save = False)

		_train(layer_id)

		logger.info("===> layer {} bestacc {:.4f}".format(layer_id,best_accs[-1]))

	logger.info(best_accs)
	logger.info([len(x) for x in ranks])

	final_checkpoint = torch.load(checkpoint_dir + args.arch+"_cov" + str(0) + '.pt', map_location=device)
	pruned_model = vgg_16_bn(_state_dict(final_checkpoint['state_dict']),ranks)
	logger.info(pruned_model)
	flops,params = model_size(pruned_model,args.input_size,device)
	state = {
		'state_dict': pruned_model.state_dict(),
		'best_prec1': best_accs[-1],
		'scheduler':scheduler.state_dict(),
		'optimizer': optimizer.state_dict(),
		'ranks':ranks,
		'flops':flops,
		'params':params,
		'compress_rate':compress_rate
	}
	
	torch.save(state, save_dir + args.arch+'_'+str(args.save_id)+'.pt')
	logger.info('storing pruned_model:'+'final_pruned_model/'+args.arch+'_'+str(args.save_id)+'.pt')
	acc1,acc5 = validate(0,pruned_model,0,testloader,criterion,save = False)
	logger.info('final model  Acc@1: {:.2f} Acc@5: {:.2f} flops: {} params:{}'.format(acc1,acc5,flops,params))

def prune_resnet_56():
	ranks = []
	layers = 55

	conv_names = get_conv_names(model)
	bn_names   = get_bn_names(model)
	optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay = args.weight_decay)
	scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_decay_step, gamma=0.1)

	#------------------------------------------------
	pruned_checkpoint = torch.load(args.resume, map_location='cuda:0')
	logger.info('loading checkpoint:' + args.resume)
	model.load_state_dict(_state_dict(pruned_checkpoint['state_dict']))

	ranks = ['no_pruned']*layers
	for layer_id in range(1,4):
		rank1 = []
		_id = (layer_id-1)*18 + 0 * 2 + 2 
		_num = len(model.state_dict()[conv_names[_id]])
		if layer_id == 0 and compress_rate[0] > 0.: 
			prune_num = int(_num*compress_rate[0])
			rank1 = list(range(_num))[prune_num//2:_num-(prune_num-prune_num//2)]
			mask.layer_mask(0, param_per_cov=3, rank=rank1, type = 1, arch=args.arch) # the first layer
			ranks[0] = rank1
		for block_id in range(0,9):
			_id = (layer_id-1)*18 + block_id * 2 + 2 
			prune_num = int(_num*compress_rate[_id])
			rank1 = list(range(_num))[prune_num//2:_num-(prune_num-prune_num//2)]
			if compress_rate[_id] > 0.:
				mask.layer_mask(_id, param_per_cov=3, rank=rank1, type = 1, arch=args.arch)
			ranks[_id] = rank1
		_train(_id)
	#------------------------------------------------
	_id = layers - 1
	for layer_id in range(3,0,-1):

		for block_id in range(8,-1,-1):
			logger.info("===> pruning layer_id {} block_id {}".format(layer_id,block_id))

			pruned_checkpoint = torch.load(checkpoint_dir + args.arch+"_cov" + str(_id) + '.pt', map_location=device)
			logger.info('loading checkpoint:' + "pruned_checkpoint/"+args.arch+"_cov" + str(_id) + '.pt')
			model.load_state_dict(pruned_checkpoint['state_dict'])

			for cov_id in range(2,0,-1):
				if cov_id == 1:
					_id = (layer_id-1)*18 + block_id * 2 + cov_id 
					rank, delegate = get_rank_onelayer(args.arch, model, _id, compress_rate[_id])
					ranks[_id] = rank
					mask.layer_mask(_id, param_per_cov=3, rank=rank, type = 1, arch=args.arch)
					incs = adjust_filter(conv_names[_id+1],delegate,model = model)
					after_acc1,_ = validate(0,model,0,testloader,criterion,save = False)
			ori_acc1,_ = validate(0,model,0,testloader,criterion,save = False)
			_train(_id)

	logger.info(best_accs)
	logger.info(compress_rate)
	logger.info([len(x) for x in ranks])

	final_state_dict = torch.load(checkpoint_dir+args.arch+"_cov" + str(1) + '.pt', map_location=device)
	rst_model = resnet_56(compress_rate = compress_rate,oristate_dict = _state_dict(final_state_dict['state_dict']),ranks=ranks)
	logger.info(rst_model)

	flops,params = model_size(rst_model,args.input_size,device)

	state = {
		'state_dict': rst_model.state_dict(),
		'best_prec1': best_accs[-1],
		'scheduler':scheduler.state_dict(),
		'optimizer': optimizer.state_dict(),
		'ranks':ranks,
		'compress_rate':compress_rate
	}

	torch.save(state,save_dir+args.arch+'_'+str(args.save_id)+'.pt')
	logger.info('storing pruned_model:'+'final_pruned_model/'+args.arch+'_'+str(args.save_id)+'.pt')
	acc1,acc5 = validate(0,rst_model,0,testloader,criterion,save = False)
	logger.info('final model  Acc@1: {:.2f} Acc@5: {:.2f} flops: {} params:{}'.format(acc1,acc5,flops,params))

def prune_googlenet():
	ranks = []
	inceptions = ['a3','b3','a4','b4','c4','d4','e4','a5','b5']
	branch_offset = [0,1,3,6] #[1,2,4,7]

	conv_names = get_conv_names(model)
	optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,weight_decay = args.weight_decay)
	scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_decay_step, gamma=0.1)

	pruned_checkpoint = torch.load(args.resume, map_location='cuda:0')
	logger.info('loading checkpoint:' + args.resume)
	model.load_state_dict(_state_dict(pruned_checkpoint['state_dict']))
	state_dict = model.state_dict()

	#--------------------------get first-layer rank-----------------------------
	rank1, delegate = get_rank_onelayer(args.arch, model, 0, compress_rate[0])
	ranks.append([rank1])
	#---------------------------------------------------------------
	logger.info("===> pruning pre_layers")
	mask.layer_mask(0, param_per_cov=4, rank=rank1, type = 1, arch=args.arch)
	acc1,acc5 = validate(0,model,0,testloader,criterion,save = False)
	_train(0)

	for i,inception_id in enumerate(inceptions):
		# i += 1
		logger.info("===> pruning inception_id {}".format(i))

		pruned_checkpoint = torch.load(checkpoint_dir+args.arch+"_cov" + str(i) + '.pt', map_location=device)
		logger.info('loading checkpoint:' + "pruned_checkpoint/"+args.arch+"_cov" + str(i) + '.pt')
		model.load_state_dict(pruned_checkpoint['state_dict'])

		i_offset = [2,4,5]
		_rank = []

		for j in range(3):
			layer_id = (i)*7 + 1 + i_offset[j]
			_num = len(model.state_dict()[conv_names[layer_id]])
			if compress_rate[(i)*3+j+1] == 0.:
				_rank.append(list(range(_num)))	 
				continue
			rank1, delegate = get_rank_onelayer(args.arch, model, layer_id, compress_rate[(i)*3+j+1])
			# delegate
			if j==1:
				incs = adjust_filter(conv_names[layer_id+1],delegate,model = model)
			_rank.append(rank1)
			mask.layer_mask(layer_id, param_per_cov=4, rank=rank1, type = 1, arch=args.arch)
			acc1,acc5 = validate(0,model,0,testloader,criterion,save = False)

		ranks.append(_rank)
		_train(i+1)
		logger.info("===> inception_id {} best_acc {:.4f}".format(i,best_accs[-1]))

	logger.info(best_accs)

	final_state_dict = torch.load(checkpoint_dir+args.arch+'_cov9.pt', map_location=device)
	rst_model =  googlenet(compress_rate = compress_rate,oristate_dict = _state_dict(final_state_dict['state_dict']),ranks = ranks).to(device)
	flops,params = model_size(rst_model,args.input_size,device)
	logger.info(rst_model)

	state = {
		'state_dict': rst_model.state_dict(),
		'best_prec1': best_accs[-1],
		'scheduler':scheduler.state_dict(),
		'optimizer': optimizer.state_dict(),
		'ranks':ranks,
		'compress_rate':compress_rate
	}

	torch.save(state,save_dir+args.arch+'_'+str(args.save_id)+'.pt')
	logger.info('storing pruned_model:'+'final_pruned_model/'+args.arch+'_'+str(args.save_id)+'.pt')
	acc1,acc5 = validate(0,rst_model,0,testloader,criterion,save = False)
	logger.info('final model  Acc@1: {:.2f} Acc@5: {:.2f} flops: {} params:{}'.format(acc1,acc5,flops,params))

def prune_resnet_50():
	ranks = []
	layers = 49

	stage_repeat  = [3, 4, 6, 3]
	layer_last_id = [0,10,23,42,52]
	branch_types  = [-1,2,1,0,0]

	conv_names = get_conv_names(model)
	bn_names   = get_bn_names(model)

	optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,weight_decay = args.weight_decay)
	scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_decay_step, gamma=0.1)

	_id = 52
	for layer_id in range(4,0,-1):
		if layer_id == 4:
			pruned_checkpoint = torch.load(args.resume) #, map_location='cuda:0'
			logger.info('loading checkpoint:' + args.resume)
			model.load_state_dict(pruned_checkpoint)
		else :
			pruned_checkpoint = torch.load(checkpoint_dir+args.arch+"_cov" + str(layer_last_id[layer_id]) + '.pt', map_location=device)
			logger.info('loading checkpoint:' + "pruned_checkpoint/"+args.arch+"_cov" + str(layer_last_id[layer_id]) + '.pt')
			model.load_state_dict(pruned_checkpoint['state_dict'])
		lid	= layer_last_id[layer_id]
		l_rank, delegate = get_rank_onelayer(args.arch, model, lid, compress_rate[lid-layer_id])
		# print(lid,compress_rate[lid-layer_id],len(l_rank))

		cid = 0
		for block_id in range(0,stage_repeat[layer_id-1]):
			if block_id == 0:
				mask.layer_mask(layer_last_id[layer_id-1]+3, param_per_cov=3, rank=l_rank, type = 1, arch=args.arch)
				mask.layer_mask(layer_last_id[layer_id-1]+4, param_per_cov=3, rank=l_rank, type = 1, arch=args.arch)
				cid = layer_last_id[layer_id-1]+4
			else :
				cid += 3
				mask.layer_mask(cid, param_per_cov=3, rank=l_rank, type = 1, arch=args.arch)
		acc1,acc5 = validate(0,model,0,testloader,criterion,save = False)
		_train(lid)

		for block_id in range(stage_repeat[layer_id-1]-1,-1,-1):
			logger.info("===> pruning layer_id {} block_id {}".format(layer_id,block_id))

			if block_id == stage_repeat[layer_id-1]-1:
				pruned_checkpoint = torch.load(checkpoint_dir+args.arch+"_cov" + str(layer_last_id[layer_id]) + '.pt', map_location=device)
				logger.info('loading checkpoint:' + "pruned_checkpoint/"+args.arch+"_cov" + str(layer_last_id[layer_id]) + '.pt')
				model.load_state_dict(pruned_checkpoint['state_dict'])
			else :
				pruned_checkpoint = torch.load(checkpoint_dir+args.arch+"_cov" + str(_id) + '.pt', map_location=device)
				logger.info('loading checkpoint:' + "pruned_checkpoint/"+args.arch+"_cov" + str(_id) + '.pt')
				model.load_state_dict(pruned_checkpoint['state_dict'])

			for cov_id in range(3,0,-1):
				if block_id == 0 and cov_id == 3: 
					_id -= 1
					ranks.insert(0,l_rank)

				cpid = _id - layer_id # 0 - 48
				if block_id == 0 : cpid += 1
				if cov_id < 3:
					rank1, delegate = get_rank_onelayer(args.arch, model, _id, compress_rate[cpid])
					adjust_filter(conv_names[_id+1],delegate,model = model)
					ranks.insert(0,rank1)
					mask.layer_mask(_id, param_per_cov=3, rank=rank1, type = 1, arch=args.arch)
					# acc1,acc5 = validate(0,model,0,testloader,criterion,save = False)
				else :
					ranks.insert(0,l_rank)

				_id -= 1
			_train(_id)
			logger.info("===> layer_id {} block_id {} bestacc {:.4f}".format(layer_id,block_id,best_accs[-1]))
	ranks.insert(0,'no_pruned')

	logger.info(best_accs)

	print([len(x) for x in ranks])

	final_state_dict = torch.load(checkpoint_dir+args.arch+'_cov0.pt', map_location=device)
	rst_model = resnet_50(compress_rate=compress_rate,oristate_dict = _state_dict(final_state_dict['state_dict']),ranks = ranks)
	logger.info(rst_model)
	flops,params = model_size(rst_model,args.input_size,device)

	state = {
		'state_dict': rst_model.state_dict(),
		'best_prec1': best_accs[-1],
		'scheduler':scheduler.state_dict(),
		'optimizer': optimizer.state_dict(),
		'ranks':ranks,
		'compress_rate':compress_rate
	}

	torch.save(state,save_dir+args.arch+'_'+str(args.save_id)+'.pt')
	logger.info('storing pruned_model:'+'final_pruned_model/'+args.arch+'_'+str(args.save_id)+'.pt')
	acc1,acc5 = validate(0,rst_model,0,testloader,criterion,save = False)
	logger.info('final model  Acc@1: {:.2f} Acc@5: {:.2f} flops: {} params:{}'.format(acc1,acc5,flops,params))

def prune_mobilenet_v2():
	ranks = []

	cfg = model.interverted_residual_setting
	n_blocknum	= [0,1,2,3,4,3,3,1]
	IR_output	 = [32]  # num of output channel
	IR_expand	 = [1]
	IR_cprate_id  = [0] # skip branch id
	IR_2th_cprate = [] # cprate*18
	sequence_id   = [0] #sequence_last_block
	for i in range(1,7+1):
		IR_output	+= [cfg[i-1][1]] * n_blocknum[i]
		IR_expand	+= [cfg[i-1][0]] * n_blocknum[i]
		sequence_id  += [i] * n_blocknum[i]
		IR_cprate_id += [IR_cprate_id[-1]+n_blocknum[i-1]+1]

	for i,x in enumerate(compress_rate) :
		if i == 0 or i not in IR_cprate_id:
			IR_2th_cprate.append(x)

	conv_names = get_conv_names(model)
	bn_names   = get_bn_names(model)
	optimizer  = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,weight_decay = args.weight_decay)
	scheduler  = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_decay_step, gamma=0.1)

	ranks.insert(0,'no_pruned') #51 IR-18
	_id = 50
	is_sequence_last = -1 # 
	sequence_last_rank = None 
	sequence_2th_rank = None 
	for IR_id in range(17,0,-1):
		logger.info("===> pruning InvertedResidual block {}".format(IR_id))

		if IR_id == 17:
			pruned_checkpoint = torch.load(args.resume, map_location='cuda:0')
			logger.info('loading checkpoint:' + args.resume)
			model.load_state_dict(pruned_checkpoint)
		else :
			pruned_checkpoint = torch.load(checkpoint_dir+args.arch+"_cov" + str(IR_id+1) + '.pt', map_location=device)
			logger.info('loading checkpoint:' + "pruned_checkpoint/"+args.arch+"_cov" + str(IR_id+1) + '.pt')
			model.load_state_dict(pruned_checkpoint['state_dict'])

		# ori_acc1,_ = validate(0,model,0,testloader,criterion,save = False)

		for cov_id in range(3,0,-1):
			if IR_id == 1 and cov_id < 3: break

			if cov_id > 1:
				_num = len(model.state_dict()[conv_names[_id]])
				
				if cov_id == 2:
					last_output = math.ceil(IR_output[IR_id-1] * (1. - compress_rate[IR_cprate_id[sequence_id[IR_id-1]]])) * IR_expand[IR_id]
					c_output_num = math.ceil(last_output * (1. - IR_2th_cprate[IR_id]))
					rank1, delegate = get_rank_onelayer(args.arch, model, _id, 1,rest_num=_num-c_output_num)
					adjust_filter(conv_names[_id+1],delegate,model = model)
					sequence_2th_rank = rank1
				else :
					if sequence_id[IR_id] != is_sequence_last:
						rest_num = int(_num*compress_rate[IR_cprate_id[sequence_id[IR_id]]])
						rank1, delegate = get_rank_onelayer(args.arch, model, _id, 1 ,rest_num=rest_num)
						is_sequence_last = sequence_id[IR_id]
						sequence_last_rank = rank1
					else :
						rank1 = sequence_last_rank
			else :
				rank1 = sequence_2th_rank

			ranks.insert(0,rank1)
			mask.layer_mask(_id, param_per_cov=3, rank=rank1, type = 1, arch=args.arch)

			_id -= 1

		ori_acc1,_ = validate(0,model,0,testloader,criterion,save = False)
		_train(IR_id)

	ranks.insert(0,'no_pruned') # 1
	ranks.insert(0,'no_pruned') # 0

	logger.info(best_accs)
	logger.info([len(x) for x in ranks])

	final_state_dict = torch.load(checkpoint_dir+args.arch+'_cov1.pt', map_location=device)
	rst_model = mobilenet_v2(compress_rate=compress_rate,oristate_dict = _state_dict(final_state_dict['state_dict']),ranks = ranks)
	logger.info(rst_model)
	flops,params = model_size(rst_model,args.input_size,device)

	state = {
		'state_dict': rst_model.state_dict(),
		'best_prec1': best_accs[-1],
		'scheduler':scheduler.state_dict(),
		'optimizer': optimizer.state_dict(),
		'ranks':ranks,
		'compress_rate':compress_rate
	}

	torch.save(state,save_dir+args.arch+'_'+str(args.save_id)+'.pt')
	logger.info('storing pruned_model:'+'final_pruned_model/'+args.arch+'_'+str(args.save_id)+'.pt')
	acc1,acc5 = validate(0,rst_model,0,testloader,criterion,save = False)
	logger.info('final model  Acc@1: {:.2f} Acc@5: {:.2f} flops: {} params:{}'.format(acc1,acc5,flops,params))

def _train(i):
	global best_acc,best_accs
	optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,weight_decay = args.weight_decay)
	scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_decay_step, gamma=0.1)
	for epoch in range(0, args.epochs):
		logger.info('epoch {} learning_rate {} '.format(epoch,optimizer.param_groups[0]['lr']))
		train(epoch, model,i,trainloader,optimizer,criterion,mask) #,mask
		scheduler.step()
		validate(epoch,model,i,testloader,criterion)
	if args.epochs > 0 and best_acc > 0.:
		best_accs.append(round(best_acc.item(),4))
	else:
		best_accs.append(0.)
	best_acc=0.

def get_optimizer(model,arch):
	parameters = model.named_parameters()
	# if args.bn_weight_decay is True:
	# 	print(args.bn_weight_decay)
	# 	optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,weight_decay = args.weight_decay)
	# else :
	# 	bn = 'bn'
	# 	if arch == 'vgg_16_bn':
	# 		bn = 'norm'
	# 	bn_params   = [v for n,v in parameters if bn in n]
	# 	rest_params = [v for n,v in parameters if not bn in n]
	# 	print(len(bn_params))
	# 	print(len(rest_params))
	# 	optimizer = torch.optim.SGD(
	# 		[
	# 			{"params": bn_params, "weight_decay": args.bn_weight_decay},
	# 			{"params": rest_params, "weight_decay": args.weight_decay},
	# 		],
	# 		lr=args.lr,
	# 		momentum=args.momentum,
	# 		weight_decay=args.weight_decay,
	# 	)
	bn = 'bn'
	if arch == 'vgg_16_bn':
		bn = 'norm'
	bn_params   = [v for n,v in parameters if bn in n]
	rest_params = [v for n,v in parameters if not bn in n]
	print(len(bn_params))
	print(len(rest_params))
	optimizer = torch.optim.SGD(
		[
			{"params": bn_params, "weight_decay": args.bn_weight_decay},
			{"params": rest_params, "weight_decay": args.weight_decay},
		],
		lr=args.lr,
		momentum=args.momentum,
		weight_decay=args.weight_decay,
	)
	return optimizer

def train_from_scratch():
	pruned_checkpoint = torch.load(args.resume, map_location='cuda:0')
	logger.info('loading checkpoint:' + args.resume)
	if args.compress_rate is None:
		compress_rate = pruned_checkpoint['compress_rate']
	else :
		compress_rate = format_compress_rate(args.compress_rate)
	model = eval(args.arch)(compress_rate=compress_rate).to(device)
	logger.info(model)
	model.load_state_dict(_state_dict(pruned_checkpoint['state_dict']))
	validate(0,model,0,testloader,criterion,save = False)

	# optimizer = get_optimizer(model,args.arch)
	optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,weight_decay = args.weight_decay)
	scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_decay_step, gamma=0.1)

	if lr_decay_step != 'cos':
		scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_decay_step, gamma=0.1)
	else :
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.epochs, eta_min=0.)

	for epoch in range(0, args.epochs):
		if lr_decay_step !='cos' and epoch in lr_decay_step[-1:] and args.arch not in ['resnet_50','mobilenet_v2','resnet_56']: #!= 'resnet_50' 'resnet_56',
			resume = args.job_dir + 'pruned_checkpoint/'+args.arch+'.pt'
			pruned_checkpoint = torch.load(resume, map_location='cuda:0')
			logger.info('loading checkpoint:' + 'pruned_checkpoint/'+args.arch+'.pt')
			model.load_state_dict(_state_dict(pruned_checkpoint['state_dict']))

		logger.info('epoch {} learning_rate {} '.format(epoch,optimizer.param_groups[0]['lr']))
		train(epoch,model,0,trainloader,optimizer,criterion) #,mask
		if lr_decay_step != 'cos': scheduler.step()
		validate(epoch,model,-1,testloader,criterion)

	flops,params = model_size(model,args.input_size,device)

	best_model = torch.load(args.job_dir + "pruned_checkpoint/"+args.arch+'.pt', map_location=device)
	final_model = eval(args.arch)(compress_rate=compress_rate).to(device)
	final_model.load_state_dict(_state_dict(best_model['state_dict']))

	state = {
		'state_dict': final_model.state_dict(),
		'best_prec1': round(best_acc.item(),4),
		'compress_rate': compress_rate
	}
	print(compress_rate)
	print(model)

	torch.save(state,save_dir +args.arch+'_'+str(args.save_id)+'_fs.pt')
	logger.info('storing pruned_model:'+'final_pruned_model/'+args.arch+'_'+str(args.save_id)+'_fs.pt')
	acc1,acc5 = validate(0,final_model,0,testloader,criterion,save = False)
	logger.info('final model  Acc@1: {:.2f} Acc@5: {:.2f} flops: {} params:{}'.format(acc1,acc5,flops,params))

def train_from_scratch2():
	pruned_checkpoint = torch.load(args.resume, map_location='cuda:0')
	logger.info('loading checkpoint:' + args.resume)
	model = eval(args.arch)().to(device)
	logger.info(model)
	model.load_state_dict(_state_dict(pruned_checkpoint['state_dict']))
	validate(0,model,0,testloader,criterion,save = False)

	# optimizer = get_optimizer(model,args.arch)
	optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,weight_decay = args.weight_decay)
	scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_decay_step, gamma=0.1)

	if lr_decay_step != 'cos':
		scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_decay_step, gamma=0.1)
	else :
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.epochs, eta_min=0.)

	for epoch in range(0, args.epochs):
		if lr_decay_step !='cos' and epoch in lr_decay_step[-1:] and args.arch not in ['resnet_50','mobilenet_v2']: #!= 'resnet_50' 'resnet_56',
			resume = args.job_dir + 'pruned_checkpoint/'+args.arch+'.pt'
			pruned_checkpoint = torch.load(resume, map_location='cuda:0')
			logger.info('loading checkpoint:' + 'pruned_checkpoint/'+args.arch+'.pt')
			model.load_state_dict(_state_dict(pruned_checkpoint['state_dict']))

		logger.info('epoch {} learning_rate {} '.format(epoch,optimizer.param_groups[0]['lr']))
		train(epoch,model,0,trainloader,optimizer,criterion) #,mask
		if lr_decay_step != 'cos': scheduler.step()
		validate(epoch,model,-1,testloader,criterion)

	flops,params = model_size(model,args.input_size,device)

	best_model = torch.load(args.job_dir + "pruned_checkpoint/"+args.arch+'.pt', map_location=device)
	final_model = eval(args.arch)(compress_rate=compress_rate).to(device)
	final_model.load_state_dict(_state_dict(best_model['state_dict']))

	state = {
		'state_dict': final_model.state_dict(),
		'best_prec1': round(best_acc.item(),4),
		'compress_rate': compress_rate
	}

	torch.save(state,save_dir +args.arch+'_'+str(args.save_id)+'_fs.pt')
	logger.info('storing pruned_model:'+'final_pruned_model/'+args.arch+'_'+str(args.save_id)+'_fs.pt')
	acc1,acc5 = validate(0,final_model,0,testloader,criterion,save = False)
	logger.info('final model  Acc@1: {:.2f} Acc@5: {:.2f} flops: {} params:{}'.format(acc1,acc5,flops,params))

def adjust_learning_rate(optimizer, epoch, step, len_iter):

	if lr_decay_step == 'cos':  
		lr = 0.5 * args.lr * (1 + math.cos(math.pi * (epoch - 5) / (args.epochs - 5)))
	else :
		gamma = 0.1
		lr = args.lr
		for b in lr_decay_step:
			if epoch >= b:
				lr *= gamma

	if epoch < args.warmup:
		lr = lr * float(1 + step + epoch * len_iter) / (5. * len_iter)

	# print(lr)

	for param_group in optimizer.param_groups:
		param_group['lr'] = lr

def adjust_filter(conv_name,delegate,model = None):
	logger.info("adjusting filter: "+conv_name)
	if len(delegate) < 1: return 
	incs = []
	state_dict_conv = model.state_dict()[conv_name]
	for i,x in enumerate(state_dict_conv):
		n,h,w = x.size()
		inc = []
		for j in range(n):
			_sum = 0.
			for k in delegate[j]:
				_sum += x[k] 
			x[j] += _sum
			inc.append(_sum)
		incs.append(inc)
	return incs

def adjust_filter_repeal(conv_name,incs,model = None):
	logger.info("undo: "+conv_name)
	state_dict_conv = model.state_dict()[conv_name]
	for i,x in enumerate(state_dict_conv):
		n,h,w = x.size()
		for j in range(n):
			x[j] -= incs[i][j]

def _state_dict(state_dict):
	rst = []
	for n, p in state_dict.items():
		if "total_ops" not in n and "total_params" not in n:
			rst.append((n.replace('module.', ''), p))
	rst = dict(rst)
	return rst

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

def justshowFeature(model,img,conv_outs,layer_id,rank=None,figsize=(8, 8)):
	
	class LayerActivations:
		features = None
		def __init__(self, feature):
			self.hook = feature.register_forward_hook(self.hook_fn)
		def hook_fn(self, module, input, output):
			self.features = output.cpu()
		def remove(self):
			self.hook.remove()
	conv_out = LayerActivations(conv_outs[layer_id])

	model.eval()
	print('running...')
	print(layer_id)
	out = model(Variable(img.cuda()))
	conv_out.remove()
	features = conv_out.features
	num = len(features[0])
	w = math.ceil(math.sqrt(num))
	
	fig = plt.figure(figsize=figsize)
	# rank = [1,2,3,4,22,55]
	if rank is None:
		for i in range(num):
			ax = fig.add_subplot(w, w, i+1, xticks=[], yticks=[])
			ax.imshow(features[0][i].detach().numpy())  #, cmap="gray"
	else :
		# for i,x in enumerate(rank):
		# 	ax = fig.add_subplot(w, w, i+1, xticks=[], yticks=[])
		# 	ax.imshow(features[0][x].detach().numpy()) 
		for i in range(num):
			ax = fig.add_subplot(w, w, i+1, xticks=[], yticks=[])
			if i in rank:
				
				ax.imshow(features[0][i].detach().numpy()) 
			else:
				# alpha = 0.7, 
				ax.imshow(features[0][i].detach().numpy(),alpha = 0.) 
				pass
	plt.show()

def visualize_Feature_by_rank(model, arch = None ,layer_id = None,rank = None):

	if arch is None : arch = 'vgg_16_bn'
	if layer_id is None : layer_id = 56
	show_data = False

	if model is None:
		if arch in ['resnet_50','shufflenet_v2'] :
			model = eval(arch)().to(device)
			resume = '../pre_train_model/ImageNet/'+arch+'.pth'
			pct = torch.load(resume, map_location='cuda:0')
			model.load_state_dict(pct)
			print('loading data...')
			testloader = load_ImageNet_demo(data_name = 'ImageNet', data_dir = 'G:\\ImageNet', batch_size = 64, num_workers = 1,pin_memory = False)
			print('OK!')
			img = list(testloader)[0][0][0] #next(iter(testloader))[0]
			img = torch.unsqueeze(img,dim=0)
		else:
			model = eval(arch)().to(device)
			PATH = '../pre_train_model/CIFAR-10/'+arch+'.pt'
			pre_train_model = torch.load(PATH,map_location='cuda:0')
			model.load_state_dict(_state_dict(pre_train_model['state_dict']))
			testloader = load_cifar10_demo(data_name = 'CIFAR10', data_dir = 'G:\\data', batch_size = 64, num_workers = 1,pin_memory = False,scale_size = 32)
			img = list(testloader)[0][0][0]
			img = torch.unsqueeze(img,dim=0)
	else :
		if arch in ['resnet_50','shufflenet_v2']:
			testloader = load_ImageNet_demo(data_name = 'ImageNet', data_dir = 'G:\\ImageNet', batch_size = 64, num_workers = 1,pin_memory = False)
			img = list(testloader)[0][0][9] #next(iter(testloader))[0] 8 熊猫
			img = torch.unsqueeze(img,dim=0)
		else:
			testloader = load_cifar10_demo(data_name = 'CIFAR10', data_dir = 'G:\\data', batch_size = 64, num_workers = 1,pin_memory = False,scale_size = 32)
			print('OK!')
			img = list(testloader)[0][0][1] #next(iter(testloader))[0]
			img = torch.unsqueeze(img,dim=0)

	print('rank: ',rank)

	conv_outs = []
	if arch == 'vgg_16_bn':
		cfg = [0,3,7,10,14,17,20,24,27,30,34,37,40] 
		for i in range(12+1):
			conv_outs.append(model.features[cfg[i]+2])
	elif arch == 'resnet_50':
		conv_outs.append(model.maxpool)
		for i in range(1,5):
			name = 'model.layer' + str(i)
			for _,x in enumerate(eval(name)) :
				conv_outs.append(x.relu1) #relu1 conv1
				conv_outs.append(x.relu2)
				conv_outs.append(x.relu3)
				if _ == 0: conv_outs.append(x.downsample[1])
	elif arch == 'resnet_56':
		conv_outs.append(model.relu) #conv1
		for i in range(1,4):
			name = 'model.layer' + str(i)
			for _,x in enumerate(eval(name)) :
				conv_outs.append(x.relu1)
				conv_outs.append(x.relu2)
	elif arch == 'googlenet':
		cfg = ['a3','b3','a4','b4','c4','d4','e4','a5','b5']
		conv_outs.append(model.pre_layers[2])
		for i,x in enumerate(cfg):
			name = 'model.inception_' + x
			conv_outs.append(eval(name + '.branch3x3')[5])
			conv_outs.append(eval(name + '.branch5x5')[5])
			conv_outs.append(eval(name + '.branch5x5')[8])
	elif arch == 'shufflenet_v2':
		conv_outs.append(model.conv1[2])
		conv_outs.append(model.maxpool)

		for stage_id in range(2,5):
			stage = 'model.stage' + str(stage_id)
			# n = len(eval(name))
			for _,IR in enumerate(eval(stage)):
				if _ == 0:
					conv_outs.append(IR.branch1[1])
					conv_outs.append(IR.branch1[4])
					conv_outs.append(IR.branch2[2])
					conv_outs.append(IR.branch2[4])
					conv_outs.append(IR.branch2[7])
				else :
					conv_outs.append(IR.branch2[2])
					conv_outs.append(IR.branch2[4])
					conv_outs.append(IR.branch2[7])

		conv_outs.append(model.conv5[2])
	print('showing feature...')

	justshowFeature(model,img,conv_outs,layer_id,rank=rank,figsize=(8,8)) #,titles=titles

def compare():
	arch = 'vgg_16_bn'
	criterion = nn.CrossEntropyLoss()
	device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
	model  = eval(arch)().to(device)
	# resume = '../pre_train_model/ImageNet/resnet50.pth'
	resume = '../pre_train_model/CIFAR-10/vgg_16_bn.pt'
	# resume = '../pre_train_model/ImageNet/mobilenet_v2.pth'
	pruned_checkpoint = torch.load(resume, map_location='cuda:0') #, map_location='cuda:0'
	# model.load_state_dict(_state_dict(pruned_checkpoint))
	model.load_state_dict(_state_dict(pruned_checkpoint['state_dict']))
	job_dir = 'C:\\Users\\huxf\\Desktop\\dyztmp\\CAF'
	# dataset = 'CIFAR10'
	# data_dir = 'G:\\data' # G:\ImageNet

	dataset = 'CIFAR10'#'ImageNet'
	data_dir = 'G:\\data' #'G:\\ImageNet' 
	trainloader,testloader = load_data(data_name = dataset, data_dir = data_dir, batch_size = 128, num_workers = 4)
	mask = eval('mask_'+arch)(model=model, job_dir=job_dir, device=device)

	acc1,acc5 = validate(0,model,0,testloader,criterion,save = False)

	# print(model)
	print(acc1)

	layer_id = 0

	rank, delegate = get_rank_onelayer(arch, model, layer_id, 0.3)
	print(len(rank),len(delegate))
	print(rank)
	print(delegate)
	print('----'*10)
	_id = layer_id
	mask.layer_mask(_id, param_per_cov=3, rank=rank, type = 1, arch=arch)
	acc1,acc5 = validate(0,model,0,testloader,criterion,save = False)
	print(acc1)

	conv_names = get_conv_names(model)
	print(conv_names[_id+1])
	adjust_filter(conv_names[_id+1],delegate,model = model)

	acc1,acc5 = validate(0,model,0,testloader,criterion,save = False)
	print(acc1)

def test():
	arch = 'resnet_50'
	device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
	model  = eval(arch)().to(device)
	resume = '../pre_train_model/ImageNet/resnet50.pth'
	# resume = '../pre_train_model/CIFAR-10/resnet_56.pt'
	pruned_checkpoint = torch.load(resume) #, map_location='cuda:0'
	model.load_state_dict(_state_dict(pruned_checkpoint))
	# model.load_state_dict(_state_dict(pruned_checkpoint['state_dict']))
	job_dir = 'C:\\Users\\huxf\\Desktop\\dyztmp\\CAF'

	
	layer_id = 0
	print(model)
	distance = get_distance_JS(arch, model, layer_id)
	print(distance[-1])
	for x in distance:
		for i in x:
			print(i)

	# rank, delegate = get_rank_onelayer(arch, model, layer_id, 0.3)
	# print(len(rank),len(delegate))
	# print(rank)
	# print(delegate)

	# for i,x in enumerate(delegate):
	# 	print(i,":",x)
	# 	pass


	# mask = eval('mask_'+arch)(model=model, job_dir=job_dir, device=device)
	# _id = 0
	# mask.layer_mask(_id, param_per_cov=3, rank=rank, type = 1, arch=args.arch)

	# for i,x in enumerate(delegate):
	# 	print(i,':',x)
	# 	pass

	# visualize_Feature_by_rank(model,arch=arch,layer_id=layer_id) #,rank = rank

	# conv_names = get_conv_names(model)
	# print(conv_names[_id+1])
	# adjust_filter(conv_names[_id+1],delegate,model = model)

	# visualize_Feature_by_rank(model,arch=arch,layer_id=layer_id+1)

	# print(torch.backends.cudnn.enabled)
	# parameters = model.named_parameters()
	# # for n,v in parameters:
	# # 	print(n,v.size())
	# bn_params = [v for n,v in parameters if 'norm' in n] #if 'norm' in n
	# print(bn_params)
	# print(len(bn_params))

def test_ensemble():
	device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
	resume1 = 'final_pruned_model/resnet_56_1230_2_9355.pt'
	resume2 = 'final_pruned_model/resnet_56_1230_5_9357.pt'
	resume3 = 'final_pruned_model/resnet_56_1229_1_9366.pt'

	checkpoint1 = torch.load(resume1, map_location='cuda:0')
	checkpoint2 = torch.load(resume2, map_location='cuda:0')
	checkpoint3 = torch.load(resume3, map_location='cuda:0')

	cp_rate1 = checkpoint1['compress_rate']

	model1 = resnet_56(compress_rate = cp_rate1).to(device)
	# model1.load_state_dict(checkpoint1['state_dict'])

	# model2 = resnet_56(compress_rate = cp_rate1).to(device)
	# model2.load_state_dict(checkpoint2['state_dict'])

	# print(cp_rate1)
	# print(model1)

	state_dict = model1.state_dict()
	# params = model1.parameters()

	cpsd1 = checkpoint1['state_dict']
	cpsd2 = checkpoint2['state_dict']
	cpsd3 = checkpoint3['state_dict']

	# for i,x in enumerate(cpsd1):
	# 	print(i,x)

	# print('+++'*10)

	for i,x in enumerate(state_dict):
		print(i,x,state_dict[x].size())
		state_dict[x] = (cpsd1[x] + cpsd2[x] + cpsd3[x])*1.005/3

	model1.load_state_dict(state_dict)

	_,testloader = load_data(data_name = 'CIFAR10', data_dir = 'G:\\data', batch_size = 128, num_workers = 2)
	criterion = nn.CrossEntropyLoss()

	validate(0,model1,0,testloader,criterion,save = False)
	# validate(0,model2,0,testloader,criterion,save = False)
	pass

def test_speedup_cifar10():
	arch = 'googlenet' # 'mobilenet_v2'
	criterion = nn.CrossEntropyLoss()
	device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
	model  = eval(arch)().to(device)
	resume = '../pre_train_model/CIFAR-10/googlenet.pt'
	pruned_checkpoint = torch.load(resume, map_location='cuda:0') #, map_location='cuda:0'
	model.load_state_dict(_state_dict(pruned_checkpoint['state_dict']))
	job_dir = 'C:\\Users\\huxf\\Desktop\\dyztmp\\CAF'

	dataset = 'CIFAR10' 
	data_dir = 'G:\\data' 
	trainloader,testloader = load_data(data_name = dataset, data_dir = data_dir, batch_size = 128, num_workers = 1)

	compress_rate = [0.2]+[0.7]*15+[0.75]*9+[0.,0.4,0.]
	# [0.2]+[0.7]*15+[0.75]*9+[0.,0.4,0.]
	# 375672192.0 2185863.0 0.7543697977203705 0.6455117778228259
	# 585733312.0 2860234.0 0.6170230456970385 0.5361469288465437
	# 9.1251  8.3743  8.4068  avg 8.6354
	# 5.9206  5.1971 5.2027 avg 5.440133333333333 0.00051971 / 0.00052027 ms  1.59
	# 6.5552  5.8230 5.8194   0.0005823  /  0.00058194 ms   1.48
	# 

	model  = eval(arch)(compress_rate).to(device)
	validate(0,model,0,testloader,criterion,save = False)
	validate(0,model,0,testloader,criterion,save = False)
	validate(0,model,0,testloader,criterion,save = False)

def test_speedup():
	arch = 'resnet_50' # 'mobilenet_v2'
	criterion = nn.CrossEntropyLoss()
	device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
	model  = eval(arch)().to(device)
	resume = '../pre_train_model/ImageNet/resnet50.pth'
	# resume = '../pre_train_model/ImageNet/mobilenet_v2.pth'
	pruned_checkpoint = torch.load(resume) #, map_location='cuda:0'
	model.load_state_dict(_state_dict(pruned_checkpoint))
	job_dir = 'C:\\Users\\huxf\\Desktop\\dyztmp\\CAF'

	dataset = 'ImageNet'#'ImageNet'
	data_dir = 'G:\\ImageNet' #'G:\\ImageNet' 
	trainloader,testloader = load_data(data_name = dataset, data_dir = data_dir, batch_size = 128, num_workers = 8)

	# compress_rate = [0.]+[0.2,0.2,0.1]*1+[0.65,0.65,0.1]*2+[0.2,0.2,0.15]*1+[0.65,0.65,0.15]*3+[0.2,0.2,0.15]*1+[0.65,0.65,0.15]*5+[0.2,0.2,0.1]+[0.2,0.2,0.1]*2
	compress_rate =[0.]+[0.2,0.2,0.2]*1+[0.75,0.75,0.2]*2+[0.2,0.2,0.2]*1+[0.75,0.75,0.2]*3+[0.2,0.2,0.2]*1+[0.75,0.75,0.2]*5+[0.2,0.2,0.2]+[0.2,0.2,0.2]*2
	# compress_rate = [0.]+[0.1]*2+[0.1]*2+[0.3]+[0.1]*2+[0.3]*2+[0.1]*2+[0.3]*3+[0.1]*2+[0.3]*2+[0.1]*2+[0.3]*2+[0.1]*2
	#resnet50 71.4244 #87.3581 1.225
	#resnet50 55.3  95.1104 92.9980 92.5516 avg 93.55 | 72.5131 70.3382 71.8511 avg 71.57   1.4ms  1.31
	#resnet50 61.7  95.1104 92.9980 92.5516 | 68.5765 66.7387 67.2707  avg67.53  1.3ms 0.04ms  1.39

	model  = eval(arch)(compress_rate).to(device)
	validate(0,model,0,testloader,criterion,save = False)
	validate(0,model,0,testloader,criterion,save = False)
	validate(0,model,0,testloader,criterion,save = False)

def ablation_resnet56():
	pass

if __name__ == '__main__':

	init()

	# test_ensemble()

	# test()
	# compare()
	# test_speedup()
	# test_speedup_cifar10()

	if args.from_scratch is True:
		train_from_scratch()
	else :
		if args.arch == 'vgg_16_bn':
			prune_vgg16bn()
		elif args.arch == 'resnet_56':
			prune_resnet_56()
		elif args.arch == 'resnet_50':
			prune_resnet_50()
		elif args.arch == 'googlenet':
			prune_googlenet()
		elif args.arch == 'mobilenet_v2':
			prune_mobilenet_v2()
		



