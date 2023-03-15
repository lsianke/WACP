import torch
import numpy as np
import pickle
import utils

class mask_vgg_16_bn:
    def __init__(self, model=None, job_dir='',device=None):
        self.model = model
        # self.compress_rate = compress_rate
        self.mask = {}
        self.job_dir=job_dir
        self.device = device

    def layer_mask(self, cov_id, resume=None, param_per_cov=4, rank = None, type = 1, arch="vgg_16_bn"):
        params = self.model.parameters()
        # prefix = "rank_conv/"+arch+"/rank_conv"
        # subfix = ".npy"

        if resume:
            with open(resume, 'rb') as f:
                self.mask = pickle.load(f)
        else:
            resume='mask/mask_vgg_16_bn'

        self.param_per_cov=param_per_cov

        for index, item in enumerate(params):

            if index == (cov_id + 1) * param_per_cov:
                break
            if index == cov_id * param_per_cov:
                f, c, w, h = item.size()

                # if rank is None:
                #     rank = np.load(prefix + str(cov_id) + subfix)
                #     utils.logger.info('loading '+prefix + str(cov_id) + subfix)

                # print(rank)

                if type == 1: # 表示保存rank数组中的通道
                    inds = torch.zeros(f, 1, 1, 1).to(self.device)
                    for i in rank:
                        inds[i, 0, 0, 0] = 1
                    self.mask[index] = inds  # covolutional weight
                else :
                    inds = torch.ones(f, 1, 1, 1).to(self.device)
                    for i in rank:
                        inds[i, 0, 0, 0] = 0
                    self.mask[index] = inds  # covolutional weight

                item.data = item.data * self.mask[index]

            if index > cov_id * param_per_cov and index < (cov_id + 1) * param_per_cov :
                self.mask[index] = torch.squeeze(inds)
                item.data = item.data * self.mask[index]

        with open(resume, "wb") as f:
            pickle.dump(self.mask, f)

    def layer_copy_filter(self, cov_id, resume=None, param_per_cov=4, cp_id = 0, cped = [], arch="vgg_16_bn"):
        params = self.model.parameters()
        
        for index, item in enumerate(params):
            if index == (cov_id + 1) * param_per_cov:
                break
            if index >= cov_id * param_per_cov and index < (cov_id + 1) * param_per_cov :
                for i in cped:
                    item.data[i] = item.data[cp_id]

    def grad_mask(self, cov_id):
        params = self.model.parameters()
        mask_keys = self.mask.keys()
        for index, item in enumerate(params):
            if index < cov_id * self.param_per_cov:
                continue
            if index in mask_keys:
                item.data = item.data * self.mask[index]#prune certain weight

class mask_resnet_56:
    def __init__(self, model=None, job_dir='',device=None):
        self.model = model
        # self.compress_rate = compress_rate
        self.mask = {}
        self.job_dir=job_dir
        self.device = device

    def layer_mask(self, cov_id, resume=None, param_per_cov=3, rank = None, type = 1, arch="resnet_56"):
        params = self.model.parameters()
        prefix = "rank_conv/"+arch+"/rank_conv"
        subfix = ".npy"

        # print('--->',cov_id)

        if resume:
            with open(resume, 'rb') as f:
                self.mask = pickle.load(f)
        else:
            resume='mask/mask_resnet_56'

        self.param_per_cov=param_per_cov

        for index, item in enumerate(params):

            if index == (cov_id + 1)*param_per_cov:
                break

            if index == cov_id * param_per_cov:
                f, c, w, h = item.size()
                if rank is None:
                    rank = np.load(prefix + str(cov_id) + subfix)
                    utils.logger.info('loading '+prefix + str(cov_id) + subfix)

                if type == 1:
                    inds = torch.zeros(f, 1, 1, 1).to(self.device)
                    for i in rank:
                        inds[i, 0, 0, 0] = 1
                    self.mask[index] = inds  
                else :
                    inds = torch.ones(f, 1, 1, 1).to(self.device)
                    for i in rank:
                        inds[i, 0, 0, 0] = 0
                    self.mask[index] = inds  
                item.data = item.data * self.mask[index]
            elif index > cov_id*param_per_cov and index < (cov_id + 1)*param_per_cov:
                self.mask[index] = torch.squeeze(inds)
                item.data = item.data * self.mask[index].to(self.device)

        with open(resume, "wb") as f:
            pickle.dump(self.mask, f)

    def grad_mask(self, cov_id):
        if len(self.mask.keys()) < 1 : return 
        params = self.model.parameters()
        mask_keys = self.mask.keys()
        for index, item in enumerate(params):
            if index == (cov_id + 1)*self.param_per_cov:
                break
            if index in mask_keys:
                item.data = item.data * self.mask[index].to(self.device)#prune certain weight

class mask_mobilenet_v2:
    def __init__(self, model=None, job_dir='',device=None):
        self.model = model
        # self.compress_rate = compress_rate
        self.mask = {}
        self.job_dir=job_dir
        self.device = device

    def layer_mask(self, cov_id, resume=None, param_per_cov=3, rank = None, type = 1, arch="mobilenet_v2"):
        params = self.model.parameters()
        # print('--->',cov_id)

        if resume:
            with open(resume, 'rb') as f:
                self.mask = pickle.load(f)
        else:
            resume='mask/mask_mobilenet_v2'

        self.param_per_cov=param_per_cov

        for index, item in enumerate(params):

            # print(index,item.size())

            if index == (cov_id + 1)*param_per_cov:
                break

            if index == cov_id * param_per_cov:
                f, c, w, h = item.size()
                
                if type == 1:
                    inds = torch.zeros(f, 1, 1, 1).to(self.device)
                    for i in rank:
                        inds[i, 0, 0, 0] = 1
                    self.mask[index] = inds  
                else :
                    inds = torch.ones(f, 1, 1, 1).to(self.device)
                    for i in rank:
                        inds[i, 0, 0, 0] = 0
                    self.mask[index] = inds  
                item.data = item.data * self.mask[index]
            elif index > cov_id*param_per_cov and index < (cov_id + 1)*param_per_cov:
                self.mask[index] = torch.squeeze(inds)
                item.data = item.data * self.mask[index].to(self.device)

        with open(resume, "wb") as f:
            pickle.dump(self.mask, f)

    def grad_mask(self, cov_id):
        if len(self.mask.keys()) < 1 : return 
        params = self.model.parameters()
        mask_keys = self.mask.keys()
        for index, item in enumerate(params):
            if index == (cov_id + 1)*self.param_per_cov:
                break
            if index in mask_keys:
                item.data = item.data * self.mask[index].to(self.device)#prune certain weight

class mask_densenet_40:
    def __init__(self, model=None, job_dir='',device=None):
        self.model = model
        self.job_dir=job_dir
        self.device=device
        self.mask = {}

    def layer_mask(self, cov_id, resume=None, param_per_cov=3,  rank = None, type = 1, arch="densenet_40"):
        params = self.model.parameters()
        prefix = "rank_conv/"+arch+"/rank_conv"
        subfix = ".npy"
        growthRate = 12
        trans_Layers = [13,26]
        bn_start = 0

        if resume:
            with open(resume, 'rb') as f:
                self.mask = pickle.load(f)
        else:
            resume='mask/mask_densenet_40'

        self.param_per_cov=param_per_cov

        _index = None
        for index, item in enumerate(params):

            if index == (cov_id + 1) * param_per_cov:
                _index = index - 1
                break
            if index == cov_id  * param_per_cov:
                f, c, w, h = item.size()
                # print(cov_id,'->',f, c, w, h)
                bn_start = c

                if type == 1:
                    inds = torch.zeros(f, 1, 1, 1).to(self.device)
                    for i in rank:
                        inds[i, 0, 0, 0] = 1
                    self.mask[index] = inds  
                else :
                    inds = torch.ones(f, 1, 1, 1).to(self.device)
                    for i in rank:
                        inds[i, 0, 0, 0] = 0
                    self.mask[index] = inds  
                item.data = item.data * self.mask[index]

            if index > cov_id * param_per_cov and index < (cov_id + 1) * param_per_cov :
                if cov_id>=1 and cov_id!=13 and cov_id!=26:
                    self.mask[index] = torch.cat([torch.ones(bn_start).to(self.device), torch.squeeze(inds)], 0).to(self.device)
                else:
                    self.mask[index] = torch.squeeze(inds).to(self.device)
                item.data = item.data * self.mask[index]
        self.expand_mask((cov_id + 1) * param_per_cov, _index)

        with open(resume, "wb") as f:
            pickle.dump(self.mask, f)

    def expand_mask(self, cov_id, _index_start):
        params = self.model.parameters()
        for index, item in enumerate(params):
            if index <= _index_start or len(item.size()) > 1:
                continue
            if index > 116 :
                break
            f = item.size()
            self.mask[index] = torch.cat([self.mask[_index_start], self.mask[index][len(self.mask[_index_start]):]], 0).to(self.device)
            item.data = item.data * self.mask[index]

    def grad_mask(self, cov_id):
        if len(self.mask.keys()) < 1 : return 
        params = self.model.parameters()
        mask_keys = self.mask.keys()
        for index, item in enumerate(params):
            if index == (cov_id + 1)*self.param_per_cov:
                break
            if index in mask_keys:
                item.data = item.data * self.mask[index].to(self.device)#prune certain weight

class mask_googlenet:
    def __init__(self, model=None, job_dir='',device=None):
        self.model = model
        self.mask = {}
        self.job_dir=job_dir
        self.device = device

    # per inception
    def layer_mask(self, cov_id, resume=None, param_per_cov=4, rank = None, type = 1, arch="googlenet"):
        params = self.model.parameters()
        prefix = "rank_conv/"+arch+"/rank_conv"
        subfix = ".npy"

        if resume:
            with open(resume, 'rb') as f:
                self.mask = pickle.load(f)
        else:
            resume='mask/mask_googlenet'

        self.param_per_cov=param_per_cov

        for index, item in enumerate(params):

            if index == (cov_id + 1) * param_per_cov:
                break

            if index == cov_id * param_per_cov:
                f, c, w, h = item.size()
                if rank is None:
                    rank = np.load(prefix + str(cov_id) + subfix)
                    utils.logger.info('loading '+prefix + str(cov_id) + subfix)

                if type == 1:
                    inds = torch.zeros(f, 1, 1, 1).to(self.device)
                    for i in rank:
                        inds[i, 0, 0, 0] = 1
                    self.mask[index] = inds  # covolutional weight
                else :
                    inds = torch.ones(f, 1, 1, 1).to(self.device)
                    for i in rank:
                        inds[i, 0, 0, 0] = 0
                    self.mask[index] = inds  # covolutional weight

                item.data = item.data * self.mask[index]

            if index > cov_id * param_per_cov and index < (cov_id + 1) * param_per_cov :
                self.mask[index] = torch.squeeze(inds)
                item.data = item.data * self.mask[index]

        with open(resume, "wb") as f:
            pickle.dump(self.mask, f)

    def grad_mask(self, cov_id):
        if len(self.mask.keys()) < 1 : return 
        params = self.model.parameters()
        mask_keys = self.mask.keys()
        for index, item in enumerate(params):
            if index == cov_id * self.param_per_cov + 4: 
                break
            if index in mask_keys:
                item.data = item.data * self.mask[index].to(self.device)#prune certain weight

class mask_resnet_50:
    def __init__(self, model=None, job_dir='',device=None):
        self.model = model
        self.mask = {}
        self.job_dir=job_dir
        self.device = device

    def layer_mask(self, cov_id, resume=None, param_per_cov=3, rank = None, type = 1, arch="resnet_50"):
        params = self.model.parameters()
        prefix = "rank_conv/"+arch+"/rank_conv"
        subfix = ".npy"
        downsample = [4,14,27,46]
        _map   = [0,1,2,3,3,4,5,6,7,8,9,10,11,12,12,13,14,15,16,17,18,19,20,21, 
                22,23,24,24, 25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,42,
                43,44,45,46,47,48]
        if resume:
            with open(resume, 'rb') as f:
                self.mask = pickle.load(f)
        else:
            resume='mask/mask_resnet_50'

        self.param_per_cov=param_per_cov

        for index, item in enumerate(params):

            if index == (cov_id + 1) * param_per_cov:
                break
            if index == cov_id * param_per_cov:
                f, c, w, h = item.size()

                if rank is None:
                    rank = np.load(prefix + str(_map[cov_id]) + subfix)
                    utils.logger.info('loading '+prefix + str(_map[cov_id]) + subfix)
                if type == 1:
                    inds = torch.zeros(f, 1, 1, 1).to(self.device)
                    for i in rank:
                        inds[i, 0, 0, 0] = 1
                    self.mask[index] = inds  
                else :
                    inds = torch.ones(f, 1, 1, 1).to(self.device)
                    for i in rank:
                        inds[i, 0, 0, 0] = 0
                    self.mask[index] = inds  
                item.data = item.data * self.mask[index]

            elif index > cov_id * param_per_cov and index < (cov_id + 1)* param_per_cov:
                self.mask[index] = torch.squeeze(inds)
                item.data = item.data * self.mask[index]

        with open(resume, "wb") as f:
            pickle.dump(self.mask, f)

    def grad_mask(self, cov_id):
        if len(self.mask.keys()) < 1 : return 
        params = self.model.parameters()
        mask_keys = self.mask.keys()
        for index, item in enumerate(params):
            if index == (cov_id + 1)*self.param_per_cov:
                break
            # if index < cov_id * self.param_per_cov:
            #     continue
            if index in mask_keys:
                item.data = item.data * self.mask[index].to(self.device)#prune certain weight