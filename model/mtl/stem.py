'''
Reference:
    [1]Jiaqi Ma et al. Modeling task relationships in multi-task learning with multi-gate mixture-of-experts. In Proceedings of the 24th ACM SIGKDD
    International Conference on Knowledge Discovery & Data Mining, pages 1930–1939, 2018.
Reference:
    https://github.com/busesese/MultiTaskModel
'''
import torch
import torch.nn as nn
from torch.nn import functional as F
from collections import defaultdict


class STEM(nn.Module):
    """
    STEM for CTCVR problem
    """

    def __init__(self, user_feature_dict, item_feature_dict, emb_dim=128, 
                task=['click', 'like', 'comment','_5s', '_10s', '_18s'], shared_expert=1,
                hidden_dim=[128, 64], tower_dim = [64, 32], dropouts=0.5,
                output_size=1, expert_activation=F.relu, device=None):
        """
        STEM模型：
        在embedding阶段分share和task分别初始化
        每一个task expert分子MoE结构
        """
        super(STEM, self).__init__()
        # check input parameters
        if user_feature_dict is None or item_feature_dict is None:
            raise Exception("input parameter user_feature_dict and item_feature_dict must be not None")
        if isinstance(user_feature_dict, dict) is False or isinstance(item_feature_dict, dict) is False:
            raise Exception("input parameter user_feature_dict and item_feature_dict must be dict")

        self.user_feature_dict = user_feature_dict
        self.item_feature_dict = item_feature_dict
        self.expert_activation = expert_activation
        self.num_task = len(task)
        self.shared_expert = shared_expert
        self.task = task
        # n_expert = len(task)

        if device:
            self.device = device

        # share task embedding
        user_cate_feature_nums, item_cate_feature_nums = 0, 0
        for user_cate, num in self.user_feature_dict.items():
            if num[0] > 1:
                user_cate_feature_nums += 1
                setattr(self, 'share_task_{}_emb'.format(user_cate), nn.Embedding(num[0], emb_dim))
        for item_cate, num in self.item_feature_dict.items():
            if num[0] > 1:
                item_cate_feature_nums += 1
                setattr(self, 'share_task_{}_emb'.format(item_cate), nn.Embedding(num[0], emb_dim))
                # user embedding + item embedding
        hidden_size = emb_dim * (user_cate_feature_nums + item_cate_feature_nums) + \
                      (len(self.user_feature_dict) - user_cate_feature_nums) + (
                              len(self.item_feature_dict) - item_cate_feature_nums)
        # 对每一个task，都进行一次embedding
        for task_name in task:
            for user_cate, num in self.user_feature_dict.items():
                if num[0] > 1:
                    setattr(self, '{}_task_{}_emb'.format(task_name, user_cate), nn.Embedding(num[0], emb_dim))
            for item_cate, num in self.item_feature_dict.items():
                if num[0] > 1:
                    setattr(self, '{}_task_{}_emb'.format(task_name, item_cate), nn.Embedding(num[0], emb_dim))


        # shared-expert
        # input_dim = embedding_dim
        # output_dim = 64
        for i in range(self.shared_expert):
            setattr(self, 'share_{}_dnn'.format(i+1), nn.ModuleList())
            share_hidden_dim = [hidden_size] + hidden_dim # embedding_dim -> 128 -> 64
            for j in range(len(share_hidden_dim)-1):
                # j= 0,1
                getattr(self, 'share_{}_dnn'.format(i+1)).add_module('share_hidden_{}'.format(j),
                                                                    nn.Linear(share_hidden_dim[j], share_hidden_dim[j+1]))
                getattr(self, 'share_{}_dnn'.format(i+1)).add_module('share_batchnorm_{}'.format(j),
                                                                    nn.BatchNorm1d(share_hidden_dim[j+1]))
                getattr(self, 'share_{}_dnn'.format(i+1)).add_module('share_dropout_{}'.format(j),
                                                                    nn.Dropout(dropouts))

        # task-expert
        # input_dim = embedding_dim
        # output_dim = 64
        for task_name in task:
            setattr(self, '{}_expert_dnn'.format(task_name), nn.ModuleList())
            task_hidden_dim = [hidden_size*6] + hidden_dim #embedding_dim,128,64
            for j in range(len(task_hidden_dim)-1):
                getattr(self, '{}_expert_dnn'.format(task_name)).add_module('{}_expert_hidden_{}'.format(task_name, j),
                                                                            nn.Linear(task_hidden_dim[j], task_hidden_dim[j+1]))
                getattr(self, '{}_expert_dnn'.format(task_name)).add_module('{}_expert_batchnorm_{}'.format(task_name, j),
                                                                            nn.BatchNorm1d(task_hidden_dim[j+1]))
                getattr(self, '{}_expert_dnn'.format(task_name)).add_module('{}_expert_dropout_{}'.format(task_name, j),
                                                                            nn.Dropout(dropouts))

            
        # gates
        for n_task in range(self.num_task):
            setattr(self, 'task_{}_gate'.format(n_task+1), nn.ModuleList())
            getattr(self, 'task_{}_gate'.format(n_task+1)).add_module('linear', nn.Linear(hidden_size, shared_expert+1, device=self.device))
            getattr(self, 'task_{}_gate'.format(n_task+1)).add_module('softmax', nn.Softmax(dim=-1))
        #self.gates = [torch.nn.Parameter(torch.rand(hidden_size, shared_expert+1), requires_grad=True) for _ in
        #              range(self.num_task)] # gate 形状是[-1,2,64]
        #for gate in self.gates:
        #    gate.data.normal_(0, 1)
        #self.gates_bias = [torch.nn.Parameter(torch.rand(shared_expert+1), requires_grad=True) for _ in range(self.num_task)]


        # tower
        for task_name in task:
            setattr(self, '{}_tower_dnn'.format(task_name), nn.ModuleList())
            tower_hidden_dim = tower_dim #embedding_dim,128,64
            for j in range(len(tower_hidden_dim)-1):
                getattr(self, '{}_tower_dnn'.format(task_name)).add_module('{}_tower_hidden_{}'.format(task_name, j),
                                                                            nn.Linear(tower_hidden_dim[j], tower_hidden_dim[j+1]))
                getattr(self, '{}_tower_dnn'.format(task_name)).add_module('{}_tower_batchnorm_{}'.format(task_name, j),
                                                                            nn.BatchNorm1d(tower_hidden_dim[j+1]))
                getattr(self, '{}_tower_dnn'.format(task_name)).add_module('{}_tower_dropout_{}'.format(task_name, j),
                                                                            nn.Dropout(dropouts))
            getattr(self,'{}_tower_dnn'.format(task_name)).add_module('{}_tower_laset_layer'.format(task_name),
                                                                            nn.Linear(tower_hidden_dim[-1], output_size))
            getattr(self, '{}_tower_dnn'.format(task_name)).add_module('{}_tower_sigmoid'.format(task_name),nn.Sigmoid())


    def forward(self, x):
        assert x.size()[1] == len(self.item_feature_dict) + len(self.user_feature_dict)

        # share-embedding
        share_user_embed_list, share_item_embed_list = list(), list()
        for user_feature, num in self.user_feature_dict.items():
            if num[0] > 1:
                share_user_embed_list.append(getattr(self, 'share_task_{}_emb'.format(user_feature))(x[:, num[1]].long()))
            else:
                share_user_embed_list.append(x[:, num[1]].unsqueeze(1))
            if user_feature == 'tab':
                scene_feature = x[:, num[1]].long() # 场景特征
        for item_feature, num in self.item_feature_dict.items():
            if num[0] > 1:
                share_item_embed_list.append(getattr(self, 'share_task_{}_emb'.format(item_feature))(x[:, num[1]].long()))
            else:
                share_item_embed_list.append(x[:, num[1]].unsqueeze(1))

        # embedding 融合
        share_user_embed = torch.cat(share_user_embed_list, axis=1)
        share_item_embed = torch.cat(share_item_embed_list, axis=1)

        # task-embedding
        task_user_embed_dict = defaultdict(list)
        task_item_embed_dict = defaultdict(list)
        for task_name in self.task:
            task_user_embed_list = list()
            for user_feature, num in self.user_feature_dict.items():
                if num[0] > 1:
                    task_user_embed_list.append(getattr(self, '{}_task_{}_emb'.format(task_name, user_feature))(x[:, num[1]].long()))
                else:
                    task_user_embed_list.append(x[:, num[1]].unsqueeze(1))
            task_item_embed_list = list()
            for item_feature, num in self.item_feature_dict.items():
                if num[0] > 1:
                    task_item_embed_list.append(getattr(self, '{}_task_{}_emb'.format(task_name, item_feature))(x[:, num[1]].long()))
                else:
                    task_item_embed_list.append(x[:, num[1]].unsqueeze(1))
            task_user_embed_dict[task_name] = torch.cat(task_user_embed_list, axis=1)
            task_item_embed_dict[task_name] = torch.cat(task_item_embed_list, axis=1)
        # 总的分任务的embedding
        task_embed_dict = dict()
        for task_name in self.task:
            task_embed_dict[task_name] = torch.cat([task_user_embed_dict[task_name], task_item_embed_dict[task_name]], axis=1)

            
        # hidden layer
        share_hidden = torch.cat([share_user_embed, share_item_embed], axis=1).float()  # batch * hidden_size
        #print(hidden.size())
        
        # shared-expert
        share_experts = list()
        for i in range(self.shared_expert):
            share_expert = share_hidden
            for share in getattr(self, 'share_{}_dnn'.format(i+1)):
                share_expert = share(share_expert)
            share_experts.append(share_expert)

        # task-expert
        task_experts = list()
        for task_name in self.task:
            task_expert = list()
            for task_embed_name, task_embed in task_embed_dict.items():
                if task_name == task_embed_name:
                    task_expert.append(task_embed)
                else:
                    task_expert.append(task_embed.detach().clone())
            task_expert = torch.cat(task_expert, axis=1)
            for task_nn in getattr(self, '{}_expert_dnn'.format(task_name)):
                task_expert = task_nn(task_expert)
            task_experts.append(task_expert)

        # gate
        gates_out = list()
        for task in range(self.num_task):
            x = share_hidden
            for nn in getattr(self, 'task_{}_gate'.format(task+1)):
                x = nn(x)
            gates_out.append(x)

        # 加权
        all_cgc = list()
        for i, task_name in enumerate(self.task):
            task_expert = torch.unsqueeze(task_experts[i], 1) # batch * 1 * num_experts
            share_expert = torch.unsqueeze(share_experts[0], 1) # batch * 1 * num_experts
            combined_expert_outputs = torch.cat((share_expert,task_expert), dim=1)
            weighted = gates_out[0].unsqueeze(-1) * combined_expert_outputs
            task_out = torch.sum(weighted, dim=1)
            all_cgc.append(task_out)

        # task tower
        task_outputs = list()
        for i, task_name in enumerate(self.task):
            x = all_cgc[i]
            for mod in getattr(self, '{}_tower_dnn'.format(task_name)):
                x = mod(x)
            task_outputs.append(x)
        return task_outputs,scene_feature



