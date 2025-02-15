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


class PEPNet(nn.Module):
    """
    PEPNet for CTCVR problem
    """

    def __init__(self, user_feature_dict, item_feature_dict, emb_dim=128, 
                task=['click', 'like', 'comment','_5s', '_10s', '_18s'], shared_expert=1,
                hidden_dim=[128, 64], tower_dim = [64, 32], dropouts=0.5,
                output_size=1, expert_activation=F.relu, device=None):
        """
        MMOE model input parameters
        :param user_feature_dict: user feature dict include: {feature_name: (feature_unique_num, feature_index)}
        :param item_feature_dict: item feature dict include: {feature_name: (feature_unique_num, feature_index)}
        :param emb_dim: int embedding dimension
        :param n_expert: int number of experts in mmoe
        :param task_hidden_dim: list task layer hidden dimension
        :param hidden_dim: list task tower hidden dimension
        :param dropouts: task dnn drop out probability
        :param output_size: int task output size
        :param expert_activation: activation function like 'relu' or 'sigmoid'
        :param task: list of task name
        """
        super(PEPNet, self).__init__()
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
        self.num_features = len(item_feature_dict) + len(user_feature_dict)
        # n_expert = len(task)

        if device:
            self.device = device

        # embedding初始化
        user_cate_feature_nums, item_cate_feature_nums = 0, 0
        for user_cate, num in self.user_feature_dict.items():
            if num[0] > 1:
                user_cate_feature_nums += 1
                setattr(self, user_cate, nn.Embedding(num[0], emb_dim))
        for item_cate, num in self.item_feature_dict.items():
            if num[0] > 1:
                item_cate_feature_nums += 1
                setattr(self, item_cate, nn.Embedding(num[0], emb_dim))

        # user embedding + item embedding
        hidden_size = emb_dim * (user_cate_feature_nums + item_cate_feature_nums) + \
                      (len(self.user_feature_dict) - user_cate_feature_nums) + (
                              len(self.item_feature_dict) - item_cate_feature_nums)
        self.hidden_size = hidden_size

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
            task_hidden_dim = [hidden_size] + hidden_dim #embedding_dim,128,64
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
            getattr(self, '{}_tower_dnn'.format(i + 1)).add_module('{}_tower_sigmoid'.format(task_name),nn.Sigmoid())
            
        # epnet
        setattr(self,'epnet', nn.ModuleList())
        epnet_hidden_dim = [emb_dim] + hidden_dim
        for j in range(len(epnet_hidden_dim)-1):
            getattr(self, 'epnet').add_module('epnet_hidden_{}'.format(j),
                                                                            nn.Linear(epnet_hidden_dim[j], epnet_hidden_dim[j+1]))
            getattr(self, 'epnet').add_module('epnet_batchnorm_{}'.format(j),
                                                                            nn.BatchNorm1d(epnet_hidden_dim[j+1]))
            getattr(self, 'epnet').add_module('epnet_dropout_{}'.format(j),nn.Dropout(dropouts))
        getattr(self, 'epnet').add_module('epnet_laset_layer', nn.Linear(epnet_hidden_dim[-1], self.num_features))



    def forward(self, x):
        assert x.size()[1] == len(self.item_feature_dict) + len(self.user_feature_dict)
        # embedding
        user_embed_list, item_embed_list = list(), list()
        scene_embed_list = list()
        for user_feature, num in self.user_feature_dict.items():
            if user_feature != 'tab':
                if num[0] > 1:
                    user_embed_list.append(getattr(self, user_feature)(x[:, num[1]].long()))
                else:
                    user_embed_list.append(x[:, num[1]].unsqueeze(1))
            else:
                scene_embed_list.append(getattr(self, user_feature)(x[:, num[1]].long()))
                scene_feature = x[:, num[1]].long()
        for item_feature, num in self.item_feature_dict.items():
            if num[0] > 1:
                item_embed_list.append(getattr(self, item_feature)(x[:, num[1]].long()))
            else:
                item_embed_list.append(x[:, num[1]].unsqueeze(1))

        # embedding 融合
        # user_embed = torch.stack(user_embed_list, axis=1)
        # print(f'user_embed: {user_embed.shape}')
        user_embed_copy = torch.stack(user_embed_list, axis=1).detach() # batch * num_features* embedding_size 并且停止梯度
        user_embed_copy = torch.sum(user_embed_copy, dim=1) # batch * embedding_size

        # item_embed = torch.stack(item_embed_list, axis=1)
        # print(f'item_embed: {item_embed.shape}')
        item_embed_copy = torch.stack(item_embed_list, axis=1).detach() # batch * num_features* embedding_size 并且停止梯度
        item_embed_copy = torch.sum(item_embed_copy, dim=1)

        scene_embed = torch.cat(scene_embed_list, axis=1)

        total_embed = user_embed_list+item_embed_list+scene_embed_list
        hidden_input = torch.stack(total_embed, axis=1).float()  # batch * num_features* embedding_size
        # print(f'hidden_input: {hidden_input.size()}')
        
        # epnet input
        epnet_input = torch.stack([user_embed_copy, item_embed_copy, scene_embed], axis=1)  # batch * 3* embedding_size
        epnet_input = torch.sum(epnet_input, dim=1)  # batch * embedding_size
        # print(f'epnet_input: {epnet_input.size()}')
        for epnet_layer in getattr(self,'epnet'):
            epnet_input = epnet_layer(epnet_input)
        epnet_gate = nn.Sigmoid()(epnet_input)*2
        epnet_gate = epnet_gate.unsqueeze(2)
        # print(f'epnet_gate: {epnet_gate.size()}')
        hidden = hidden_input * epnet_gate
        hidden = hidden.view(-1,self.hidden_size)
        # print(f'hidden: {hidden.size()}')

        # hidden layer
        # hidden = torch.cat([user_embed, item_embed,scene_embed], axis=1).float()  # batch * hidden_size
        
        # shared-expert
        share_experts = list()
        for i in range(self.shared_expert):
            share_expert = hidden
            for share in getattr(self, 'share_{}_dnn'.format(i+1)):
                share_expert = share(share_expert)
            share_experts.append(share_expert)

        # task-expert
        task_experts = list()
        for task_name in self.task:
            task_expert = hidden
            for task_nn in getattr(self, '{}_expert_dnn'.format(task_name)):
                task_expert = task_nn(task_expert)
            task_experts.append(task_expert)

        # gate
        gates_out = list()
        for task in range(self.num_task):
            x = hidden
            for gate in getattr(self, 'task_{}_gate'.format(task+1)):
                x = gate(x)
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
        return task_outputs, scene_feature



