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


class HiNet(nn.Module):
    """
    HiNet for CTCVR problem
    """

    def __init__(self, user_feature_dict, item_feature_dict, emb_dim=128, 
                task=['click', 'like', 'comment','_5s', '_10s', '_18s'], shared_expert=1,
                hidden_dim=[128, 64], tower_dim = [64, 32], subexpert_unit = [512, 256],dropouts=0.5,
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
        super(HiNet, self).__init__()
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
        self.subexpert_unit = subexpert_unit
        self.dropouts = dropouts
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

        # # user embedding + item embedding
        # hidden_size_1 = emb_dim * (user_cate_feature_nums + item_cate_feature_nums) + \
        #               (len(self.user_feature_dict) - user_cate_feature_nums) + (
        #                       len(self.item_feature_dict) - item_cate_feature_nums)
        hidden_size = 3 * subexpert_unit[-1] # shared+scene+SAN

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
                # getattr(self, 'share_{}_dnn'.format(i+1)).add_module('share_dropout_{}'.format(j),
                #                                                     nn.Dropout(dropouts))

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
                # getattr(self, '{}_expert_dnn'.format(task_name)).add_module('{}_expert_dropout_{}'.format(task_name, j),
                #                                                             nn.Dropout(dropouts))

            
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
                # getattr(self, '{}_tower_dnn'.format(task_name)).add_module('{}_tower_dropout_{}'.format(task_name, j),
                #                                                             nn.Dropout(dropouts))
            getattr(self,'{}_tower_dnn'.format(task_name)).add_module('{}_tower_laset_layer'.format(task_name),
                                                                            nn.Linear(tower_hidden_dim[-1], output_size))
            getattr(self, '{}_tower_dnn'.format(task_name)).add_module('{}_tower_sigmoid'.format(task_name),nn.Sigmoid())

    def subexpert_integration(self, input, subexpert_unit, subexpert_num=5, init_scale=1.0):
        """
        子MoE架构
        input : 输入的embedding [-1, E]
        prefix : 前缀
        subexpert_unit : 子专家单元 [128, 64]
        subexpert_num : 子专家数量
        输出：集成MoE之后的输出 (-1,subexpert_unit[-1])
        """
        subexperts = []
        # scales = []
        input_dim = input.size()[1]
        hidden_dim = [input_dim] + subexpert_unit
        # for i in range(len(subexpert_unit)):
        #     scales.append(init_scale / subexpert_unit[i] ** 0.5)
        for _ in range(subexpert_num):
            layers = []
            for i in range(len(hidden_dim)-1):
                layers.append(nn.Linear(hidden_dim[i], hidden_dim[i+1]))
                layers.append(nn.BatchNorm1d(hidden_dim[i+1]))
                layers.append(nn.ReLU())
                # layers.append(nn.Dropout(self.dropouts))
            subexperts.append(nn.Sequential(*layers))
        gate_net = nn.Sequential(
            nn.Linear(input_dim, subexpert_num),
            nn.Softmax(dim=1)
        )
        out = []
        for i in range(subexpert_num):
            out.append(subexperts[i](input))
        out = torch.stack(out, dim=1)
        gate = gate_net(input)
        out = torch.sum(out * gate.unsqueeze(2), dim=1)
        return out,gate
    def get_all_scenario_experts(self, input, subexpert_unit, subexpert_num=5, scenario_num=4):
        """
        得到所有的场景专家
        输出 (-1,scenario_num,E)
        """
        scenario_experts = []
        scenario_experts_gate = []
        for i in range(scenario_num):
            scenario_expert, gate = self.subexpert_integration(input, subexpert_unit, subexpert_num)
            scenario_experts.append(scenario_expert)
            scenario_experts_gate.append(gate)
        output = torch.stack(scenario_experts, dim=1)
        gate_out = torch.stack(scenario_experts_gate, dim=1)
        return output,gate_out
    def get_current_scenario_expert(self, all_experts, scenario_index):
        """
        得到当前场景专家
        all_experts: (-1,scenario_num,E)
        scenario_index: 当前场景的索引
        输出 (-1,E)
        """
        _, num_scenario, E = all_experts.size()
        mask = torch.nn.functional.one_hot(scenario_index, num_classes=num_scenario).float()
        mask = mask.unsqueeze(1)
        output = torch.matmul(mask, all_experts)
        output = output.squeeze(1)
        return output

    def SAN(self, all_experts, scenario_index, scenario_emb):
        """
        SAN模块
        input : 输入的embedding
        subexpert_unit : MoE结构的单元数
        subexpert_num : MoE结构的专家数
        scenario_index : 场景的索引
        scenario_emb : 场景的embedding
        """ 
        _, num_scenario, E = all_experts.size()
        _, scenario_emb_dim = scenario_emb.size()
        cur_scene_index = torch.nn.functional.one_hot(scenario_index, num_classes=num_scenario).float()
        cur_scene_index = cur_scene_index.unsqueeze(-1)
        mask = 1-cur_scene_index
        mask_expert = all_experts * mask
        valid_expert = mask_expert[mask.squeeze(-1).bool()]
        valid_expert = valid_expert.view(-1, num_scenario-1, E) #(-1, num_scenario-1, E)
        san_gate_net = nn.Sequential(
            nn.Linear(scenario_emb_dim, num_scenario-1),
            nn.Softmax(dim=-1)
        )
        san_gate = san_gate_net(scenario_emb)
        san_gate = san_gate.unsqueeze(-1)
        output = valid_expert * san_gate
        output =torch.sum(output, dim=1)
        return output, san_gate
    def forward(self, x):
        assert x.size()[1] == len(self.item_feature_dict) + len(self.user_feature_dict)
        # embedding
        user_embed_list, item_embed_list = list(), list()
        for idx, (user_feature, num) in enumerate(self.user_feature_dict.items()):
            if num[0] > 1:
                user_embed_list.append(getattr(self, user_feature)(x[:, num[1]].long()))
            else:
                user_embed_list.append(x[:, num[1]].unsqueeze(1))
            if user_feature == 'tab':
                scene_feature = x[:, num[1]].long() # 场景特征
                scene_idx = idx
        for item_feature, num in self.item_feature_dict.items():
            if num[0] > 1:
                item_embed_list.append(getattr(self, item_feature)(x[:, num[1]].long()))
            else:
                item_embed_list.append(x[:, num[1]].unsqueeze(1))

        # embedding 融合
        user_embed = torch.cat(user_embed_list, axis=1)
        item_embed = torch.cat(item_embed_list, axis=1)

        # hidden layer
        hidden = torch.cat([user_embed, item_embed], axis=1).float()  # batch * hidden_size(-1,1664)

        # HiNet模型结构
        share_scene_expert, share_scene_gate = self.subexpert_integration(input=hidden, subexpert_unit=self.subexpert_unit)
        all_scene_expert, all_scene_gate = self.get_all_scenario_experts(input=hidden, subexpert_unit=self.subexpert_unit)
        current_scene_expert = self.get_current_scenario_expert(all_scene_expert, scene_feature)
        scene_emb = user_embed_list[scene_idx]
        # print(scene_emb.size())
        atten_scene_expert, san_gate = self.SAN(all_scene_expert, scene_feature, scene_emb)
        # print(san_gate)
        # print(scene_feature)
        # print(f'atten_scene_expert : {atten_scene_expert.size()}')
        hinet_output = torch.cat([share_scene_expert, current_scene_expert, atten_scene_expert],axis=1)
        
        # shared-expert
        share_experts = list()
        for i in range(self.shared_expert):
            x = hinet_output
            # x = hidden
            for share_expert in getattr(self, 'share_{}_dnn'.format(i+1)):
                x = share_expert(x)
            share_experts.append(x)

        # task-expert
        task_experts = list()
        for task_name in self.task:
            x = hinet_output
            # x = hidden
            for task_expert in getattr(self, '{}_expert_dnn'.format(task_name)):
                x = task_expert(x)
            task_experts.append(x)

        # gate
        gates_out = list()
        for task in range(self.num_task):
            x = hinet_output
            # x = hidden
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
        return task_outputs, scene_feature, san_gate, share_scene_gate, all_scene_gate



