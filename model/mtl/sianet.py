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

class SubexpertIntegration(nn.Module):
    def __init__(self, subexpert_unit, dropouts, subexpert_num=5, sideinfo_dim=512,input_dim=None, init_scale=1.0):
        """
        子MoE架构类
        
        subexpert_unit : 子专家单元 [128, 64]
        subexpert_num : 子专家数量
        input_dim : 输入维度
        init_scale : 初始缩放因子
        
        输出：集成MoE之后的输出 (-1, subexpert_unit[-1])
        """
        super(SubexpertIntegration, self).__init__()

        self.subexpert_num = subexpert_num
        self.input_dim = input_dim
        self.subexpert_unit = subexpert_unit
        self.init_scale = init_scale
        self.dropouts = dropouts
        
        # 定义子专家网络
        hidden_dim = [self.input_dim] + self.subexpert_unit
        self.subexperts = nn.ModuleList()

        for _ in range(self.subexpert_num):
            layers = []
            for i in range(len(hidden_dim) - 1):
                layers.append(nn.Linear(hidden_dim[i], hidden_dim[i+1]))
                layers.append(nn.BatchNorm1d(hidden_dim[i+1]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(self.dropouts))
            self.subexperts.append(nn.Sequential(*layers))
        
        # 定义门控网络
        self.gate_net = nn.Sequential(
            nn.Linear(sideinfo_dim, self.subexpert_num),
            nn.Softmax(dim=1)
        )

    def forward(self, input, sideinfo):
        """
        前向传播
        
        input : 输入的embedding [-1, E]
        
        返回：集成MoE之后的输出 (-1, subexpert_unit[-1])
        """
        # 对每个子专家进行前向传播
        out = []
        for i in range(self.subexpert_num):
            out.append(self.subexperts[i](input))
        
        # 将子专家的输出堆叠起来
        out = torch.stack(out, dim=1)
        
        # 计算门控网络的输出
        gate = self.gate_net(sideinfo)
        
        # 将门控权重应用到子专家输出上，得到最终结果
        out = torch.sum(out * gate.unsqueeze(2), dim=1)
        
        return out

class SIANet(nn.Module):
    """
    HiNet for CTCVR problem
    """

    def __init__(self, user_feature_dict, item_feature_dict, emb_dim=128, 
                task=['click', 'like', 'comment','_5s', '_10s', '_18s'], shared_expert=1,
                hidden_dim=[128, 64], tower_dim = [64, 32], subexpert_unit = [512, 256], dropouts=0.3,
                output_size=1, expert_activation=F.relu, device=None, num_heads = 4,
                subexpert_num=5):
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
        super(SIANet, self).__init__()
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
        self.emb_dim = emb_dim
        self.subexpert_unit = subexpert_unit
        self.dropouts = dropouts
        self.num_heads = num_heads
        self.sideinfo_dim = 4*emb_dim
        self.SHARE_EXPERT = SubexpertIntegration(subexpert_unit=self.subexpert_unit, dropouts = self.dropouts, 
                                                 sideinfo_dim=self.sideinfo_dim, input_dim=3072, subexpert_num=subexpert_num)
        self.SCENE_EXPERT = {
            0 : SubexpertIntegration(subexpert_unit=self.subexpert_unit, dropouts = self.dropouts,
                                     sideinfo_dim=self.sideinfo_dim, input_dim=3072, subexpert_num=subexpert_num),
            1 : SubexpertIntegration(subexpert_unit=self.subexpert_unit, dropouts = self.dropouts,
                                     sideinfo_dim=self.sideinfo_dim, input_dim=3072, subexpert_num=subexpert_num),
            2 : SubexpertIntegration(subexpert_unit=self.subexpert_unit, dropouts = self.dropouts,
                                     sideinfo_dim=self.sideinfo_dim, input_dim=3072, subexpert_num=subexpert_num),
            3 : SubexpertIntegration(subexpert_unit=self.subexpert_unit, dropouts = self.dropouts,
                                     sideinfo_dim=self.sideinfo_dim, input_dim=3072, subexpert_num=subexpert_num)
        }
        # n_expert = len(task)
        if device:
            self.device = device

        # 场景特征
        self.scene_gate_nn = nn.Sequential(
            nn.Linear(emb_dim, 1),
            # nn.ReLU(),
            # nn.Linear(emb_dim // 2, 1),
            nn.Sigmoid()
        )
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
            getattr(self, '{}_tower_dnn'.format(task_name)).add_module('{}_tower_sigmoid'.format(task_name),nn.Sigmoid())
    def KL_loss(self, expert_dict):
        """
        计算存储的expert字典中两两expert之间的KL散度
        """
        kl_loss = 0
        for expert_idx, expert in expert_dict.items():
            expert = torch.mean(expert, dim=0)
            for other_expert_idx, other_expert in expert_dict.items():
                if expert_idx != other_expert_idx:
                    other_expert = torch.mean(other_expert, dim=0)
                    kl_loss += F.kl_div(F.log_softmax(expert, dim=-1), F.softmax(other_expert, dim=-1), reduction='batchmean')
        return -1*kl_loss
    
    def SIA(self, expert_dict, num_heads):
        """
        计算场景expert之间的交叉注意力
        """
        # 从字典中取出来所有的expert
        # 并且进行平均池化
        scene_1_expert = expert_dict[0].mean(dim=0, keepdim=True) #（1, 256）
        scene_2_expert = expert_dict[1].mean(dim=0, keepdim=True) #（1, 256）
        scene_3_expert = expert_dict[2].mean(dim=0, keepdim=True) #（1, 256）
        scene_4_expert = expert_dict[3].mean(dim=0, keepdim=True) #（1, 256）
        # 拼接成序列
        scene_all_expert= torch.cat([scene_1_expert, scene_2_expert, scene_3_expert, scene_4_expert], dim=0) #（4, 256）
        mha = nn.MultiheadAttention(embed_dim=256, num_heads=num_heads, batch_first=True)
        output, _ = mha(scene_all_expert, scene_all_expert, scene_all_expert)
        out = output.squeeze(0) #（4, 256）
        return out

    def forward(self, x):
        assert x.size()[1] == len(self.item_feature_dict) + len(self.user_feature_dict)
        # embedding
        user_embed_list, item_embed_list = list(), list()
        sideinfo_list = list()
        for idx, (user_feature, num) in enumerate(self.user_feature_dict.items()):
            if num[0] > 1:
                user_embed_list.append(getattr(self, user_feature)(x[:, num[1]].long()))
            else:
                user_embed_list.append(x[:, num[1]].unsqueeze(1))
            if user_feature == 'tab':
                scene_feature = x[:, num[1]].long() # 场景特征
                scene_emb = getattr(self, user_feature)(x[:, num[1]].long()).detach() #（1, 256）
                # scene_gate_input = scene_emb.clone().detach()
                sideinfo_list.append(scene_emb)
            if user_feature == 'user_id':
                userID_emb = getattr(self, user_feature)(x[:, num[1]].long()).detach()
                sideinfo_list.append(userID_emb)
        for item_feature, num in self.item_feature_dict.items():
            if num[0] > 1:
                item_embed_list.append(getattr(self, item_feature)(x[:, num[1]].long()))
            else:
                item_embed_list.append(x[:, num[1]].unsqueeze(1))
            if item_feature == 'video_id' or item_feature == 'author_id':
                sideinfo_list.append(getattr(self, item_feature)(x[:, num[1]].long()).detach())
        # embedding 融合
        user_embed = torch.cat(user_embed_list, axis=1)
        item_embed = torch.cat(item_embed_list, axis=1)
        sideinfo_embed = torch.cat(sideinfo_list, axis=1) #无梯度

        # hidden layer
        hidden = torch.cat([user_embed, item_embed], axis=1).float()  # batch * hidden_size(-1,1664)

        # 新架构
        share_scene_expert = self.SHARE_EXPERT(hidden, sideinfo_embed) # share_scene_expert (B,E)
        unique_scenes = torch.unique(scene_feature) # 场景index（去重）
        # debug
        # if len(unique_scenes) != 4:
        #     print('There are not 4 scenes')

        # 将 input 数据按场景划分
        scene_grouped_inputs = {}
        for scene in unique_scenes:
            indices = (scene_feature == scene).nonzero(as_tuple=True)[0]
            scene_grouped_inputs[scene.item()] = hidden[indices]

        # 将 sideinfo 按场景划分    
        scene_sideinfo_dict = {}
        for scene in unique_scenes:
            indices = (scene_feature == scene).nonzero(as_tuple=True)[0]
            scene_sideinfo_dict[scene.item()] = sideinfo_embed[indices]

        # 循环创建当前batch下个场景的expert
        scene_outputs = {}
        for scene,scene_input in scene_grouped_inputs.items():
            scene_sideinfo = scene_sideinfo_dict[scene]
            scene_outputs[scene] = self.SCENE_EXPERT[scene](scene_input, scene_sideinfo)

        # 注意力机制
        scene_atten_outs = {}
        atten_outputs = self.SIA(scene_outputs,self.num_heads)
        for scene, scene_expert in scene_outputs.items():
            n_sample, hidden_size = scene_expert.size()
            scene_atten_out = atten_outputs[scene,:].repeat(n_sample,1)
            scene_atten_outs[scene]= scene_atten_out

        # 拼接
        scene_merge_outs = {key: torch.cat([scene_outputs[key], scene_atten_outs[key]], dim = 1) for key in scene_outputs}


        # KL loss
        kl_loss = self.KL_loss(scene_outputs)
        
        # 准备一个空张量来存储最终拼接的结果
        _, scene_out_dim = scene_merge_outs[0].size() #取出来某一个场景的输出维度
        final_output = torch.zeros(hidden.size(0), scene_out_dim) 
        for scene, data in scene_grouped_inputs.items():
            # 获取该场景在原始数据中的索引
            indices = (scene_feature == scene).nonzero(as_tuple=True)[0]
            # 将该场景的输出放到 final_output 的对应位置
            final_output[indices] = scene_merge_outs[scene]
        
        # 对场景进行gate加权
        # scene_gate = self.scene_gate_nn(scene_emb)*2 #类似于一个权重 epnet
        hinet_output = torch.cat((share_scene_expert, final_output), dim=1)
        # hinet_output = torch.mul(hinet_output, scene_gate)

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
            for task_gate_nn in getattr(self, 'task_{}_gate'.format(task+1)):
                x = task_gate_nn(x)
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
            for tower in getattr(self, '{}_tower_dnn'.format(task_name)):
                x = tower(x)
            task_outputs.append(x)

        # kl_loss = self.KL_loss(current_scene_expert, other_scene_expert)
        return task_outputs, scene_feature, kl_loss



