
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch_geometric as pyg
import copy

from torch_geometric.nn import MessagePassing
from .base_gnn import BaseGNNLayer


VERY_NEG_NUMBER = -100000000000
"""
Nutrea Layer
"""
class SubgraphPool(MessagePassing):
    def __init__(self, aggr='max'):
        super().__init__(aggr=aggr)

    def forward(self, x, edge_index):
        # 这是一个基于MessagePassing的子图池化类。它在构造函数中接收一个聚合方式参数aggr（默认为'max'），并定义了前向传播方法。在前向传播中，使用propagate方法进行消息传递，并将输入的特征x和边索引edge_index作为参数。
        return self.propagate(edge_index.coalesce(), x=x)


class NuTreaLayer(BaseGNNLayer):
    """
    NuTreaLayer Reasoning
    """
    def __init__(self, args, num_entity, num_relation, entity_dim):
        super(NuTreaLayer, self).__init__(args, num_entity, num_relation)
        self.num_entity = num_entity
        self.num_relation = num_relation
        self.entity_dim = entity_dim
        self.num_expansion_ins = args['num_expansion_ins']
        self.num_backup_ins = args['num_backup_ins']
        self.num_layers = args['num_layers']
        self.context_coef = args['context_coef']
        self.backup_depth = args['backup_depth']
        self.agg = 'max'
        self.post_norm = args['post_norm']
        self.use_posemb = args['pos_emb']

        self.init_layers(args)



    def init_layers(self, args):
        # 定义激活函数（如sigmoid、softmax）、线性变换函数（如g_score_func、h_score_func、glob_lin等）和Dropout层。
        # 初始化SubgraphPool实例（用于子图池化）。
        # 根据num_layers循环创建多组关系线性层、约束线性层、融合线性层等，并添加到模型中。
        # 如果use_posemb为True，则为每层创建位置嵌入层（用于编码关系的位置信息）。
        entity_dim = self.entity_dim
        self.sigmoid = nn.Sigmoid()
        self.softmax_d1 = nn.Softmax(dim=1)
        self.g_score_func = nn.Linear(in_features=entity_dim, out_features=1)
        self.h_score_func = nn.Linear(in_features=entity_dim, out_features=1)
        self.glob_lin = nn.Linear(in_features=entity_dim, out_features=entity_dim)
        self.lin = nn.Linear(in_features=2*entity_dim, out_features=entity_dim)
        self.aggregator = SubgraphPool(aggr=self.agg)
        self.linear_dropout = args['linear_dropout']
        self.linear_drop = nn.Dropout(p=self.linear_dropout)
        # 动态决定层数
        for i in range(self.num_layers):
            self.add_module('rel_linear' + str(i), nn.Linear(in_features=entity_dim, out_features=entity_dim))
            self.add_module('con_linear' + str(i), nn.Linear(in_features=entity_dim, out_features=entity_dim))
            self.add_module('e2e_linear' + str(i), nn.Linear(in_features=2*self.num_expansion_ins*entity_dim + entity_dim, out_features=entity_dim))
            self.add_module('s2e_linear' + str(i), nn.Linear(in_features=2*self.num_backup_ins*entity_dim + entity_dim, out_features=entity_dim))
            self.add_module('e2e_linear2' + str(i), nn.Linear(in_features=entity_dim, out_features=entity_dim))
            self.add_module('s2e_linear2' + str(i), nn.Linear(in_features=entity_dim, out_features=entity_dim))
            self.add_module('pos_emb' + str(i), nn.Embedding(self.num_relation, entity_dim))
            self.add_module('pos_emb_inv' + str(i), nn.Embedding(self.num_relation, entity_dim))

        self.lin_m =  nn.Linear(in_features=(self.num_expansion_ins)*entity_dim, out_features=entity_dim)
        heads = 2
        dims = entity_dim
        dropout_pro = 0.4
        self.MultiHeadLayer = torch.nn.MultiheadAttention(embed_dim=entity_dim, num_heads=heads, dropout=dropout_pro, batch_first=True)
        self.MultiHeadLayer2 = torch.nn.MultiheadAttention(embed_dim=entity_dim, num_heads=heads, dropout=dropout_pro, batch_first=True)
    def init_reason(self, local_entity, kb_adj_mat, local_entity_emb, rel_features, rel_features_inv, query_entities, init_dist, query_node_emb=None):
        # 初始化推理过程，包括构建稀疏矩阵和设置批次信息。
        # 计算实体掩码、存储批大小和最大本地实体数量。
        # 将输入参数赋值给类内相应属性。
        # 构建相关矩阵（如head2tail_mat、tail2head_mat等）。
        # 初始化叶节点分布（即初始推理状态）。
        batch_size, max_local_entity = local_entity.size()
        self.local_entity_mask = (local_entity != self.num_entity).float()
        self.batch_size = batch_size
        self.max_local_entity = max_local_entity
        self.edge_list = kb_adj_mat
        # 全局relation
        self.rel_features = rel_features
        self.rel_features_inv = rel_features_inv
        self.local_entity_emb = local_entity_emb
        self.num_relation = self.rel_features.size(0)
        self.possible_cand = []
        self.build_matrix()
        self.query_entities = query_entities
        self.leaf_nodes = init_dist.detach()
       

    def reason_layer(self, curr_dist, instruction, rel_linear, pos_emb, inverse=False):
        """
        Aggregates neighbor representations
        """
        # 该方法对邻居表示进行聚合，接收当前分布、指令（如关系指令、约束指令）、关系线性层、位置嵌入层等作为参数。根据inverse标志判断是否处理反向关系。方法内部：

        # 获取当前关系特征和指令特征。
        # 计算关系与指令之间的交互（乘积），并应用ReLU激活。
        # 若使用位置嵌入，则将其与关系特征相加后参与交互计算。
        # 根据inverse标志选择合适的邻接矩阵，计算邻居节点对当前节点的影响。
        # 返回更新后的邻居表示。
        batch_size = self.batch_size
        max_local_entity = self.max_local_entity
        # num_relation = self.num_relation
        rel_features = self.rel_features_inv if inverse else self.rel_features
        
        fact_rel = torch.index_select(rel_features, dim=0, index=self.batch_rels)  # total edge num x D
        
        fact_query = torch.index_select(instruction, dim=0, index=self.batch_ids)  # total edge num x D
        if pos_emb is not None:
            pe = pos_emb(self.batch_rels)
            # fact_rel = torch.cat([fact_rel, pe], 1)
            fact_val = F.relu((rel_linear(fact_rel)+pe) * fact_query)
        else :
            fact_val = F.relu(rel_linear(fact_rel) * fact_query)
            
        if inverse :
            fact_prior = torch.sparse.mm(self.tail2fact_mat, curr_dist.view(-1, 1))
            fact_val = fact_val * fact_prior
            f2e_emb = torch.sparse.mm(self.fact2head_mat, fact_val)
        else :
            fact_prior = torch.sparse.mm(self.head2fact_mat, curr_dist.view(-1, 1))
            fact_val = fact_val * fact_prior
            f2e_emb = torch.sparse.mm(self.fact2tail_mat, fact_val)

        assert not torch.isnan(f2e_emb).any()
        neighbor_rep = f2e_emb.view(batch_size, max_local_entity, self.entity_dim)
        # neighbor_rep, _ = self.MultiHeadLayer2(neighbor_rep,neighbor_rep,neighbor_rep)
        return neighbor_rep
    # 这三个方法分别为设置批处理邻接矩阵、获取下一层叶节点和执行子图池化。这些方法用于辅助推理过程中对知识图谱结构的操作和信息传递。
    def set_batch_adj(self):
        adj = self.head2tail_mat.transpose(1,0).coalesce()
        idx, vals = pyg.utils.remove_self_loops(adj.indices(), adj.values())
        self.adj = torch.sparse_coo_tensor(idx, vals, adj.size()).coalesce()
        
        adj_inv = self.head2tail_mat.coalesce()
        idx, vals = pyg.utils.remove_self_loops(adj_inv.indices(), adj_inv.values())
        self.adj_inv = torch.sparse_coo_tensor(idx, vals, adj_inv.size()).coalesce()


    def get_next_leaf(self, leaf_nodes, inverse=False):
        adj = self.adj if not inverse else self.adj_inv
        x = torch.sparse.mm(adj, leaf_nodes.view(-1, 1)).view(leaf_nodes.shape)
        return (x > 0.0).float()
    

    def pool_subgraph(self, leaf_nodes, constrainer, con_linear, depth=1, inverse=False):

        batch_size = self.batch_size
        max_local_entity = self.max_local_entity

        # leaf node acculation
        leaf_nodes = self.get_next_leaf(leaf_nodes, inverse=inverse)
        leaf_nodes_list = [leaf_nodes]
        for _ in range(depth-1) :
            leaf_nodes = self.get_next_leaf(leaf_nodes, inverse=inverse)
            leaf_nodes_list.append((leaf_nodes_list[-1] + leaf_nodes > 0.0).float())
        leaf_nodes_list.reverse()

        # query for constraints
        rel_features = self.rel_features_inv if inverse else self.rel_features
        fact_rel = torch.index_select(rel_features, dim=0, index=self.batch_rels)
        con_query = torch.index_select(constrainer, dim=0, index=self.batch_ids)
        con_val = F.relu(con_linear(fact_rel) * con_query)

        # neighbor relation aggregation
        fact2node_mat = self.fact2tail_mat if inverse else self.fact2head_mat
        # 聚合公式(7)
        con_pooled = self.aggregator(con_val, fact2node_mat)
        con_pooled = leaf_nodes_list[0].flatten().unsqueeze(-1) * con_pooled
        if depth > 1 :
            adj = self.head2tail_mat.transpose(1,0).coalesce() if inverse else self.head2tail_mat.coalesce()
            idx, vals = pyg.utils.add_remaining_self_loops(adj.indices(), adj.values())
            adj = torch.sparse_coo_tensor(idx, vals, adj.size()).coalesce()
            for d in range(depth-1) :
                con_pooled = self.aggregator(con_pooled, adj)
                con_pooled = leaf_nodes_list[d+1].flatten().unsqueeze(-1) * con_pooled
        con_pooled, _ = self.MultiHeadLayer(con_pooled.view(batch_size, max_local_entity, self.entity_dim),
                                            con_pooled.view(batch_size, max_local_entity, self.entity_dim),
                                            con_pooled.view(batch_size, max_local_entity, self.entity_dim))

        # pooled_rep = con_pooled.view(batch_size, max_local_entity, self.entity_dim)
        return con_pooled
        # return pooled_rep
    # ============================

    def forward(self, current_dist, relational_ins, relational_con, step=0, return_score=False):
        # forward 函数是 NuTreaLayer 类的核心方法，负责计算下一时刻的概率向量和当前节点表示。函数接收五个参数
        # current_dist：当前节点的概率分布，形状为 (batch_size, max_local_entity)
        # relational_ins：关系指令，形状为 (batch_size, num_relations, entity_dim)
        # relational_con：约束指令，形状为 (batch_size, num_constraints, entity_dim)
        # step：当前推理步骤，默认为0。
        # return_score：布尔值，指示是否返回未归一化的得分，默认为False
        """
        Compute next probabilistic vectors and current node representations.
        """
        # 函数首先根据当前步数 step 获取相关层和函数：
        #
        # rel_linear：对应步数的关系线性层。
        # con_linear：对应步数的约束线性层。
        # e2e_linear、e2e_linear2、s2e_linear、s2e_linear2：用于融合局部实体和邻居/子图表示的线性层。
        # pos_emb、pos_emb_inv：如果开启位置嵌入（self.use_posemb），则获取对应步数的关系位置嵌入层及其反向版本；否则设为None。
        # expansion_score_func、backup_score_func：分别用于计算扩张得分和备份得分的线性层。

        rel_linear = getattr(self, 'rel_linear' + str(step))
        con_linear = getattr(self, 'con_linear' + str(step))
        e2e_linear = getattr(self, 'e2e_linear' + str(step))
        s2e_linear = getattr(self, 's2e_linear' + str(step))
        e2e_linear2 = getattr(self, 'e2e_linear2' + str(step))
        s2e_linear2 = getattr(self, 's2e_linear2' + str(step))
        if self.use_posemb :
            pos_emb = getattr(self, 'pos_emb' + str(step))
            pos_emb_inv = getattr(self, 'pos_emb_inv' + str(step))
        else :
            pos_emb, pos_emb_inv = None, None
        # score_func = getattr(self, 'score_func' + str(step))
        expansion_score_func = self.g_score_func
        backup_score_func = self.h_score_func

        # 接着调用 set_batch_adj() 方法设置批处理邻接矩阵，为后续消息传递做准备。
        self.set_batch_adj()
        
        # 接下来，函数按照“扩张”、“回溯”和“分布更新”三个阶段进行计算
        """
        Expansion
        """
        neighbor_reps = []

        for j in range(relational_ins.size(1)):
            # 对于关系指令中的每一个关系，利用 reason_layer 方法计算邻居表示，并分别考虑正向和反向关系。将所有邻居表示堆叠到一起（按特征维度堆叠），得到形状为 (batch_size, max_local_entity, 2*entity_dim) 的邻居表示集合 neighbor_reps。

            # we do the same procedure for existing and inverse relations
            neighbor_rep = self.reason_layer(current_dist, relational_ins[:,j,:], rel_linear, pos_emb) # B x 2000 x D
            neighbor_reps.append(neighbor_rep)

            neighbor_rep = self.reason_layer(current_dist, relational_ins[:,j,:], rel_linear, pos_emb_inv, inverse=True)
            neighbor_reps.append(neighbor_rep)
             # 将当前局部实体表示与邻居表示拼接，经过两次线性变换（e2e_linear 和 e2e_linear2，中间加入ReLU激活和Dropout），更新局部实体表示 self.local_entity_emb。
        neighbor_reps = torch.cat(neighbor_reps, dim=2)
        #
        next_local_entity_emb = torch.cat((self.local_entity_emb, neighbor_reps), dim=2)
        # 局部实体表示更新
        # next_local_entity_emb= self.MultiHeadLayer2(next_local_entity_emb,next_local_entity_emb,next_local_entity_emb)
        processed_emb = e2e_linear2(F.relu(e2e_linear(self.linear_drop(next_local_entity_emb))))
        self.local_entity_emb,_ = self.MultiHeadLayer2(processed_emb,processed_emb,processed_emb)
        # 利用 expansion_score_func 计算扩张得分 expansion_score，形状为 (batch_size, max_local_entity)。
        expansion_score = expansion_score_func(self.linear_drop(self.local_entity_emb)).squeeze(dim=2)


        """
        Backup
        对于约束指令中的每一个约束，利用 pool_subgraph 方法分别计算正向和反向子图表示，并将它们堆叠在一起。将更新后的局部实体表示与子图表示拼接，经过两次线性变换（s2e_linear 和 s2e_linear2，中间加入ReLU激活和Dropout），再次更新局部实体表示 self.local_entity_emb。
        """
        # list(6): 8, 2000, 100
        subgraph_reps = []
        for j in range(relational_con.size(1)):
            pooled_rep = self.pool_subgraph(self.leaf_nodes, relational_con[:,j,:], con_linear, depth=self.backup_depth)
            subgraph_reps.append(pooled_rep)

            pooled_rep = self.pool_subgraph(self.leaf_nodes, relational_con[:,j,:], con_linear, depth=self.backup_depth, inverse=True)
            subgraph_reps.append(pooled_rep)
        #
        #subgraph_reps = Cv
        subgraph_reps = torch.cat(subgraph_reps, dim=2)

        # 论文公式(8)
        next_local_entity_emb = torch.cat((self.local_entity_emb, subgraph_reps), dim=2)
        self.local_entity_emb = s2e_linear2(F.relu(s2e_linear(self.linear_drop(next_local_entity_emb))))
        # 利用 backup_score_func 计算备份得分 backup_score，形状为 (batch_size, max_local_entity)
        backup_score = backup_score_func(self.linear_drop(self.local_entity_emb)).squeeze(dim=2)

        # 合并扩张得分和备份得分（乘以系数 self.context_coef），得到总得分 score_tp
        score_tp = expansion_score + self.context_coef * backup_score


        """
        Distribution Update
        """

        # 创建 answer_mask，表示每个实体是否属于当前知识图谱（即非虚拟节点）。
        answer_mask = self.local_entity_mask
        # 将 score_tp 添加到可能候选实体列表 self.possible_cand。
        self.possible_cand.append(answer_mask)
        # 如果启用后置归一化（self.post_norm），则保存未归一化的得分 prenorm_score。
        prenorm_score = score_tp if self.post_norm else None
        # 对 score_tp 进行边界处理，用 VERY_NEG_NUMBER 填充不在知识图谱内的实体得
        score_tp = score_tp + (1 - answer_mask) * VERY_NEG_NUMBER
        # 使用 softmax_d1 对 score_tp 进行归一化，得到下一时刻的概率分布 current_dist
        current_dist = self.softmax_d1(score_tp)
        # 根据 return_score 参数决定返回内容
        if return_score:
            # 若 return_score=True，返回未归一化的得分 prenorm_score、概率分布 current_dist、更新后的局部实体表示 self.local_entity_emb，以及单独的扩张得分和备份得分（已进行边界处理）。
            return prenorm_score, current_dist, self.local_entity_emb, \
                expansion_score + (1 - answer_mask) * VERY_NEG_NUMBER, backup_score + (1 - answer_mask) * VERY_NEG_NUMBER
        
        # 若 return_score=False，仅返回未归一化的得分 prenorm_score、概率分布 current_dist 和更新后的局部实体表示 self.local_entity_emb。
        return prenorm_score, current_dist, self.local_entity_emb


