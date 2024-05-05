
import torch
import torch.nn as nn
import torch.nn.functional as F
VERY_NEG_NUMBER = -100000000000
VERY_SMALL_NUMBER = 1e-10


class TypeLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, linear_drop, device):
        super(TypeLayer, self).__init__()
        """
        in_features: 输入特征维度。
        out_features: 输出特征维度。
        linear_drop: 线性层中可能应用的 dropout 率。
        device: 指定模型将在哪个设备（如 CPU 或 GPU）上运行
        """
        self.in_features = in_features
        self.out_features = out_features
        self.linear_drop = linear_drop
        # self.kb_head_linear = nn.Linear(in_features, out_features)
        self.kb_self_linear = nn.Linear(in_features, out_features)
        # self.kb_tail_linear = nn.Linear(out_features, out_features)
        self.device = device

    def forward(self, local_entity, edge_list, rel_features):
        """
        input_vector: (batch_size, max_local_entity)
        curr_dist: (batch_size, max_local_entity)
        instruction: (batch_size, hidden_size)
        local_entity: 形状为 (batch_size, max_local_entity) 的张量，包含局部实体的特征向量
        rel_features: 形状为 (num_relations, in_features) 的张量，包含关系类型的特征向量
        """

        # edge_list: 包含边信息的元组，包含六个元素
        batch_heads, batch_rels, batch_tails, batch_ids, fact_ids, weight_list = edge_list
        # batch_heads: 所有边的头实体在批次内的索引。
        # batch_rels: 所有边的关系类型索引。
        # batch_tails: 所有边的尾实体在批次内的索引。
        # batch_ids: 所有边所属批次的索引。
        # fact_ids: 所有边对应的事实（即三元组：头实体、关系、尾实体）的唯一标识符。
        # weight_list: 边的权重列表（注：在代码中未使用）
        # 54138: 事实数量
        num_fact = len(fact_ids)
        batch_size, max_local_entity = local_entity.size()
        hidden_size = self.in_features
        fact2head = torch.LongTensor([batch_heads, fact_ids]).to(self.device)
        fact2tail = torch.LongTensor([batch_tails, fact_ids]).to(self.device)
        batch_rels = torch.LongTensor(batch_rels).to(self.device)
        batch_ids = torch.LongTensor(batch_ids).to(self.device)
        val_one = torch.ones_like(batch_ids).float().to(self.device)

        
        # print("Prepare data:{:.4f}".format(time.time() - st))
        # Step 1: Calculate value for every fact with rel and head
        fact_rel = torch.index_select(rel_features, dim=0, index=batch_rels)
        # fact_val = F.relu(self.kb_self_linear(fact_rel) + self.kb_head_linear(self.linear_drop(fact_ent)))
        fact_val = self.kb_self_linear(fact_rel)
        # fact_val = self.kb_self_linear(fact_rel)#self.kb_head_linear(self.linear_drop(fact_ent))

        # Step 3: Edge Aggregation with Sparse MM
        # 使用 _build_sparse_tensor 辅助方法构建稀疏张量，分别表示事实到头实体（fact2head_mat）和事实到尾实体（fact2tail_mat）的连接。
        fact2tail_mat = self._build_sparse_tensor(fact2tail, val_one, (batch_size * max_local_entity, num_fact))
        fact2head_mat = self._build_sparse_tensor(fact2head, val_one, (batch_size * max_local_entity, num_fact))

        # neighbor_rep = torch.sparse.mm(fact2tail_mat, self.kb_tail_linear(self.linear_drop(fact_val)))
        # 使用稀疏矩阵乘法（Sparse MM）计算邻接实体的特征向量。具体而言，对每个实体，其邻接实体的特征向量通过加权求和得到，权重由对应的头实体和尾实体的事实特征向量共同决定。这里只使用了 kb_self_linear 输出的 fact_val，而没有使用 kb_tail_linear。
        f2e_emb = F.relu(torch.sparse.mm(fact2tail_mat, fact_val) + torch.sparse.mm(fact2head_mat, fact_val))
        assert not torch.isnan(f2e_emb).any()
        # 将聚合后的特征向量重塑为 (batch_size, max_local_entity, hidden_size) 的形状，以便与输入的 local_entity 维度匹配。
        f2e_emb = f2e_emb.view(batch_size, max_local_entity, hidden_size)

        return f2e_emb

    def _build_sparse_tensor(self, indices, values, size):
        return torch.sparse.FloatTensor(indices, values, size).to(self.device)
