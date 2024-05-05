import torch
import numpy as np
from collections import defaultdict

VERY_NEG_NUMBER = -100000000000

class BaseGNNLayer(torch.nn.Module):
    """
    Builds sparse tensors that represent structure.
    BaseGNNLayer类继承自torch.nn.Module，用于构建表示知识图谱结构的稀疏张量。构造函数接收三个参数：

    """
    def __init__(self, args, num_entity, num_relation):
        """
        args：包含模型超参数和配置的字典，如是否使用CUDA设备、是否进行归一化等。
        num_entity：知识图谱中实体数量。
        num_relation：知识图谱中关系数量。
        """
        super(BaseGNNLayer, self).__init__()
        # 将传入参数赋值给类内相应属性（如num_entity、num_relation）
        self.num_relation = num_relation
        self.num_entity = num_entity
        self.device = torch.device('cuda' if args['use_cuda'] else 'cpu')
        # 存储是否进行归一化的标志normalized_gnn
        self.normalized_gnn = args['normalized_gnn']

    """
    此方法负责构建知识图谱的稀疏张量矩阵。方法内部
    """
    def build_matrix(self):
        # 从类属性edge_list中提取所需信息，包括头实体、尾实体、关系、ID、事实ID和权重列表
        # edge_list = kd_adj_
        batch_heads, batch_rels, batch_tails, batch_ids, fact_ids, weight_list = self.edge_list
        # 计算事实数量num_fact、批大小batch_size、最大本地实体数量max_local_entity等，并存储为类属性
        num_fact = len(fact_ids)
        num_relation = self.num_relation
        batch_size = self.batch_size
        max_local_entity = self.max_local_entity
        self.num_fact = num_fact

        # 根据头实体、尾实体、关系和事实ID，创建多个张量，如fact2head、fact2tail、head2fact、tail2fact、head2tail、rel2fact、fact2rel等，用于构建稀疏张量。这些张量均转换为与模型设备相同的数据类型。
        fact2head = torch.LongTensor([batch_heads, fact_ids]).to(self.device)
        fact2tail = torch.LongTensor([batch_tails, fact_ids]).to(self.device)
        head2fact = torch.LongTensor([fact_ids, batch_heads]).to(self.device)
        tail2fact = torch.LongTensor([fact_ids, batch_tails]).to(self.device)
        head2tail = torch.LongTensor([batch_heads, batch_tails]).to(self.device)
        rel2fact = torch.LongTensor([fact_ids, batch_rels + batch_ids * num_relation]).to(self.device)
        fact2rel = torch.LongTensor([batch_rels + batch_ids * num_relation, fact_ids]).to(self.device)
        # 将关系和ID信息存储为类属性（如batch_rels、batch_ids、batch_heads、batch_tails）
        self.batch_rels = torch.LongTensor(batch_rels).to(self.device)
        self.batch_ids = torch.LongTensor(batch_ids).to(self.device)
        self.batch_heads = torch.LongTensor(batch_heads).to(self.device)
        self.batch_tails = torch.LongTensor(batch_tails).to(self.device)
        # self.batch_ids = batch_ids
        # 根据normalized_gnn标志，确定权重列表vals的来源（实际权重或全为1的张量）。将其转换为与模型设备相同的数据类型
        if self.normalized_gnn:
            vals = torch.FloatTensor(weight_list).to(self.device)
        else:
            vals = torch.ones_like(self.batch_ids).float().to(self.device)

        # 使用_build_sparse_tensor辅助方法，根据上述创建的张量和权重列表，构建多个稀疏张量，如fact2head_mat、head2fact_mat、fact2tail_mat、tail2fact_mat、head2tail_mat、fact2rel_mat、rel2fact_mat。这些稀疏张量表示知识图谱中的各种连接关系，如头实体到事实、事实到头实体、尾实体到事实等
        #vals = torch.ones_like(self.batch_ids).float().to(self.device)
        # Sparse Matrix for reason on graph
        self.fact2head_mat = self._build_sparse_tensor(fact2head, vals, (batch_size * max_local_entity, num_fact))
        self.head2fact_mat = self._build_sparse_tensor(head2fact, vals, (num_fact, batch_size * max_local_entity))
        self.fact2tail_mat = self._build_sparse_tensor(fact2tail, vals, (batch_size * max_local_entity, num_fact))
        self.tail2fact_mat = self._build_sparse_tensor(tail2fact, vals, (num_fact, batch_size * max_local_entity))
        self.head2tail_mat = self._build_sparse_tensor(head2tail, vals, (batch_size * max_local_entity, batch_size * max_local_entity))
        self.fact2rel_mat = self._build_sparse_tensor(fact2rel, vals, (batch_size * num_relation, num_fact))
        self.rel2fact_mat = self._build_sparse_tensor(rel2fact, vals, (num_fact, batch_size * num_relation))

        """
        该辅助方法接收三个参数：
        
        indices：张量索引，形状为 (2, N)，表示稀疏张量中非零元素的行索引和列索引。
        values：张量值，形状为 (N,)，表示稀疏张量中非零元素的值。
        size：张量大小，元组形式 (M, N)，表示稀疏张量的行数和列数
        """
    def _build_sparse_tensor(self, indices, values, size):
        # 方法内部直接使用torch.sparse.FloatTensor创建一个稀疏张量，并将其转换为与模型设备相同的数据类型。最后返回构建好的稀疏张量。
        return torch.sparse.FloatTensor(indices, values, size).to(self.device)