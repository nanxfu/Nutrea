import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn

from models.base_model import BaseModel
from modules.kg_reasoning.nutrealayer import NuTreaLayer
from modules.question_encoding.lstm_encoder import LSTMInstruction
from modules.question_encoding.bert_encoder import BERTInstruction
from modules.layer_init import TypeLayer
from modules.query_update import AttnEncoder, Fusion, QueryReform

VERY_SMALL_NUMBER = 1e-10
VERY_NEG_NUMBER = -100000000000
"""
full inference
"""


class NuTrea(BaseModel):
    def __init__(self, args, num_entity, num_relation, num_word):
        """
        Init NuTrea model.
        """
        super(NuTrea, self).__init__(args, num_entity, num_relation, num_word)
        self.args = args
        # True
        self.rf_ief = args['rf_ief']
        self.layers(args)
        self.EF = 0
        self.V = 0
        self.IEF = nn.Parameter(torch.zeros((num_relation+1)*2), requires_grad=False)
        self.emb_cache = {}

        # kl
        self.loss_type =  args['loss_type']
        # 2
        self.num_iter = args['num_iter']
        # 2
        self.num_layers = args['num_layers']
        # 2
        self.num_expansion_ins = args['num_expansion_ins']
        self.num_backup_ins = args['num_backup_ins']
        self.lm = args['lm']
        self.post_norm = args['post_norm']

        self.private_module_def(args, num_entity, num_relation)

        self.to(self.device)
        self.lin = nn.Linear(3*self.entity_dim, self.entity_dim)

        self.fusion = Fusion(self.entity_dim)
        self.reforms = []
        for i in range(self.num_expansion_ins):
            self.add_module('reform' + str(i), QueryReform(self.entity_dim))
        for i in range(self.num_backup_ins):
            self.add_module('conreform' + str(i), QueryReform(self.entity_dim))
        # self.reform_rel = QueryReform(self.entity_dim)
        # self.add_module('reform', QueryReform(self.entity_dim))
        self.freezed_bsize = nn.Parameter(torch.Tensor([args['test_batch_size']]), requires_grad=False)


    def layers(self, args):
        # initialize entity embedding
        # 768
        word_dim = self.word_dim
        # 100
        kg_dim = self.kg_dim
        # 100
        entity_dim = self.entity_dim

        #self.lstm_dropout = args['lstm_dropout']
        # 0.3
        self.linear_dropout = args['linear_dropout']

        self.entity_linear = nn.Linear(in_features=self.ent_dim, out_features=entity_dim)
        # self.relation_linear = nn.Linear(in_features=self.rel_dim, out_features=entity_dim)
        # self.relation_linear_inv = nn.Linear(in_features=self.rel_dim, out_features=entity_dim)
        #self.relation_linear = nn.Linear(in_features=self.rel_dim, out_features=entity_dim)

        # dropout
        #self.lstm_drop = nn.Dropout(p=self.lstm_dropout)
        self.linear_drop = nn.Dropout(p=self.linear_dropout)

        if self.rf_ief :
            self.rfief_linear = nn.Linear(entity_dim, entity_dim)
        elif self.encode_type :
            self.type_layer = TypeLayer(in_features=entity_dim, out_features=entity_dim,
                                        linear_drop=self.linear_drop, device=self.device)

        self.self_att_r = AttnEncoder(self.entity_dim)
        #self.self_att_r_inv = AttnEncoder(self.entity_dim)
        self.softmax_d1 = nn.Softmax(dim=1)
        self.kld_loss = nn.KLDivLoss(reduction='none')
        self.bce_loss_logits = nn.BCEWithLogitsLoss(reduction='none')
        self.mse_loss = torch.nn.MSELoss()

    def get_ent_init(self, local_entity, kb_adj_mat, rel_features):
        # True
        if self.encode_type:
            local_entity_emb = self.type_layer(local_entity=local_entity,
                                               edge_list=kb_adj_mat,
                                               rel_features=rel_features)
        else:
            local_entity_emb = self.entity_embedding(local_entity)  # batch_size, max_local_entity, word_dim
            local_entity_emb = self.entity_linear(local_entity_emb)

        return local_entity_emb


    def get_rel_feature(self):
        """
        Encode relation tokens to vectors.
        """
        if self.rel_texts is None:
            rel_features = self.relation_embedding.weight
            rel_features_inv = self.relation_embedding_inv.weight
            rel_features = self.relation_linear(rel_features)
            rel_features_inv = self.relation_linear(rel_features_inv)
        else:
            rel_features = self.instruction.question_emb(self.rel_features)
            rel_features_inv = self.instruction.question_emb(self.rel_features_inv)

            if self.lm == 'lstm':
                rel_features = self.self_att_r(rel_features, (self.rel_texts != self.num_relation+1).float())
                rel_features_inv = self.self_att_r(rel_features_inv, (self.rel_texts_inv != self.num_relation+1).float())
            else :
                rel_features = self.self_att_r(rel_features,  (self.rel_texts != self.instruction.pad_val).float())
                rel_features_inv = self.self_att_r(rel_features_inv,  (self.rel_texts != self.instruction.pad_val).float())

        return rel_features, rel_features_inv


    def private_module_def(self, args, num_entity, num_relation):
        """
        Building modules: LM encoder, GNN, etc.
        """
        # initialize entity embedding
        word_dim = self.word_dim
        kg_dim = self.kg_dim
        entity_dim = self.entity_dim
        # 用NutraLayer层代替推理GNN层
        self.reasoning = NuTreaLayer(args, num_entity, num_relation, entity_dim)
        if args['lm'] == 'lstm':
            self.instruction = LSTMInstruction(args, self.word_embedding, self.num_word)
            self.constraint = LSTMInstruction(args, self.word_embedding, self.num_word)
            self.relation_linear = nn.Linear(in_features=entity_dim, out_features=entity_dim)
        else:
            self.instruction = BERTInstruction(args, self.word_embedding, self.num_word, args['lm'])
            self.constraint = BERTInstruction(args, self.word_embedding, self.num_word, args['lm'], constraint=True)
            #self.relation_linear = nn.Linear(in_features=self.instruction.word_dim, out_features=entity_dim)
        # 多头注意力层
        # self.LN = nn.LayerNorm
        # self.relation_linear = nn.Linear(in_features=entity_dim, out_features=entity_dim)
        # self.relation_linear_inv = nn.Linear(in_features=entity_dim, out_features=entity_dim)

    def init_reason(self, curr_dist, local_entity, kb_adj_mat, q_input, query_entities):
        """
        Initializing Reasoning
        """
        # kb_adj_mat = batch_heads, batch_rels, batch_tails, batch_ids, fact_ids, weight_list
        # batch_size = local_entity.size(0)
        self.local_entity = local_entity
        self.instruction_list, self.attn_list = self.instruction(q_input)
        self.constraint_list, self.cons_attn_list = self.constraint(q_input, self.instruction.node_encoder)
        rel_features, rel_features_inv  = self.get_rel_feature()
        if not self.rf_ief :
            # 获取实体
            self.local_entity_emb = self.get_ent_init(local_entity, kb_adj_mat, rel_features)
            self.init_entity_emb = self.local_entity_emb
        else :
            self.local_entity_emb = None
        self.curr_dist = curr_dist
        self.dist_history = []
        self.score_history = []
        self.action_probs = []
        self.seed_entities = curr_dist

        self.reasoning.init_reason(local_entity=local_entity,
                                   kb_adj_mat=kb_adj_mat,
                                   local_entity_emb=self.local_entity_emb,
                                   rel_features=rel_features,
                                   rel_features_inv=rel_features_inv,
                                   query_entities=query_entities,
                                   init_dist=curr_dist)


    def calc_loss_label(self, curr_dist, teacher_dist, label_valid):
        tp_loss = self.get_loss(pred_dist=curr_dist, answer_dist=teacher_dist, reduction='none')
        tp_loss = tp_loss * label_valid
        cur_loss = torch.sum(tp_loss) / curr_dist.size(0)
        return cur_loss
    # warmup时训练ief(整个模型开始训练之前训练ief)
    def train_ief(self, batch):
        # local_entity, query_entities, kb_adj_mat, query_text, seed_dist, answer_dist = batch
        local_entity, query_entities, kb_adj_mat, query_text, seed_dist, true_batch_id, answer_dist = batch
        local_entity = torch.from_numpy(local_entity).type('torch.LongTensor').to(self.device)
        # local_entity_mask = (local_entity != self.num_entity).float()
        query_entities = torch.from_numpy(query_entities).type('torch.FloatTensor').to(self.device)
        answer_dist = torch.from_numpy(answer_dist).type('torch.FloatTensor').to(self.device)
        seed_dist = torch.from_numpy(seed_dist).type('torch.FloatTensor').to(self.device)
        current_dist = Variable(seed_dist, requires_grad=True)

        q_input= torch.from_numpy(query_text).type('torch.LongTensor').to(self.device)

        self.init_reason(curr_dist=current_dist, local_entity=local_entity,
                         kb_adj_mat=kb_adj_mat, q_input=q_input, query_entities=query_entities)

        # 根据答案分布的形状计算批次大小bsize、每个批次的节点数bnode_num和节点总数node_num，以及关系类型的数量rtype_num。
        # 8, 2000
        bsize, bnode_num = answer_dist.shape
        node_num = bsize * bnode_num
        # 关系数量
        rtype_num = self.reasoning.rel_features.shape[0]

        h2r_idx = self.reasoning.head2fact_mat.coalesce().indices()
        h2r_idx[0] = self.reasoning.batch_rels
        # 构建头实体到关系（Head-to-Relation）的稀疏矩阵：使用head2fact_mat的索引，并结合关系ID，构建表示每个关系类型与所有节点连接情况的稀疏矩阵h2r_RF，值全部设为1。
        h2r_RF = torch.sparse_coo_tensor(h2r_idx, torch.ones_like(h2r_idx[0]), (rtype_num*2, node_num))

        t2r_idx = self.reasoning.tail2fact_mat.coalesce().indices()
        t2r_idx[0] = self.reasoning.batch_rels + rtype_num
        # 构建尾实体到关系（Tail-to-Relation）的稀疏矩阵：类似地，构建表示尾实体连接情况的稀疏矩阵t2r_RF，并将关系ID偏移以区分头尾实体的关系。
        t2r_RF = torch.sparse_coo_tensor(t2r_idx, torch.ones_like(t2r_idx[0]), (rtype_num*2, node_num))
        # 合并关系频率：合并h2r_RF和t2r_RF得到总的RF，表示每个关系类型与所有节点的连接总和
        RF = (h2r_RF + t2r_RF).coalesce()
        # 构建关系到实体（Relation-to-Entity）的稀疏矩阵：利用RF的索引构建矩阵R2E，值也为1，表示每种关系与实体的连接。
        R2E = torch.sparse_coo_tensor(RF.indices(), torch.ones_like(RF.values()), RF.shape).coalesce()

        """
        应该要改这里
        """
        # 计算实体频率（EF）：通过计算R2E中每行的总和，得到每个实体的连接数量，即实体频率
        EF = torch.sparse.sum(R2E, 1)
        # 累积统计量：将当前批次的实体频率EF累加到模型的累积实体频率self.EF中，同时累加local_entity_mask的和与批次大小到累积实体总数self.V。
        # todense是转为密集张量
        self.EF += EF.to_dense()
        # 结点数
        self.V += self.reasoning.local_entity_mask.sum() + bsize
        # 计算逆实体频率（IEF）：根据累积实体总数self.V和累积实体频率self.EF计算IEF，通过取对数并进行裁剪操作确保非负值。
        # clip(x)确保值不低于x
        """
        加和所有关系频率
        除
        当前问题关系的频率
        """
        IEF = torch.log(self.V / self.EF.clip(1))
        # 更新IEF状态：将计算出的IEF值更新到模型的self.IEF中，同样进行裁剪以排除负值
        # 取最后一次结果(EF与V都是)
        self.IEF.data = IEF.clip(0)

    def train_iwf(self, batch):
        """统计所有关系为r的实体频率"""
        local_entity, query_entities, kb_adj_mat, query_text, seed_dist, true_batch_id, answer_dist = batch
        local_entity = torch.from_numpy(local_entity).type('torch.LongTensor').to(self.device)
        # local_entity_mask = (local_entity != self.num_entity).float()
        query_entities = torch.from_numpy(query_entities).type('torch.FloatTensor').to(self.device)
        answer_dist = torch.from_numpy(answer_dist).type('torch.FloatTensor').to(self.device)
        seed_dist = torch.from_numpy(seed_dist).type('torch.FloatTensor').to(self.device)
        current_dist = Variable(seed_dist, requires_grad=True)

        q_input= torch.from_numpy(query_text).type('torch.LongTensor').to(self.device)

        self.init_reason(curr_dist=current_dist, local_entity=local_entity,
                         kb_adj_mat=kb_adj_mat, q_input=q_input, query_entities=query_entities)

        # 根据答案分布的形状计算批次大小bsize、每个批次的节点数bnode_num和节点总数node_num，以及关系类型的数量rtype_num。
        # 8, 2000
        bsize, bnode_num = answer_dist.shape
        node_num = bsize * bnode_num
        # 关系数量
        rtype_num = self.reasoning.rel_features.shape[0]

        h2r_idx = self.reasoning.head2fact_mat.coalesce().indices()
        h2r_idx[0] = self.reasoning.batch_rels
        # 构建头实体到关系（Head-to-Relation）的稀疏矩阵：使用head2fact_mat的索引，并结合关系ID，构建表示每个关系类型与所有节点连接情况的稀疏矩阵h2r_RF，值全部设为1。
        h2r_RF = torch.sparse_coo_tensor(h2r_idx, torch.ones_like(h2r_idx[0]), (rtype_num*2, node_num))

        t2r_idx = self.reasoning.tail2fact_mat.coalesce().indices()
        t2r_idx[0] = self.reasoning.batch_rels + rtype_num
        # 构建尾实体到关系（Tail-to-Relation）的稀疏矩阵：类似地，构建表示尾实体连接情况的稀疏矩阵t2r_RF，并将关系ID偏移以区分头尾实体的关系。
        t2r_RF = torch.sparse_coo_tensor(t2r_idx, torch.ones_like(t2r_idx[0]), (rtype_num*2, node_num))
        # 合并关系频率：合并h2r_RF和t2r_RF得到总的RF，表示每个关系类型与所有节点的连接总和
        RF = (h2r_RF + t2r_RF).coalesce()
        # 构建关系到实体（Relation-to-Entity）的稀疏矩阵：利用RF的索引构建矩阵R2E，值也为1，表示每种关系与实体的连接。
        R2E = torch.sparse_coo_tensor(RF.indices(), torch.ones_like(RF.values()), RF.shape).coalesce()

        # 计算实体频率（EF）：通过计算R2E中每行的总和，得到每个实体的连接数量，即实体频率
        EF = torch.sparse.sum(R2E, 1)
        # todense是转为密集张量
        self.EF += EF.to_dense()

    def forward(self, batch, training=False):
        """
        Forward function: creates instructions and performs GNN reasoning.
        """
        # local_entity, query_entities, kb_adj_mat, query_text, seed_dist, answer_dist = batch
        local_entity, query_entities, kb_adj_mat, query_text, seed_dist, true_batch_id, answer_dist = batch
        local_entity = torch.from_numpy(local_entity).type('torch.LongTensor').to(self.device)
        # local_entity_mask = (local_entity != self.num_entity).float()
        query_entities = torch.from_numpy(query_entities).type('torch.FloatTensor').to(self.device)
        answer_dist = torch.from_numpy(answer_dist).type('torch.FloatTensor').to(self.device)
        seed_dist = torch.from_numpy(seed_dist).type('torch.FloatTensor').to(self.device)
        # 将 current_dist 设置为 seed_dist 的可训练变量版本，准备进行梯度反向传播

        current_dist = Variable(seed_dist, requires_grad=True)

        q_input= torch.from_numpy(query_text).type('torch.LongTensor').to(self.device)

        #query_text2 = torch.from_numpy(query_text2).type('torch.LongTensor').to(self.device)
        if self.lm != 'lstm':
            pad_val = self.instruction.pad_val #tokenizer.convert_tokens_to_ids(self.instruction.tokenizer.pad_token)
            query_mask = (q_input != pad_val).float()

        else:
            query_mask = (q_input != self.num_word).float()

        """
        Expansion and Backup Instruction Generation
        """
        # 调用 init_reason 方法初始化推理过程，设置当前分布、处理指令文本、获取关系特征，并调用推理层的相应方法来初始化内部状态。
        self.init_reason(curr_dist=current_dist, local_entity=local_entity,
                         kb_adj_mat=kb_adj_mat, q_input=q_input, query_entities=query_entities)
        self.instruction.init_reason(q_input)
        # 分别使用 instruction 和 constraint 模块（基于 LSTM 或 BERT）为扩展和回溯步骤生成多个指令。
        for i in range(self.num_expansion_ins):
            relational_ins, attn_weight = self.instruction.get_instruction(self.instruction.relational_ins, step=i)
            self.instruction.instructions.append(relational_ins.unsqueeze(1))
            self.instruction.relational_ins = relational_ins
        self.constraint.init_reason(q_input)
        for i in range(self.num_backup_ins):
            relational_ins, attn_weight = self.constraint.get_instruction(self.constraint.relational_ins, step=i)
            self.constraint.instructions.append(relational_ins.unsqueeze(1))
            self.constraint.relational_ins = relational_ins
        #relation_ins = torch.cat(self.instruction.instructions, dim=1)
        #query_emb = None
        self.dist_history.append(self.curr_dist)

        """
        RF-IEF : RF(node x relations) * log(node num / (1 + EF))
        计算实体与关系的关联频率（RF），即每个实体参与的每种关系的数量。
        """
        if self.rf_ief :
            # Relation Frequency
            bsize, bnode_num = answer_dist.shape
            node_num = bsize * bnode_num
            rtype_num = self.reasoning.rel_features.shape[0]
            # 通过.coalesce()方法去除重复项并合并值。
            # 随后构造两个稀疏矩阵h2r_RF和t2r_RF，分别表示头实体和尾实体到所有节点的关系连接情况
            # 矩阵大小为(rtype_num*2, node_num)，因为每个关系类型在矩阵中都有正向和反向两种表示。
            h2r_idx = self.reasoning.head2fact_mat.coalesce().indices()
            h2r_idx[0] = self.reasoning.batch_rels
            h2r_RF = torch.sparse_coo_tensor(h2r_idx, torch.ones_like(h2r_idx[0]), (rtype_num*2, node_num))

            t2r_idx = self.reasoning.tail2fact_mat.coalesce().indices()
            t2r_idx[0] = self.reasoning.batch_rels + rtype_num
            t2r_RF = torch.sparse_coo_tensor(t2r_idx, torch.ones_like(t2r_idx[0]), (rtype_num*2, node_num))
            # 头实体与尾实体的关系数量
            RF = (h2r_RF + t2r_RF).coalesce()

            # pre computed IEF
            # 使用预计算的 IEF（实体频率逆指数）与 RF 相乘，得到 RF-IEF 特征。

            IEF = self.IEF.data
            """
            1.torch.arange(IEF.shape[0], device=IEF.device): 这段代码生成一个一维张量，其范围从0到IEF的长度减1。这实际上为每个实体生成了一个连续的索引。.device确保生成的张量位于与IEF相同的设备上（CPU或GPU）。
            2..tile(2,1): 这个操作将上述生成的索引张量沿第一个维度重复两次，沿第二个维度不重复。例如，如果原始索引为[0, 1, 2]，则重复后变为[[0, 1, 2], [0, 1, 2]]。这个操作是为了构造稀疏张量的坐标，使得每个实体在稀疏张量中有两份“副本”——一份对应于其作为关系的头实体，另一份对应于其作为关系的尾实体。
            3.torch.sparse_coo_tensor: 这是创建稀疏张量的关键函数。它接受三个主要参数：坐标（coordinates）、值（values）和形状（size）。在这个例子中，坐标由重复后的索引提供，值直接取自IEF向量，意味着每个实体的逆频率值同时用于表示它作为头实体和尾实体的情况。由于IEF的长度已经暗示了实体的数量，且每个实体在新张量中出现两次，因此形状参数可以通过计算得出，但在此代码片段中未显式指定，通常框架会根据提供的坐标和值自动推断形状。
            """

            # 从模型中获取预计算的IEF，并转换为稀疏COO格式IEF_coo，其中每个实体的IEF值在矩阵中重复两次，对应于它作为关系的头和尾。
            IEF_coo = torch.sparse_coo_tensor(torch.arange(IEF.shape[0], device=IEF.device).tile(2,1), IEF)
            # 通过稀疏矩阵乘法(torch.sparse.mm)将IEF_coo与RF相乘，得到RFIEF矩阵
            # 论文里的公式

            RFIEF = torch.sparse.mm(IEF_coo, RF.float()).coalesce()
            # if not self.rf_ief_normalize :
            #     RFIEF = torch.sparse.mm(IEF_coo, RF.float()).coalesce()
            # else :
            #     rf_idx, rf_val = RF.indices(), RF.values()
            #     for bidx in range(bsize):
            #         idx = (rf_idx[1]>bidx*bnode_num)*(rf_idx[1]<(bidx+1)*bnode_num)
            #         smpl_idx = rf_idx[:, idx]
            #         smpl_rf_val = rf_val[idx]

            #         # RF
            #         smpl_RF = torch.sparse_coo_tensor(smpl_idx, smpl_rf_val, RF.shape).coalesce()
            #         smpl_RF = smpl_RF / smpl_RF.values().sum()

            #         smpl_RFIEF = torch.sparse.mm(IEF_coo, smpl_RF)

            #         if bidx == 0 :
            #             RFIEF = smpl_RFIEF
            #         else :
            #             RFIEF += smpl_RFIEF

            #     RFIEF = RFIEF.coalesce()


            """
            RF-IEF node init
            使用 RF-IEF 特征初始化实体嵌入 local_entity_emb 及相关变量。
            """
            # 将原始的关系特征与它们的逆序版本拼接起来，然后通过一个线性层rfief_linear进一步变换这些特征，可能旨在引入额外的表达能力或调整关系特征。
            rel_features = torch.cat([self.reasoning.rel_features, self.reasoning.rel_features_inv])
            rel_features = self.rfief_linear(rel_features)
            # 利用变换后的RFIEF矩阵（通过转置进行适当的维度调整）与调整后的关系特征进行稀疏矩阵乘法，得到每个节点的初始嵌入表示node_init。
            node_init = torch.sparse.mm(RFIEF.transpose(1,0), rel_features.to_sparse()).to_dense()
            # 将node_init调整形状以匹配local_entity的形状，以便适配模型中的实体嵌入。
            node_init = node_init.reshape(local_entity.shape[0], local_entity.shape[1], -1)
            # 如何设置了RF-IEF则进行训练并赋值
            # 8,2000,100
            self.local_entity_emb = node_init
            self.init_entity_emb = node_init
            self.reasoning.local_entity_emb = node_init

        """
        注意力层
        """

        """
        NuTrea reasoning
        """
        # 在指定的迭代次数（num_iter）内，对于每一层（num_layers）：
        for t in range(self.num_iter):
            # 将所有扩展和备份指令连接成一个张量。
            relation_ins = torch.cat(self.instruction.instructions, dim=1)
            relation_con = torch.cat(self.constraint.instructions, dim=1)
            self.curr_dist = current_dist
            for j in range(self.num_layers):
                # 通过推理层（reasoning）进行一次推理，更新当前分布 curr_dist 并获取原始得分 raw_score。
                # raw_score, self.curr_dist, global_rep = self.reasoning(self.curr_dist, relation_ins, relation_con, step=j)
                raw_score, self.curr_dist, global_rep, gs, hs = self.reasoning(self.curr_dist, relation_ins, relation_con, step=j, return_score=True)
            # 将当前分布添加到历史记录列表中。
            self.dist_history.append(self.curr_dist)
            self.score_history.append(raw_score)

            """
            Expansion Instruction Updates
            每一轮迭代后，根据生成的扩展指令更新 instruction 中的扩展指令历史。
            """
            for j in range(self.num_expansion_ins):
                exp_reform = getattr(self, 'reform' + str(j))
                q = exp_reform(self.instruction.instructions[j].squeeze(1), global_rep, query_entities, local_entity)
                self.instruction.instructions[j] = q.unsqueeze(1)

            """
            Backup Instruction Updates
            同样地，根据生成的备份指令更新 constraint 中的备份指令历史。
            """
            for j in range(self.num_backup_ins):
                bak_reform = getattr(self, 'conreform' + str(j))
                q = bak_reform(self.constraint.instructions[j].squeeze(1), global_rep, query_entities, local_entity)
                self.constraint.instructions[j] = q.unsqueeze(1)


        """
        Answer Predictions
        
        """
        if self.post_norm :
            # 如果启用 post_norm，则将所有历史得分求和，并添加负无穷惩罚项（针对非实体位置），然后进行softmax运算得到预测分布。
            pred_logit = sum(self.score_history) + (1-self.reasoning.local_entity_mask) * VERY_NEG_NUMBER
            pred_dist = self.softmax_d1(pred_logit)
        else :
            # 否则，直接使用最后一次迭代的当前分布作为预测分布。
            pred_dist = self.dist_history[-1]
        # 计算每个样本的有效答案数量（answer_number），并据此过滤掉无答案的训练案例。
        answer_number = torch.sum(answer_dist, dim=1, keepdim=True)
        # 计算与教师分布（answer_dist）之间的损失（loss），使用 calc_loss_label 方法。
        case_valid = (answer_number > 0).float()
        # filter no answer training case
        # loss = 0
        # for pred_dist in self.dist_history:
        loss = self.calc_loss_label(curr_dist=pred_dist, teacher_dist=answer_dist, label_valid=case_valid)

        if self.post_norm :
            pass
        else :
            pred_dist = self.dist_history[-1]
        pred = torch.max(pred_dist, dim=1)[1]

        if training:
            h1, f1 = self.get_eval_metric(pred_dist, answer_dist)
            tp_list = [h1.tolist(), f1.tolist()]
        else:
            tp_list = None
        return loss, pred, pred_dist, tp_list

