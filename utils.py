import numpy as np
from scipy.io.wavfile import read
import torch
from hparams import hparams
import dgl


def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len))
    mask = (ids < lengths.unsqueeze(1)).bool()
    return mask


def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text


def to_gpu(x):
    x = x.contiguous()

    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return torch.autograd.Variable(x)

def to_gpu_graph(g):
    edge_tensor1 = to_gpu(g.edges()[0]).long()
    edge_tensor2 = to_gpu(g.edges()[1]).long()
    nodes_num = len(g.nodes())

    if hparams.dep_graph_type == "uni_type":
        # 单向有类型边
        new_g = dgl.graph((edge_tensor1, edge_tensor2), num_nodes=nodes_num)
        new_g.edata['type'] = to_gpu(g.edata['type']).long()

    if hparams.dep_graph_type == "uni_nonetype":
        # 单向无类型边
        new_g = dgl.graph((edge_tensor1, edge_tensor2), num_nodes=nodes_num)
        new_g.edata['type'] = to_gpu(torch.zeros_like(g.edata['type'])).long()

    if hparams.dep_graph_type == "rev_type":
        # 反向有类型边
        new_g = dgl.graph((edge_tensor2, edge_tensor1), num_nodes=nodes_num)
        new_g.edata['type'] = to_gpu(g.edata['type']).long()

    if hparams.dep_graph_type == "rev_nonetype":
        # 反向无类型边
        new_g = dgl.graph((edge_tensor2, edge_tensor1), num_nodes=nodes_num)
        new_g.edata['type'] = to_gpu(torch.zeros_like(g.edata['type'])).long()

    if hparams.dep_graph_type == "bi_type":
        # 双向有类型边
        new_g = dgl.graph((edge_tensor1, edge_tensor2), num_nodes=nodes_num)
        new_g.edata['type'] = to_gpu(g.edata['type']).long()

    if hparams.dep_graph_type == "bi_nonetype":
        # 双向无类型边
        new_g = dgl.graph((edge_tensor1, edge_tensor2), num_nodes=nodes_num)
        new_g.edata['type'] = to_gpu(torch.zeros_like(g.edata['type'])).long()


    return new_g
