from clftools.web.webpage import Webpage
import networkx as nx
import dgl
import torch


def create_graph_for_webpage(webpage: Webpage):
    graph_as_nx = webpage.dom_as_graph
    return dgl.from_networkx(graph_as_nx)


def set_node_features(graph: dgl.DGLGraph, feat_name, feat_mat: torch.Tensor):
    num_nodes = graph.num_nodes()
    if not isinstance(feat_mat, torch.Tensor):
        feat_mat = torch.Tensor(feat_mat)
    if feat_mat.dim != 2:
        raise ValueError(feat_mat, 'Node feature matrix is not in 2-D.')
    if num_nodes != feat_mat.size(0):
        raise ValueError(feat_mat, "Node feature matrix doesn't fit the given graph")
    graph.ndata[feat_name] = feat_mat
    return graph


def set_edge_features(graph: dgl.DGLGraph, feat_name, feat_mat: torch.Tensor):
    num_edges = graph.num_edges()
    if not isinstance(feat_mat, torch.Tensor):
        feat_mat = torch.Tensor(feat_mat)
    if feat_mat.dim != 2:
        raise ValueError(feat_mat, 'Edge feature matrix is not in 2-D.')
    if num_edges != feat_mat.size(0):
        raise ValueError(feat_mat, "Edge feature matrix doesn't fit the given graph")
    graph.ndata[feat_name] = feat_mat
    return graph


if __name__ == '__main__':
    url = 'https://www.cs.tsinghua.edu.cn/info/1111/3486.htm'
    webpage = Webpage(url=url)
    print(create_graph_for_webpage(webpage))
