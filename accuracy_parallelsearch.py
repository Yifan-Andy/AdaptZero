import torch
import os
import argparse
import numpy as np
import time
import torch.nn.functional as F
import math
import ast
from utils import load_query_n_gt
from sklearn.metrics import f1_score, normalized_mutual_info_score, jaccard_score

LOG_PATH = "logs/statistics.log"
EPSILON = 1e-8

def parse_args():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser()

    # main parameters
    parser.add_argument('--dataset', type=str, default='cora', help='dataset name')
    parser.add_argument('--embedding_tensor_name', type=str, help='embedding tensor name')
    parser.add_argument('--embedding_path', type=str, default='./pretrain_result/', help='embedding path')

    # search parameters
    parser.add_argument('--residual_alpha', type=float, default=0.5, help='')

    return parser.parse_args()

def build_query_feat(query, emb):
    query_sum = torch.mm(query, emb)
    query_count = torch.sum(query, dim=1, keepdim=True) + EPSILON
    query_feat = query_sum / query_count
    query_feat = F.normalize(query_feat, p=2, dim=1)
    return query_feat

def score_to_prob(score, tau=0.1):
    prob = torch.softmax(score / tau, dim=0)
    return prob

def _safe_zscore(x):
    std = x.std()
    if std < EPSILON:
        return x - x.mean()
    return (x - x.mean()) / (std + EPSILON)

def softmax(x):
    exp_x = [math.exp(i) for i in x]
    sum_exp = sum(exp_x)
    return [i / sum_exp for i in exp_x]

def parallel_search(
    full_emb,
    x_emb,
    adj_emb,
    reverse_emb,
    args
):
    """
    Parallel reverse-aware approximation of serial_search.

    Key idea:
        - remove iterative agent selection and dynamic pool refinement
        - compute all agent scores in parallel
        - build query-level routing weights once
        - build node-level reverse-aware gates once
        - fuse all agents in one shot
        - take adaptive candidate pool and final top-k from fused score

    Inputs / outputs are kept aligned with serial_search as much as possible.
    """

    query, labels = load_query_n_gt("./dataset/", args.dataset, full_emb.shape[0])

    t_start = time.time()

    # ===== normalize =====
    full_emb = F.normalize(full_emb.float(), dim=1)
    x_emb = F.normalize(x_emb.float(), dim=1)
    adj_emb = F.normalize(adj_emb.float(), dim=1)
    reverse_emb = F.normalize(reverse_emb.float(), dim=1)

    query = query.float()
    labels = labels.float()

    num_nodes = full_emb.shape[0]
    num_queries = query.shape[0]
    device = full_emb.device
    target_k = int(args.topk)

    # ===== query features =====
    full_query_feat = build_query_feat(query, full_emb)
    x_query_feat = build_query_feat(query, x_emb)
    adj_query_feat = build_query_feat(query, adj_emb)
    reverse_query_feat = build_query_feat(query, reverse_emb)

    # ===== raw scores =====
    full_score = torch.mm(full_query_feat, full_emb.T)
    x_score = torch.mm(x_query_feat, x_emb.T)
    adj_score = torch.mm(adj_query_feat, adj_emb.T)
    reverse_score = torch.mm(reverse_query_feat, reverse_emb.T)

    preds = torch.zeros((num_queries, num_nodes), device=device)
    labels_part = torch.zeros((num_queries, num_nodes), device=device)

    total_overlap = 0.0

    def _topk_indices(x, k):
        k = max(1, min(int(k), x.numel()))
        return torch.topk(x, k).indices

    def _safe_norm_weights(x):
        # x: [M]
        x = torch.clamp(x, min=0.0)
        s = x.sum()
        if s.item() < EPSILON:
            return torch.ones_like(x) / x.numel()
        return x / (s + EPSILON)

    for i in range(num_queries):
        # ==========================================================
        # Step 1: reverse-based query priors
        # ==========================================================
        full_probe = _topk_indices(full_score[i], target_k)
        topx_probe = _topk_indices(x_score[i], target_k)
        topadj_probe = _topk_indices(adj_score[i], target_k)
        toprev_probe = _topk_indices(reverse_score[i], target_k)

        overlap_full = torch.isin(full_probe, toprev_probe).float().mean()
        overlap_x = torch.isin(topx_probe, toprev_probe).float().mean()
        overlap_adj = torch.isin(topadj_probe, toprev_probe).float().mean()

        overlap_mean = (overlap_full + overlap_x + overlap_adj) / 3.0
        total_overlap += overlap_mean.item()

        adaptive_scale = 1.0 + torch.sqrt(
            torch.clamp(-torch.log(overlap_mean + EPSILON), min=0.0)
        )
        adaptive_candidate_k = int(target_k * adaptive_scale.item())
        adaptive_candidate_k = max(target_k, min(num_nodes, adaptive_candidate_k))

        # ==========================================================
        # Step 2: z-score normalization
        # ==========================================================
        full_global = _safe_zscore(full_score[i])     # [N]
        x_global = _safe_zscore(x_score[i])           # [N]
        adj_global = _safe_zscore(adj_score[i])       # [N]
        rev_global = _safe_zscore(reverse_score[i])   # [N]

        # three active agents
        agent_scores = torch.stack([full_global, x_global, adj_global], dim=0)  # [3, N]
        overlap_vec = torch.stack(
            [overlap_full, overlap_x, overlap_adj], dim=0
        ).to(device)  # [3]

        # ==========================================================
        # Step 3: initial score / initial candidate pool
        # ==========================================================
        init_score = agent_scores.mean(dim=0)  # [N]
        init_pool = _topk_indices(init_score, adaptive_candidate_k)

        # ==========================================================
        # Step 4: query-level routing weights (parallel soft routing)
        # ==========================================================
        # mimic serial select_score = cur_bias * rev_bias
        # but do it once for all agents without usage count / iterative path
        pool_cur = init_score[init_pool].unsqueeze(0)              # [1, Kc]
        pool_agents = agent_scores[:, init_pool]                   # [3, Kc]

        cur_bias = torch.abs(pool_agents - pool_cur).mean(dim=1)   # [3]
        route_score = cur_bias * overlap_vec                       # [3]

        # linear normalized weights, no extra hyper-parameter
        route_weight = _safe_norm_weights(route_score)             # [3]

        # ==========================================================
        # Step 5: node-level reverse-aware reliability gates
        # ==========================================================
        # same spirit as step = exp(-|pool_aux - pool_rev|)
        node_gate = torch.exp(-torch.abs(agent_scores - rev_global.unsqueeze(0)))  # [3, N]
        node_gate = torch.clamp(node_gate, min=0.05, max=0.95)

        # ==========================================================
        # Step 6: one-shot parallel fusion
        # ==========================================================
        # weighted average with query-level routing and node-level reverse gate
        fused_num = (route_weight.unsqueeze(1) * node_gate * agent_scores).sum(dim=0)  # [N]
        fused_den = (route_weight.unsqueeze(1) * node_gate).sum(dim=0) + EPSILON       # [N]
        fused_score = fused_num / fused_den                                            # [N]

        # optional stabilization with reverse/global init:
        # this keeps the fused result from deviating too far
        # no extra hyper-param: simple average
        final_score = (1 - args.residual_alpha) * fused_score + args.residual_alpha * init_score

        # ==========================================================
        # Step 7: final candidate pool + final top-k
        # ==========================================================
        final_k = min(target_k, final_score.numel())
        final_nodes = _topk_indices(final_score, final_k)
        final_nodes = torch.unique(final_nodes, sorted=False)

        preds[i, final_nodes] = 1.0
        labels_part[i, final_nodes] = labels[i, final_nodes].to(device)

        print(
            f"query={i}, "
            f"path=parallel_fusion("
            f"pool={adaptive_candidate_k}, "
            f"route=[full:{route_weight[0].item():.3f}, "
            f"x:{route_weight[1].item():.3f}, "
            f"adj:{route_weight[2].item():.3f}])"
        )

    t_search = time.time() - t_start

    # ===== evaluation =====
    flat_preds = preds.view(-1).detach().cpu().numpy()
    flat_labels = labels_part.view(-1).detach().cpu().numpy()

    f1_total = f1_score(flat_labels, flat_preds)
    nmi_total = normalized_mutual_info_score(flat_labels, flat_preds)
    jac_total = jaccard_score(flat_labels, flat_preds)
    avg_overlap = total_overlap / num_queries

    print(
        f"[Top-{target_k}] F1: {f1_total:.4f} | NMI: {nmi_total:.4f} | JAC: {jac_total:.4f} | "
        f"Avg. Overlap: {avg_overlap:.4f} | "
        f"Searching Time: {t_search:.4f}"
    )

    # file_path = f"logs_final/{args.dataset}.log"
    # print(file_path)
    # if not os.path.exists(file_path):
    #     with open(file_path, 'w') as file:
    #         print("Create Logs")
    
    # with open(file_path, 'a') as file:
    #     file.write(f'Parallel Args: {args}\n')
    #     file.write('----------------------------------------------\n')
    #     file.write(f'[Top-{target_k}] F1: {f1_total:.4f} | NMI: {nmi_total:.4f} | JAC: {jac_total:.4f}\n')
    #     file.write(f'Parallel Search Time: {t_search:.4f} s\n')
    #     file.write('==============================================\n\n')
    #     print("Final Logs Write Successful")

if __name__ == "__main__":
    args = parse_args()
    print(args)

    if args.embedding_tensor_name is None:
        args.embedding_tensor_name = args.dataset

    if args.dataset in ["cora", "citeseer", "pubmed", "dblp", "photo", "computer", "wikics", "roman", "chameleon", "actor", "cocs"]:
        args.topk = 150
    elif args.dataset in ["reddit", "products"]:
        args.topk = 1000
    elif args.dataset in ["texas", "cornell", "wisconsin"]:
        args.topk = 30

    embedding_full = torch.from_numpy(np.load(args.embedding_path + args.embedding_tensor_name + '_mlp_full.npy'))
    embedding_x = torch.from_numpy(np.load(args.embedding_path + args.embedding_tensor_name + '_mlp_x.npy'))
    embedding_adj = torch.from_numpy(np.load(args.embedding_path + args.embedding_tensor_name + '_mlp_adj.npy'))
    embedding_reverse = torch.from_numpy(np.load(args.embedding_path + args.embedding_tensor_name + '_mlp_reverse.npy'))

    parallel_search(embedding_full, embedding_x, embedding_adj, embedding_reverse, args)
