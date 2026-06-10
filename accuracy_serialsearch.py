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
    parser.add_argument('--max_rounds', type=int, default=10, help='')

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

def serial_search(
    full_emb,
    x_emb,
    adj_emb,
    reverse_emb,
    args
):
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

    target_k = int(args.topk)

    # ===== hyper-params =====
    max_rounds = args.max_rounds

    total_overlap = 0.0

    def _topk_indices(x, k):
        k = max(1, min(int(k), x.numel()))
        return torch.topk(x, k).indices

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
        # Step 2: initial pool from reverse score
        # ==========================================================
        full_global = _safe_zscore(full_score[i])
        x_global = _safe_zscore(x_score[i])
        adj_global = _safe_zscore(adj_score[i])
        rev_global = _safe_zscore(reverse_score[i])

        agent_scores_global = {
            "full": full_global,
            "x": x_global,
            "adj": adj_global,
        }

        agent_overlap = {
            "full": overlap_full,
            "x": overlap_x,
            "adj": overlap_adj,
        }

        agent_used_count = {
            "full": 0,
            "x": 0,
            "adj": 0,
        }

        # current_score_global = rev_global.clone()
        current_score_global = (full_global + x_global + adj_global) / 3.0
        pool = _topk_indices(current_score_global, adaptive_candidate_k)

        chosen_path = ["fusion(the stable init)"]
        old_pool = None

        # ==========================================================
        # Step 3: first choose agent, then update only once
        # ==========================================================
        for _ in range(max_rounds - 1):
            # ---------- A. lightweight agent choose with reverse ----------
            best_agent = None
            best_select = None
            old_pool = pool.clone()

            pool_cur = current_score_global[pool]
            pool_rev = rev_global[pool]
            for agent_name, aux_global in agent_scores_global.items():
                pool_aux = aux_global[pool]
                cur_bias = torch.abs(pool_aux - pool_cur).mean()
                # rev_bias = torch.abs(pool_aux - pool_rev).mean()
                rev_bias = agent_overlap[agent_name]

                # 所有round共享的衰减权重：出现越多，权重越低
                usage_weight = 1.0 / (1.0 + agent_used_count[agent_name])

                select_score = cur_bias * rev_bias * usage_weight

                if best_select is None or select_score > best_select:
                    best_select = select_score
                    best_agent = agent_name

            # ---------- B. only update pool and scores with best_agent ----------
            usage_weight = 1.0 / (1.0 + agent_used_count[best_agent])
            redundancy = max(1, adaptive_candidate_k - target_k)
            buffer_swap_k = int(np.ceil(np.sqrt(redundancy)))
            # size for prune and expand
            swap_k = max(1, int(buffer_swap_k * usage_weight))

            aux_global = agent_scores_global[best_agent]

            pool_aux = aux_global[pool]
            pool_rev = rev_global[pool]
            pool_cur = current_score_global[pool]

            # 1) reweight
            step = torch.exp(-torch.abs(pool_aux - pool_rev))
            step = torch.clamp(step, min=0.05, max=0.95)

            current_score_global[pool] = pool_cur + step * (pool_aux - pool_cur)

            # 2) prune: 所有 pool 内节点都可删
            inside_aux = current_score_global[pool]
            inside_rev = rev_global[pool]
            inside_agree = torch.exp(-torch.abs(inside_aux - inside_rev))
            inside_scores = inside_aux * inside_agree

            drop_local = torch.topk(-inside_scores, swap_k).indices
            keep_mask = torch.ones(pool.numel(), dtype=torch.bool, device=device)
            keep_mask[drop_local] = False
            pruned_pool = pool[keep_mask]

            # 3) expand: 所有 pool 外节点都可扩
            outside_mask = torch.ones(num_nodes, dtype=torch.bool, device=device)
            outside_mask[pruned_pool] = False
            outside_idx = torch.where(outside_mask)[0]

            outside_aux = aux_global[outside_idx]
            outside_rev = rev_global[outside_idx]
            outside_agree = torch.exp(-torch.abs(outside_aux - outside_rev))
            outside_scores = outside_aux * outside_agree

            add_local = torch.topk(outside_scores, swap_k).indices
            add_nodes = outside_idx[add_local]

            pool = torch.cat([pruned_pool, add_nodes], dim=0)

            chosen_path.append(
                f"{best_agent}(pool = {pool.shape[0]}, swap = {swap_k}, update = {(~torch.isin(pool, old_pool)).sum().item()})"
            )

            agent_used_count[best_agent] += 1

        # ==========================================================
        # Step 4: final top-k from current dynamic pool
        # ==========================================================
        final_pool_score = current_score_global[pool]
        final_k = min(target_k, pool.numel())
        final_nodes_local = _topk_indices(final_pool_score, final_k)
        final_nodes = pool[final_nodes_local]

        final_nodes = torch.unique(final_nodes, sorted=False)
        global_fill_mask = torch.ones(num_nodes, dtype=torch.bool, device=device)
        global_fill_mask[final_nodes] = False

        preds[i, final_nodes] = 1.0
        labels_part[i, final_nodes] = labels[i, final_nodes].to(device)

        print(
            f"query={i}, "
            f"path={' -> '.join(chosen_path)}\n"
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
    #     file.write(f'Serial Args: {args}\n')
    #     file.write('----------------------------------------------\n')
    #     file.write(f'[Top-{target_k}] F1: {f1_total:.4f} | NMI: {nmi_total:.4f} | JAC: {jac_total:.4f}\n')
    #     file.write(f'Serial Search Time: {t_search:.4f} s\n')
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

    serial_search(embedding_full, embedding_x, embedding_adj, embedding_reverse, args)
