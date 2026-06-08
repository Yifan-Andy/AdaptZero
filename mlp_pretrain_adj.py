# Prre-train with ADJ only

import time
import utils
import random
import numpy as np
import torch
import os.path
import torch.utils.data as Data
from model import MLPModel
from lr import PolynomialDecayLR
from data_loader import get_dataset
from early_stop import EarlyStopping, Stop_args
from utils import parse_args, transform_coo_to_csr, transform_sp_csr_to_coo, deepwalk_from_edges

if __name__ == "__main__":
    args = parse_args()
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    adj, features = get_dataset(args.dataset, args.pe_dim)

    # process feature
    start_feature_processing = time.time()
    feature_path = f"checkpoints/processed_features/{args.dataset}_adj.pt"
    if not os.path.exists(feature_path):
        # get deepwalk feature from edges
        adj_c = adj.coalesce()
        row, col = adj_c.indices()
        edges = list(zip(row.tolist(), col.tolist()))
        if args.dataset in ["reddit", "products"]:
            features = torch.load(f"./dataset/{args.dataset}/deepwalk_{args.dataset}.pt")
        else:
            features = deepwalk_from_edges(edges, num_nodes=adj.size(0))

        # process feature with hops
        processed_features = utils.re_features(adj, features, args.hops)  # return (N, hops+1, d)
        torch.save(processed_features, feature_path)
        print("feature saved")
    else:
        processed_features = torch.load(feature_path)
        print("feature process from saved checkpoint")

    t_feature_precessing = time.time() - start_feature_processing
    print("feature process time: {:.4f}s".format(t_feature_precessing))

    start = time.time()
    print("starting transformer to coo")
    adj = transform_coo_to_csr(adj) # transform to csr to support slicing operation
    print("start mini batch processing")
    adj_batch, minus_adj_batch = transform_sp_csr_to_coo(adj, args.batch_size, features.shape[0]) # transform to coo to support tensor operation
    print(len(adj_batch[0]), len(minus_adj_batch[0]))
    print("adj process time: {:.4f}s".format(time.time() - start))
    

    data_loader = Data.DataLoader(processed_features, batch_size=args.batch_size, shuffle = False)

    # model configuration
    model = MLPModel(input_dim=processed_features.shape[2], config=args).to(args.device)

    print(model)
    print('total params:', sum(p.numel() for p in model.parameters()))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.peak_lr, weight_decay=args.weight_decay)
    lr_scheduler = PolynomialDecayLR(
                    optimizer,
                    warmup_updates=args.warmup_updates,
                    tot_updates=args.tot_updates,
                    lr=args.peak_lr,
                    end_lr=args.end_lr,
                    power=1.0)
    
    stopping_args = Stop_args(patience=args.patience, max_epochs=args.epochs)
    early_stopping = EarlyStopping(model, **stopping_args)

    print("starting training...")
    # model train
    model.train()

    loss_train_b = []
    t_start = time.time()
    for epoch in range(args.epochs):
        for index, item in enumerate(data_loader):

            start_index = index*args.batch_size
            nodes_features = item.to(args.device)
            adj_ = adj_batch[index].to(args.device)
            minus_adj = minus_adj_batch[index].to(args.device)
            # print(nodes_features.shape)
            optimizer.zero_grad()
            node_tensor, neighbor_tensor = model(nodes_features)

            # print(node_tensor.shape, neighbor_tensor.shape, adj_.shape, minus_adj.shape)
            loss_train = model.contrastive_link_loss(node_tensor, neighbor_tensor, adj_, minus_adj)
            loss_train.backward()
            optimizer.step()
            lr_scheduler.step()
            loss_train_b.append(loss_train.item())
            # break

        if early_stopping.simple_check(loss_train_b):
            break

        print('Epoch: {:04d}'.format(epoch+1), 'loss_train: {:.4f}'.format(loss_train.item()))
    t_train = time.time() - t_start
    
    print("Optimization Finished!")
    print(f"Train time: {t_train}s")

    # model save
    print("Start Save Model...")

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    if not os.path.exists(args.embedding_path):
        os.makedirs(args.embedding_path)

    # torch.save(model.state_dict(), args.save_path + args.model_name + '.pth')
    
    # obtain all the node embedding from the learned model
    model.eval()
    node_embedding = []
    for _, item in enumerate(data_loader):
        nodes_features = item.to(args.device)
        node_tensor, neighbor_tensor = model(nodes_features)
        if len(node_embedding) == 0:
            node_embedding = np.concatenate((node_tensor.cpu().detach().numpy(), neighbor_tensor.cpu().detach().numpy()), axis=1)
            # node_embedding = node_tensor.cpu().detach().numpy()
        else:
            new_node_embedding = np.concatenate((node_tensor.cpu().detach().numpy(), neighbor_tensor.cpu().detach().numpy()), axis=1)
            # new_node_embedding = node_tensor.cpu().detach().numpy()
            node_embedding = np.concatenate((node_embedding, new_node_embedding), axis=0)
    
    np.save(args.embedding_path + args.dataset + '_mlp_adj.npy', node_embedding)

    # file_path = f"logs_final/{args.dataset}.log"
    # print(file_path)
    # if not os.path.exists(file_path):
    #     with open(file_path, 'w') as file:
    #         print("Create Logs")
    
    # with open(file_path, 'a') as file:
    #     file.write(f'ADJ Args: {args}\n')
    #     file.write('----------------------------------------------\n')
    #     file.write(f'Train Time: {t_train} s\n')
    #     file.write('==============================================\n\n')
    #     print("Final Logs Write Successful")
