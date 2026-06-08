import torch

def get_dataset(dataset, pe_dim):
    if dataset in {"pubmed", "photo", "computer", "cocs", "cora", "physics", "citeseer"}:
        if dataset in {"photo", "computer", "cocs"}:
            file_path = "dataset/"+dataset+"_dgl.pt"
        else:
            file_path = "dataset/"+dataset+"_pyg.pt"

        data_list = torch.load(file_path)
        adj = data_list[0]
        features = data_list[1]
    else:
        file_path = "dataset/" + dataset + "_pyg.pt"

        data_list = torch.load(file_path)
        adj = data_list[0]
        features = data_list[1]

    print(f"Dataset: {dataset}, #Nodes: {adj.shape[0]}, #Edges: {adj.sum().item()}, #Features: {features.shape[1]}")
    
    return adj.cpu().type(torch.LongTensor), features
