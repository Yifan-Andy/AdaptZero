# AdaptZero

Official implementation of **AdaptZero: Query-Adaptive Zero-Shot Community Search with Mixture of Agents**.

## Code Map

| Paper component | Agent view | Implementation | Saved embedding |
|---|---|---|---|
| Alignment Agent | `(A, X)` | `mlp_pretrain_full.py` | `<dataset>_mlp_full.npy` |
| Attribute Agent | `(A', X)` | `mlp_pretrain_x.py` | `<dataset>_mlp_x.npy` |
| Structure Agent | `(A, X')` | `mlp_pretrain_adj.py` | `<dataset>_mlp_adj.npy` |
| Stable Agent | `(A', X')` | `mlp_pretrain_stable.py` | `<dataset>_mlp_stable.npy` |
| Serial Search | Stable-guided | `accuracy_serialsearch.py` | terminal metrics |
| Parallel Search | Stable-guided | `accuracy_parallelsearch.py` | terminal metrics |

## Datasets

The experiments use 10 graphs.

| Dataset argument | Source | Serialized graph |
|---|---|---|
| `cornell` | PyG WebKB | `dataset/cornell_pyg.pt` |
| `texas` | PyG WebKB | `dataset/texas_pyg.pt` |
| `citeseer` | PyG Planetoid | `dataset/citeseer_pyg.pt` |
| `actor` | PyG Actor | `dataset/actor_pyg.pt` |
| `photo` | PyG Amazon Photo | `dataset/photo_dgl.pt` |
| `wikics` | PyG WikiCS | `dataset/wikics_pyg.pt` |
| `cocs` | Coauthor CS | `dataset/cocs_dgl.pt` |
| `pubmed` | PyG Planetoid | `dataset/pubmed_pyg.pt` |
| `reddit` | PyG Reddit | `dataset/reddit_pyg.pt` |
| `products` | OGBN-Products | `dataset/products_pyg.pt` |

## Ready-to-Run Data

Run data processing methods below. The resulting layout should be:

```text
dataset/
  citeseer_pyg.pt
  <dataset>_pyg.pt
  ...
  citeseer/
    citeseer.query
    citeseer.gt
  <dataset>/
    <dataset>.query
    <dataset>.gt
```

## Requirements
```
python==3.10.14
torch==2.2.0+cu121
torch_geometric==2.4.0
numpy==1.26.3
scikit-learn==1.3.0
scipy==1.11.2
```

## How to Run

You can run our code with:
```
# Download dataset for both stages
python3 dataset_dealings/*_download_pyg.py

# Generate test data
python3 dataset/*/*_data.py

# Offline agent pre-training and online agent collaboration
bash run.sh
```


## Repository Structure

```text
AdaptZero/
  dataset/                       processed graphs and query files
  dataset_dealings/              dataset conversion scripts
  mlp_pretrain_full.py           Alignment Agent
  mlp_pretrain_x.py              Attribute Agent
  mlp_pretrain_adj.py            Structure Agent
  mlp_pretrain_stable.py         Stable Agent
  accuracy_serialsearch.py       Serial Search
  accuracy_parallelsearch.py     Parallel Search
  model.py                       attentive community encoder and losses
  utils.py                       preprocessing, query loading, kNN, and DeepWalk
  run.sh                         commands for all evaluated datasets
```