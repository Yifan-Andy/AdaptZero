# AdaptZero
This is the official implementation of AdaptZero.

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
python3 dataset/*_download_pyg.py

# Generate test data
python3 dataset/*/*_data.py

# MoA Pre-training and Searching
bash run.sh
```