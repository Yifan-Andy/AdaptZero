# run experiments on different datasets with AdaptZero

python mlp_pretrain_full.py --dataset citeseer --batch_size 3327 --epochs 100 --dropout 0.1 --hidden_dim 512 --hops 5 --peak_lr 0.01 --weight_decay 1e-05 --device 1
python mlp_pretrain_x.py --dataset citeseer --batch_size 3327 --epochs 100 --dropout 0.1 --hidden_dim 512 --hops 5 --peak_lr 0.01 --weight_decay 1e-05 --device 1
python mlp_pretrain_adj.py --dataset citeseer --batch_size 3327 --epochs 100 --dropout 0.1 --hidden_dim 512 --hops 5 --peak_lr 0.01 --weight_decay 1e-05 --device 1
python mlp_pretrain_reverse.py --dataset citeseer --batch_size 3327 --epochs 100 --dropout 0.1 --hidden_dim 512 --hops 5 --peak_lr 0.01 --weight_decay 1e-05 --device 1
python3 accuracy_serialsearch.py --dataset citeseer
python3 accuracy_parallelsearch.py --dataset citeseer

python mlp_pretrain_full.py --dataset pubmed --batch_size 19717 --epochs 100 --dropout 0.1 --hidden_dim 512 --hops 5 --peak_lr 0.01 --weight_decay 1e-05 --device 1
python mlp_pretrain_x.py --dataset pubmed --batch_size 19717 --epochs 100 --dropout 0.1 --hidden_dim 512 --hops 5 --peak_lr 0.01 --weight_decay 1e-05 --device 1
python mlp_pretrain_adj.py --dataset pubmed --batch_size 19717 --epochs 100 --dropout 0.1 --hidden_dim 512 --hops 5 --peak_lr 0.01 --weight_decay 1e-05 --device 1
python mlp_pretrain_reverse.py --dataset pubmed --batch_size 19717 --epochs 100 --dropout 0.1 --hidden_dim 512 --hops 5 --peak_lr 0.01 --weight_decay 1e-05 --device 1
python3 accuracy_serialsearch.py --dataset pubmed
python3 accuracy_parallelsearch.py --dataset pubmed

python mlp_pretrain_full.py --dataset cocs --batch_size 18333 --epochs 100 --dropout 0.1 --hidden_dim 512 --hops 5 --peak_lr 0.01 --weight_decay 1e-05 --device 1
python mlp_pretrain_x.py --dataset cocs --batch_size 18333 --epochs 100 --dropout 0.1 --hidden_dim 512 --hops 5 --peak_lr 0.01 --weight_decay 1e-05 --device 1
python mlp_pretrain_adj.py --dataset cocs --batch_size 18333 --epochs 100 --dropout 0.1 --hidden_dim 512 --hops 5 --peak_lr 0.01 --weight_decay 1e-05 --device 1
python mlp_pretrain_reverse.py --dataset cocs --batch_size 18333 --epochs 100 --dropout 0.1 --hidden_dim 512 --hops 5 --peak_lr 0.01 --weight_decay 1e-05 --device 1
python3 accuracy_serialsearch.py --dataset cocs
python3 accuracy_parallelsearch.py --dataset cocs

python mlp_pretrain_full.py --dataset texas --batch_size 183 --epochs 100 --dropout 0.1 --hidden_dim 128 --hops 5 --peak_lr 0.01 --weight_decay 1e-05 --device 1
python mlp_pretrain_x.py --dataset texas --batch_size 183 --epochs 100 --dropout 0.1 --hidden_dim 128 --hops 5 --peak_lr 0.01 --weight_decay 1e-05 --device 1
python mlp_pretrain_adj.py --dataset texas --batch_size 183 --epochs 100 --dropout 0.1 --hidden_dim 128 --hops 5 --peak_lr 0.01 --weight_decay 1e-05 --device 1
python mlp_pretrain_reverse.py --dataset texas --batch_size 183 --epochs 100 --dropout 0.1 --hidden_dim 128 --hops 5 --peak_lr 0.01 --weight_decay 1e-05 --device 1
python3 accuracy_serialsearch.py --dataset texas
python3 accuracy_parallelsearch.py --dataset texas

python mlp_pretrain_full.py --dataset cornell --batch_size 183 --epochs 100 --dropout 0.1 --hidden_dim 128 --hops 5 --peak_lr 0.01 --weight_decay 1e-05 --device 1
python mlp_pretrain_x.py --dataset cornell --batch_size 183 --epochs 100 --dropout 0.1 --hidden_dim 128 --hops 5 --peak_lr 0.01 --weight_decay 1e-05 --device 1
python mlp_pretrain_adj.py --dataset cornell --batch_size 183 --epochs 100 --dropout 0.1 --hidden_dim 128 --hops 5 --peak_lr 0.01 --weight_decay 1e-05 --device 1
python mlp_pretrain_reverse.py --dataset cornell --batch_size 183 --epochs 100 --dropout 0.1 --hidden_dim 128 --hops 5 --peak_lr 0.01 --weight_decay 1e-05 --device 1
python3 accuracy_serialsearch.py --dataset cornell
python3 accuracy_parallelsearch.py --dataset cornell

python mlp_pretrain_full.py --dataset actor --batch_size 7600 --epochs 100 --dropout 0.1 --hidden_dim 512 --hops 5 --peak_lr 0.001 --weight_decay 1e-05 --device 1
python mlp_pretrain_x.py --dataset actor --batch_size 7600 --epochs 100 --dropout 0.1 --hidden_dim 512 --hops 5 --peak_lr 0.001 --weight_decay 1e-05 --device 1
python mlp_pretrain_adj.py --dataset actor --batch_size 7600 --epochs 100 --dropout 0.1 --hidden_dim 512 --hops 5 --peak_lr 0.001 --weight_decay 1e-05 --device 1
python mlp_pretrain_reverse.py --dataset actor --batch_size 7600 --epochs 100 --dropout 0.1 --hidden_dim 512 --hops 5 --peak_lr 0.001 --weight_decay 1e-05 --device 1
python3 accuracy_serialsearch.py --dataset actor
python3 accuracy_parallelsearch.py --dataset actor

python mlp_pretrain_full.py --dataset photo --batch_size 7650 --epochs 100 --dropout 0.1 --hidden_dim 128 --hops 5 --peak_lr 0.01 --weight_decay 1e-05 --device 1
python mlp_pretrain_x.py --dataset photo --batch_size 7650 --epochs 100 --dropout 0.1 --hidden_dim 128 --hops 5 --peak_lr 0.01 --weight_decay 1e-05 --device 1
python mlp_pretrain_adj.py --dataset photo --batch_size 7650 --epochs 100 --dropout 0.1 --hidden_dim 128 --hops 5 --peak_lr 0.01 --weight_decay 1e-05 --device 1
python mlp_pretrain_reverse.py --dataset photo --batch_size 7650 --epochs 100 --dropout 0.1 --hidden_dim 128 --hops 5 --peak_lr 0.01 --weight_decay 1e-05 --device 1
python3 accuracy_serialsearch.py --dataset photo
python3 accuracy_parallelsearch.py --dataset photo

python mlp_pretrain_full.py --dataset wikics --batch_size 11701 --epochs 100 --dropout 0.1 --hidden_dim 128 --hops 5 --peak_lr 0.01 --weight_decay 1e-05 --device 1
python mlp_pretrain_x.py --dataset wikics --batch_size 11701 --epochs 100 --dropout 0.1 --hidden_dim 128 --hops 5 --peak_lr 0.01 --weight_decay 1e-05 --device 1
python mlp_pretrain_adj.py --dataset wikics --batch_size 11701 --epochs 100 --dropout 0.1 --hidden_dim 128 --hops 5 --peak_lr 0.01 --weight_decay 1e-05 --device 1
python mlp_pretrain_reverse.py --dataset wikics --batch_size 11701 --epochs 100 --dropout 0.1 --hidden_dim 128 --hops 5 --peak_lr 0.01 --weight_decay 1e-05 --device 1
python3 accuracy_serialsearch.py --dataset wikics
python3 accuracy_parallelsearch.py --dataset wikics

python mlp_pretrain_full.py --dataset reddit --alpha 0.01 --epochs 100 --batch_size 4000 --dropout 0.1 --hidden_dim 512 --hops 5 --peak_lr 0.001 --weight_decay 1e-05 --device 1
python mlp_pretrain_x.py --dataset reddit --alpha 0.01 --epochs 100 --batch_size 4000 --dropout 0.1 --hidden_dim 512 --hops 5 --peak_lr 0.001 --weight_decay 1e-05 --device 1
python mlp_pretrain_adj.py --dataset reddit --alpha 0.01 --epochs 100 --batch_size 4000 --dropout 0.1 --hidden_dim 512 --hops 5 --peak_lr 0.001 --weight_decay 1e-05 --device 1
python mlp_pretrain_reverse.py --dataset reddit --alpha 0.01 --epochs 100 --batch_size 4000 --dropout 0.1 --hidden_dim 512 --hops 5 --peak_lr 0.001 --weight_decay 1e-05 --device 1
python3 accuracy_serialsearch.py --dataset reddit
python3 accuracy_parallelsearch.py --dataset reddit

python mlp_pretrain_full.py --dataset products --alpha 0.01 --epochs 100 --batch_size 4000 --dropout 0.1 --hidden_dim 512 --hops 5 --peak_lr 0.001 --weight_decay 1e-05 --device 1
python mlp_pretrain_x.py --dataset products --alpha 0.01 --epochs 100 --batch_size 4000 --dropout 0.1 --hidden_dim 512 --hops 5 --peak_lr 0.001 --weight_decay 1e-05 --device 1
python mlp_pretrain_adj.py --dataset products --alpha 0.01 --epochs 100 --batch_size 4000 --dropout 0.1 --hidden_dim 512 --hops 5 --peak_lr 0.001 --weight_decay 1e-05 --device 1
python mlp_pretrain_reverse.py --dataset products --alpha 0.01 --epochs 100 --batch_size 4000 --dropout 0.1 --hidden_dim 512 --hops 5 --peak_lr 0.001 --weight_decay 1e-05 --device 1
python3 accuracy_serialsearch.py --dataset products
python3 accuracy_parallelsearch.py --dataset products
