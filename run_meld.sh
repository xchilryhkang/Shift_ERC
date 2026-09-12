python train.py --Dataset="MELD" --data_dir="data" --save_model_path="./MELD" \
  --lr=0.00005 --dropout=0.6 --l2=0.00005 --batch-size=16 --hidden_dim=256 --epochs=20 --seed=123 \
  --use_graph --heads=4 --layers=1 --window=4 --link_prev_same --init_lambda=0.5 --prior_lr=0.01 \
  --use_shift --w_shift=0.3
