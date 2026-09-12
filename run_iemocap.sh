python train.py --Dataset="IEMOCAP" --data_dir="data" --save_model_path="./IEMOCAP" \
  --lr=0.0001 --dropout=0.5 --l2=0.00005 --batch-size=16 --hidden_dim=512 --epochs=50 --seed=2094 \
  --use_graph --heads=4 --layers=1 --window=4 --init_lambda=0.5 --prior_lr=0.01 \
  --use_shift --w_shift=0.3
