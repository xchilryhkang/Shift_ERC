# shift_ERC — baseline

Code gốc lấy từ ReDiFu (liuying2023912/ReDiFu). Toàn bộ kiến trúc của ReDiFu đã được gỡ bỏ
(RGAT, Differential Transformer, filter module, MutualFormer, speaker embedding, loss phụ từng
modality, ma trận kề trong dataloader). Model hiện tại chỉ còn:

    h_i^m = Linear_m(x_i^m)             m in {t, a, v}, cùng chiều hidden_dim
    h_i   = h_i^t + h_i^a + h_i^v
    y_i   = Linear(Dropout(ReLU(h_i)))

Loss: masked NLL trên output cuối (IEMOCAP có class weight, MELD không, giữ như code gốc).

## Dữ liệu
Đặt `iemocap_multimodal_features.pkl` và `meld_multimodal_features.pkl` (bản preprocessed của ReDiFu)
vào một thư mục rồi truyền qua `--data_dir`.

## Chạy
```bash
bash run_iemocap.sh
bash run_meld.sh
```
Tuỳ chọn thêm: `--modals t` (chỉ text), `--modals ta`, `--modals tav` (mặc định).
