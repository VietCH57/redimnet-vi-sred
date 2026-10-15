# Chạy trên Kaggle (2× T4)

## 0. Chuẩn bị

```python
!git clone https://github.com/VietCH57/redimnet-vi-sred.git /kaggle/working/redimnet-vi-sred
%cd /kaggle/working/redimnet-vi-sred
```

Cần **bật Internet**, vì checkpoint pretrained được tải từ GitHub releases ở lần chạy đầu.

| Dataset | Đường dẫn |
|---|---|
| Train | `/kaggle/input/datasets/hongvitchugo/vi-sred26-train` |
| Public test | `/kaggle/input/datasets/hongvitchugo/vi-sred26-publictest` |
| MUSAN | `/kaggle/input/datasets/nhattruongdev/musan-noise` |
| RIR | `/kaggle/input/datasets/nhattruongdev/rirs-noises` |

## 1. Chọn batch size (một lần, khoảng 1 phút)

```bash
!python probe_memory.py --crop_sec 3 --target_batch 128
```

Lệnh in ra batch lớn nhất mỗi GPU chịu được, cùng một cặp `--batch_size` / `--accum_steps` gợi ý. Config mặc định dùng `batch_size: 64`, tức 32 mẫu mỗi GPU.

## 2. Zero-shot (mốc so sánh, không cần train)

```bash
!python infer.py --mode eval --model redimnet:M-vb2+vox2+cnc-ft_mix \
    --trials /kaggle/input/datasets/hongvitchugo/vi-sred26-train/val_trials.txt \
    --data_path /kaggle/input/datasets/hongvitchugo/vi-sred26-train/Vi-SRED26-Train/data --out /kaggle/working/zs/val.txt
```

## 3. Fine-tune

```bash
!python train.py --config configs/ft_redimnet_m.yaml
```

- **Kiểm tra log trước khi để chạy lâu.** Đầu log in thư mục MUSAN/RIR tìm được, xác suất từng loại augment, số lớp (= số speaker × 3 vì speed perturb) và EER ở epoch 0 (phải xấp xỉ zero-shot, khoảng 2.3% trên val).
- **Dòng tiến độ** in thời gian mỗi epoch và bộ nhớ GPU đỉnh. Nếu bị OOM, giảm `--batch_size` và tăng `--accum_steps`.
- **Resume khi session bị ngắt:** chạy lại đúng lệnh cũ. Nếu `last.ckpt` nằm trong output của notebook trước (đã add làm dataset), dùng `--resume /kaggle/input/.../model/last.ckpt --save_path /kaggle/working/exps/<tên>`.

## 4. LMF (tuỳ chọn)

```bash
!python train.py --config configs/lmf_redimnet_m.yaml --initial_model /kaggle/working/exps/redimnet_m_ft/model/last.ckpt
```

## 5. File nộp

```bash
!python infer.py --mode score --model /kaggle/working/exps/redimnet_m_ft/model/best.model \
    --trials /kaggle/input/datasets/hongvitchugo/vi-sred26-publictest/public-test.txt \
    --data_path /kaggle/input/datasets/hongvitchugo/vi-sred26-publictest/data \
    --out /kaggle/working/submission/results.txt --zip
```

- **Định dạng:** `results.txt` có mỗi dòng `file1.wav file2.wav score`, cùng thứ tự với `public-test.txt`. `results.zip` là file để nộp.
- **Chấm theo center:** thêm `--score_norm center` để dùng embedding trung bình lưu trong model. Chỉ nên dùng khi `EER_center` trên val tốt hơn `EER`.
- **Nhãn của public test:** vì `public-test.txt` có nhãn, script in luôn EER trên public test.
