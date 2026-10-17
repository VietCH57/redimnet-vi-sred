# Chạy trên Kaggle (2× T4)

Cách nhanh nhất là dùng notebook `vi_sred26_finetune_redimnet.ipynb`. Ô đầu tiên của notebook có hai lựa chọn:

- `MODEL`: `redimnet_m` (ReDimNet-M, 4.81M) hoặc `redimnet2_b4` (ReDimNet2-B4, 6.66M).
- `LABEL_SET`: `V0` / `V1` / `V2` / `V3`, các bộ nhãn từ repo `vi-sred-data-cleaning`.

Notebook chạy lần lượt: zero-shot trên val → FT → LM → chấm chính xác val và public test → đóng gói `results.zip`.

Cần **bật Internet**, vì repo và checkpoint pretrained được tải từ GitHub.

| Dataset | Đường dẫn |
|---|---|
| Nhãn đã làm sạch | `/kaggle/input/datasets/hongvitchugo/vi-sred-cleaned-labels/labels` (`V0..V3/train_split.txt`, `val_trials.txt`, `val_trials_meta.csv`, `val_split.txt`) |
| Audio train | `/kaggle/input/datasets/hongvitchugo/vi-sred26-train` |
| Public test | `/kaggle/input/datasets/hongvitchugo/vi-sred26-publictest` |
| MUSAN | `/kaggle/input/datasets/nhattruongdev/musan-noise` |
| RIR | `/kaggle/input/datasets/nhattruongdev/rirs-noises` |

Chạy tay (tương đương notebook):

```bash
L=/kaggle/input/datasets/hongvitchugo/vi-sred-cleaned-labels/labels
D=/kaggle/input/datasets/hongvitchugo/vi-sred26-train/Vi-SRED26-Train/data
DATA="--train_list $L/V0/train_split.txt --train_path $D --eval_list $L/val_trials.txt --eval_path $D --eval_meta $L/val_trials_meta.csv
      --musan_path /kaggle/input/datasets/nhattruongdev/musan-noise --rir_path /kaggle/input/datasets/nhattruongdev/rirs-noises"
python probe_memory.py --pretrained redimnet2:b4-vox2-lm --loss sf2 --crop_sec 6.015 --target_batch 128
torchrun --standalone --nproc_per_node 2 train.py --config configs/redimnet2_b4/ft.yaml $DATA --save_path /kaggle/working/exps/b4_V0_ft
torchrun --standalone --nproc_per_node 2 train.py --config configs/redimnet2_b4/lm.yaml $DATA --save_path /kaggle/working/exps/b4_V0_lm \
         --initial_model /kaggle/working/exps/b4_V0_ft/model/last.ckpt
python infer.py --mode score --model /kaggle/working/exps/b4_V0_lm/model/best.model \
    --trials /kaggle/input/datasets/hongvitchugo/vi-sred26-publictest/public-test.txt \
    --data_path /kaggle/input/datasets/hongvitchugo/vi-sred26-publictest/data \
    --score_norm center --out /kaggle/working/submission/results.txt --zip
```

Nếu bị OOM: giảm `--batch_size` và tăng `--accum_steps` sao cho tích không đổi. LR tự nhân theo `batch_size × accum_steps / 64`, nên công thức vẫn giữ nguyên.

**Resume khi session bị ngắt:** chạy lại đúng lệnh cũ. Nếu `last.ckpt` nằm trong output của notebook trước (đã add làm dataset), dùng `--resume /kaggle/input/.../model/last.ckpt --save_path /kaggle/working/exps/<tên>`.
