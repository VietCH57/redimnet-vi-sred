# redimnet-vi-sred

Fine-tune **ReDimNet** ([IDRnD/redimnet](https://github.com/IDRnD/redimnet)) và **ReDimNet2** ([PalabraAI/redimnet2](https://github.com/PalabraAI/redimnet2)) đã pretrained cho bài toán speaker verification trên Vi-SRED26.

Công thức train theo đúng recipe chính thức của tác giả (ReDimNet2 `wespeaker_lite`, WeSpeaker), xem [`configs/README.md`](configs/README.md). Cách dùng giống repo [VietCH57/ECAPA-TDNN](https://github.com/VietCH57/ECAPA-TDNN): `train.py` / `infer.py`, mọi tham số truyền qua dòng lệnh hoặc YAML, tự resume, có `best.model` và `results.zip` để nộp bài.

**Code mô hình không bị sửa.** Hai package `redimnet/` và `redimnet2/` được chép nguyên vào `third_party/`, ghim ở một commit cố định (xem `third_party/*/SOURCE.md`). Checkpoint repo này lưu ra có đúng định dạng của tác giả (`model_config` + `state_dict`), nên nạp lại được bằng code gốc hoặc `torch.hub`.

---

## Cài đặt

```bash
git clone https://github.com/VietCH57/redimnet-vi-sred.git
cd redimnet-vi-sred
pip install -r requirements.txt     # Kaggle đã có sẵn các thư viện này
```

Checkpoint pretrained được tải tự động từ GitHub releases ở lần chạy đầu và lưu cache tại `~/.cache/redimnet-vi-sred`. Đổi thư mục cache bằng `--ckpt_dir` hoặc biến môi trường `REDIMNET_CACHE`.

## Chọn mô hình: `--pretrained`

| Giá trị | Tham số | Recipe |
|---|---|---|
| `redimnet:M-vb2+vox2+cnc-ft_mix` | 4.81M | `configs/redimnet_m/{ft,lm}.yaml` |
| `redimnet2:b4-vox2-lm` | 6.66M | `configs/redimnet2_b4/{ft,lm}.yaml` |
| `redimnet:S-vb2+vox2+cnc-ft_mix` | 3.15M | |
| `redimnet2:b6-vb2+vox2+cnc2_v0-lm` | 12.46M | vượt 7M; muốn dùng phải đặt `--max_params 0` (chỉ để làm teacher) |
| đường dẫn file `.pt` / `.model` / `.ckpt` | | checkpoint của tác giả hoặc của repo này |

Dạng `redimnet:<tên>` / `redimnet2:<tên>` nhận mọi tên checkpoint trong release của hai repo gốc. Với `--max_params 7e6` (mặc định), repo dừng ngay nếu encoder vượt 7M tham số.

## Định dạng dữ liệu

- **Train list:** mỗi dòng `<speaker> <đường/dẫn.wav>`. Ví dụ: `V*/train_split.txt` của repo [vi-sred-data-cleaning](https://github.com/VietCH57/vi-sred-data-cleaning).
- **Trial list:** mỗi dòng `<label> <a> <b>`, hoặc `<a> <b> target|nontarget`, hoặc `<a> <b>`.
- **Audio:** WAV, nếu không phải 16 kHz sẽ tự resample.

## Train

**Mọi giá trị trong `configs/` đều lấy từ recipe chính thức** của tác giả ReDimNet / ReDimNet2 và WeSpeaker. Bảng nguồn của từng giá trị, cùng các chỗ buộc phải lệch, nằm trong [`configs/README.md`](configs/README.md). `tests/test_official.py` so bằng số loss, lịch LR, lịch margin và low-pass của repo này với code chính thức.

YAML chỉ chứa siêu tham số; đường dẫn dữ liệu truyền qua dòng lệnh:

```bash
L=/kaggle/input/datasets/hongvitchugo/vi-sred-cleaned-labels/labels
D=/kaggle/input/datasets/hongvitchugo/vi-sred26-train/Vi-SRED26-Train/data
DATA="--train_list $L/V0/train_split.txt --train_path $D --eval_list $L/val_trials.txt --eval_path $D
      --eval_meta $L/val_trials_meta.csv --musan_path <MUSAN> --rir_path <RIR>"

# stage FT (tuỳ chọn): đoạn 2 s, speed perturb, margin như lúc pre-train
torchrun --standalone --nproc_per_node 2 train.py --config configs/redimnet2_b4/ft.yaml $DATA --save_path exps/b4_ft
# stage LM: recipe large-margin chính thức; tiếp từ FT (chép các hàng head của cùng speaker)
torchrun --standalone --nproc_per_node 2 train.py --config configs/redimnet2_b4/lm.yaml $DATA --save_path exps/b4_lm \
         --initial_model exps/b4_ft/model/last.ckpt
python train.py -h        # mọi tham số và giá trị mặc định
```

Tóm tắt recipe (chi tiết trong `configs/README.md`):

| | ReDimNet2-B4 | ReDimNet-M |
|---|---|---|
| loss | SphereFace2-C, s 32 | AAM, s 32 |
| margin FT / LM | 0,2 / 0,3 | 0,2 / 0,5 |
| crop FT / LM | 2,015 s / 6,015 s | 2,015 s / 6,015 s |
| speed perturb FT / LM | 0.9, 1.1 / tắt | 0.9, 1.1 / tắt |
| augment | `wespeaker_lite`, p 0,6 | `wespeaker`, p 0,6 |
| SpecAugment | theo checkpoint (tắt) | theo checkpoint (bật) |
| optimizer | SGD Nesterov 0,9, wd 1e-4 | như B4 |
| LR | 1e-4 → 2,5e-5 (exponential, warm-up 1 epoch từ 0), × batch/64 | như B4 |
| epoch | 5 mỗi stage | như B4 |

**Nhiều GPU: dùng `torchrun`.** Với `torchrun`:
- `--batch_size` là batch tổng, chia đều cho các GPU.
- `--accum_steps` cộng dồn gradient. LR nhân với `batch_size × accum_steps / 64`, đúng quy tắc `scale_ratio` của code chính thức.
- `--n_cpu` là tổng số worker đọc dữ liệu, chia đều cho các tiến trình.
- Khởi tạo head và đánh giá val được chia cho các GPU.
- Chỉ tiến trình 0 in log và ghi file.

**Head cho speaker mới:** mỗi hàng là embedding trung bình của speaker đó, trích bằng encoder pretrained (`--head_init centroid`). Khi truyền `--initial_model .../last.ckpt`, các hàng của cùng speaker được chép từ checkpoint, như `reassemble_projection_head` của code chính thức.

**Đánh giá trên val mỗi epoch.**
- Epoch 0 là mô hình trước khi train, dùng làm mốc so sánh.
- In `EER` (cosine) và `EER_center` (trừ trung bình embedding của 1000 file train, như `cal_mean` của WeSpeaker), kèm EER theo từng nhóm trial nếu có `--eval_meta`.
- `best.model` được chọn theo `--select_score`.

**Đầu ra** trong `--save_path`:

| File | Nội dung |
|---|---|
| `config.json` | toàn bộ tham số của lần chạy |
| `speakers.txt` | speaker → chỉ số lớp |
| `score.txt` | mỗi lần đánh giá một dòng |
| `model/model_XXXX.model`, `model/best.model`, `model/last.model` | encoder theo định dạng tác giả, kèm `center_mean` |
| `model/last.ckpt` | encoder, head, danh sách speaker, optimizer, epoch: dùng để resume và cho stage sau |

**Resume:** chạy lại đúng lệnh cũ, code tự train tiếp từ `last.ckpt`. `--max_hours` dừng sạch trước giới hạn 12 giờ của Kaggle.

**Bộ nhớ GPU:** `python probe_memory.py --pretrained redimnet2:b4-vox2-lm --loss sf2 --crop_sec 6.015` tìm batch lớn nhất vừa GPU và gợi ý `--batch_size` / `--accum_steps`.

**Đánh giá trong lúc train là phép xấp xỉ nhanh.** Các file có độ dài lệch nhau ≤ 2 % (`--eval_len_tol 0.02`) được gom chung batch và cắt về file ngắn nhất, kèm fp16. Số EER cuối cùng thì tính bằng `infer.py`, mặc định chính xác.

## Inference

```bash
# chấm public test -> results.txt (cùng thứ tự trial list) + results.zip để nộp
python infer.py --mode score --model <exp>/model/best.model \
    --trials .../public-test.txt --data_path .../data --out submission/results.txt --zip

# đánh giá trên trial list có nhãn; dùng được cả cho zero-shot (truyền tên checkpoint release)
python infer.py --mode eval --model redimnet:M-vb2+vox2+cnc-ft_mix --trials val_trials.txt --data_path .../data

# trích embedding -> .npz
python infer.py --mode embed --model <exp>/model/best.model --file_list val_split.txt --data_path .../data --out emb.npz
```

Một số cờ hay dùng:
- **`--score_norm center`:** dùng embedding trung bình đã lưu trong model. Hoặc tính lại từ `--cohort_list/--cohort_path`.
- **`--emb_cache file.npz`:** cache embedding, lần sau chấm lại không phải tính lại.
- **`--eval_mode full|crops|both`:** cách lấy embedding (cả câu, các đoạn cắt, hoặc cả hai).
- **`--amp`:** chạy fp16 cho nhanh.
- **`--eval_len_tol`:** để 0 (mặc định) cho kết quả chính xác.

## Cấu trúc repo

| File | Vai trò |
|---|---|
| `third_party/redimnet`, `third_party/redimnet2` | code mô hình gốc, không sửa (MIT) |
| `models.py` | nạp checkpoint release hoặc file, đoán family, xuất theo định dạng tác giả |
| `loss.py` | AAM (WeSpeaker `ArcMarginProduct`) và SphereFace2 (code chính thức ReDimNet2), khởi tạo bằng centroid |
| `dataLoader.py` | crop theo batch, speed perturb, các chuỗi augment chính thức (`wespeaker`, `wespeaker_lite`, `hard`) |
| `trainer.py` | vòng train/eval: SGD + ExponentialDecrease + MarginScheduler như code chính thức, ghép head, AMP, DDP, checkpoint |
| `dist_utils.py` | hàm phụ cho `torchrun` (khởi tạo, barrier, gom kết quả các GPU) |
| `train.py`, `config.py` | train qua dòng lệnh hoặc YAML |
| `infer.py`, `inference.py`, `tools.py` | chấm điểm, đánh giá, trích embedding, EER/minDCF |
| `probe_memory.py` | tìm batch size vừa GPU |
| `configs/` | `redimnet_m/{ft,lm}.yaml`, `redimnet2_b4/{ft,lm}.yaml`, `README.md` (nguồn của từng giá trị) |
| `backend.py` | back-end: nội suy trọng số, chuẩn hoá in-domain, AS-norm, QMF |
| `tests/` | `test_official.py` (so với code chính thức), dữ liệu giả và `smoke_test.sh` (CPU, khoảng 10 phút) |

## Giấy phép và nguồn

Repo theo giấy phép MIT. `third_party/` giữ nguyên giấy phép MIT của ID R&D, Inc. và Palabra.ai.

Loss, augment, lịch LR/margin chép theo code train chính thức của ReDimNet2 ([PalabraAI/redimnet2](https://github.com/PalabraAI/redimnet2) `applications/speaker_rec/wespeaker_lite`, Apache-2.0, fork của WeSpeaker) và WeSpeaker. Cách dùng dòng lệnh theo [VietCH57/ECAPA-TDNN](https://github.com/VietCH57/ECAPA-TDNN).

## Reference

```
@inproceedings{yakovlev24_interspeech,
  title     = {Reshape Dimensions Network for Speaker Recognition},
  author    = {Ivan Yakovlev and Rostislav Makarov and Andrei Balykin and Pavel Malov and Anton Okhotnikov and Nikita Torgashov},
  booktitle = {Interspeech 2024},
  year      = {2024},
  pages     = {3235--3239},
  doi       = {10.21437/Interspeech.2024-2116}
}
```

```
@inproceedings{redimnet2_2026,
  title     = {ReDimNet2: Scaling Speaker Verification via Time-Pooled Dimension Reshaping},
  author    = {Yakovlev, Ivan and Okhotnikov, Anton},
  booktitle = {Proceedings of Interspeech 2026},
  year      = {2026},
  eprint    = {2603.11841},
  archivePrefix = {arXiv},
  url       = {https://arxiv.org/abs/2603.11841}
}
@inproceedings{wang2023wespeaker,
  title     = {Wespeaker: A research and production oriented speaker embedding learning toolkit},
  author    = {Wang, Hongji and Liang, Chengdong and Wang, Shuai and Chen, Zhengyang and Zhang, Binbin and Xiang, Xu and Deng, Yanlei and Qian, Yanmin},
  booktitle = {ICASSP 2023},
  year      = {2023}
}
@inproceedings{qi2018lowshot,
  title     = {Low-Shot Learning with Imprinted Weights},
  author    = {Qi, Hang and Brown, Matthew and Lowe, David G.},
  booktitle = {CVPR 2018},
  year      = {2018}
}
@inproceedings{han2023sphereface2sv,
  title     = {Exploring Binary Classification Loss for Speaker Verification},
  author    = {Han, Bing and Chen, Zhengyang and Qian, Yanmin},
  booktitle = {ICASSP 2023},
  year      = {2023}
}
@inproceedings{li2018explicit,
  title     = {Explicit Inductive Bias for Transfer Learning with Convolutional Networks},
  author    = {Li, Xuhong and Grandvalet, Yves and Davoine, Franck},
  booktitle = {ICML 2018},
  year      = {2018}
}
```
