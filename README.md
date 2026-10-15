# redimnet-vi-sred

Fine-tune **ReDimNet** ([IDRnD/redimnet](https://github.com/IDRnD/redimnet)) và **ReDimNet2** ([PalabraAI/redimnet2](https://github.com/PalabraAI/redimnet2)) đã pretrained cho bài toán speaker verification trên Vi-SRED26.

Cách dùng giống repo [VietCH57/ECAPA-TDNN](https://github.com/VietCH57/ECAPA-TDNN): `train.py` / `infer.py`, mọi tham số truyền qua dòng lệnh hoặc YAML, tự resume, có `best.model` và `results.zip` để nộp bài.

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

| Giá trị | Tham số | Ghi chú |
|---|---|---|
| `redimnet:M-vb2+vox2+cnc-ft_mix` | 4.81M | mặc định; zero-shot Vi-SRED26: val 2.26%, public test 5.37% |
| `redimnet:S-vb2+vox2+cnc-ft_mix` | 3.15M | |
| `redimnet2:b4-vox2-lm` | 6.66M | |
| `redimnet2:b6-vb2+vox2+cnc2_v0-lm` | 12.46M | vượt 7M; muốn dùng phải đặt `--max_params 0` |
| đường dẫn file `.pt` / `.model` / `.ckpt` | | checkpoint của tác giả hoặc của repo này |

Dạng `redimnet:<tên>` / `redimnet2:<tên>` nhận mọi tên checkpoint trong release của hai repo gốc. Với `--max_params 7e6` (mặc định), repo dừng ngay nếu encoder vượt 7M tham số.

## Định dạng dữ liệu

Giống repo ECAPA:
- **Train list:** mỗi dòng `<speaker> <đường/dẫn.wav>`.
- **Trial list:** mỗi dòng `<label> <a> <b>`, hoặc `<a> <b> target|nontarget`, hoặc `<a> <b>`.
- **Audio:** WAV, nếu không phải 16 kHz sẽ tự resample.

## Train

```bash
python train.py --config configs/ft_redimnet_m.yaml                 # fine-tune
python train.py --config configs/ft_redimnet_m.yaml --batch_size 48  # ghi đè tham số bất kỳ
python train.py -h                                                  # mọi tham số và giá trị mặc định
```

### Công thức mặc định (`configs/ft_redimnet_m.yaml`)

1. **Khởi tạo lớp AAM bằng centroid** (`--head_init centroid`): trọng số mỗi lớp là embedding trung bình của vài câu của speaker đó, trích bằng mô hình pretrained. Lớp speed-perturb dùng audio đã đổi tốc độ, nên ngay từ đầu các lớp đã phân biệt được với nhau.
2. **Khởi động lớp AAM** (`--freeze_epochs 1`): chỉ train lớp AAM, encoder đứng yên.
3. **Fine-tune toàn mạng.**
   - AdamW: LR encoder `1e-4`, LR lớp AAM `1e-3`.
   - LR warmup 1 epoch, sau đó giảm theo cosine.
   - L2-SP `1e-4` kéo encoder về gần trọng số ban đầu, chống quên kiến thức pretrained.
   - BatchNorm giữ running statistics pretrained (`--bn_mode frozen`); weight/bias của BN vẫn được học.
   - Grad clip 5, AMP fp16.
4. **Augment:**
   - MUSAN và RIR như repo ECAPA.
   - Speed perturb 0.9/1.1: mỗi tốc độ được coi là speaker mới, nên số lớp nhân 3.
   - Kênh băng hẹp (`--narrowband_prob 0.2`): resample xuống 8 kHz rồi lên lại, hoặc lọc dải điện thoại 300–3400 Hz.
   - SpecAugment trong front-end của mô hình.
   - Mỗi batch chọn ngẫu nhiên đoạn dài 2 s hoặc 3 s (`--crop_sec 2,3`).
5. **Đánh giá trên val mỗi epoch.**
   - Epoch 0 là mô hình trước khi train, dùng làm mốc so sánh.
   - In hai EER: `EER` (cosine) và `EER_center` (trừ embedding trung bình của 1000 file train trước khi tính cosine).
   - `best.model` được chọn theo `--select_score`.

Large-margin fine-tuning (LMF) sau đó:

```bash
python train.py --config configs/lmf_redimnet_m.yaml --initial_model <exp>/model/last.ckpt
```

`last.ckpt` có cả lớp AAM, nên LMF dùng tiếp được. Nếu truyền `best.model`, lớp AAM được khởi tạo lại bằng centroid.

### Các nhóm tham số

| Nhóm | Tham số |
|---|---|
| Dữ liệu | `--train_list --train_path --crop_sec --crop_probs --batch_size --n_cpu --seed` |
| Augment | `--musan_path --rir_path --aug_probs --snr_* --num_* --speed_perturb --speed_probs --narrowband_prob --narrowband_types --specaug --specaug_freq --specaug_time` |
| Mô hình / loss | `--pretrained --ckpt_dir --max_params --m --m_start --m_warmup_epochs --s --head_init --head_init_utts --head_init_sec` |
| Tối ưu | `--max_epoch --optimizer --lr_backbone --lr_head --layer_decay --scheduler --lr_min_ratio --lr_decay --warmup_epochs --freeze_epochs --weight_decay --l2sp --accum_steps --bn_mode --grad_clip --amp` |
| Đánh giá | `--eval_list --eval_path --eval_meta --test_step --eval_init --eval_mode --eval_sec --num_eval --eval_min_sec --eval_max_sec --eval_len_tol --eval_batch --eval_amp --cohort_size --select_score` |
| Checkpoint | `--save_path --initial_model --resume --max_hours --keep_last --save_best --log_interval --device --data_parallel` |

**Đầu ra** trong `--save_path`:

| File | Nội dung |
|---|---|
| `config.json` | toàn bộ tham số của lần chạy |
| `speakers.txt` | speaker → chỉ số lớp |
| `score.txt` | mỗi lần đánh giá một dòng |
| `model/model_XXXX.model`, `model/best.model` | encoder theo định dạng tác giả, kèm `center_mean` |
| `model/last.ckpt` | encoder, lớp AAM, optimizer, epoch: dùng để resume |
| `model/anchor.pt` | trọng số tham chiếu của L2-SP |

**Resume:** chạy lại đúng lệnh cũ, code tự train tiếp từ `last.ckpt`. `--max_hours` dừng sạch trước giới hạn 12 giờ của Kaggle.

**Bộ nhớ GPU:** ReDimNet tốn nhiều bộ nhớ khi train. Chạy `python probe_memory.py --crop_sec 3` để tìm batch lớn nhất vừa GPU và nhận gợi ý `--batch_size` / `--accum_steps`.

**Đánh giá trong lúc train là phép xấp xỉ nhanh.** Các file có độ dài lệch nhau ≤ 5% (`--eval_len_tol 0.05`) được gom chung batch và cắt về file ngắn nhất, kèm fp16. Kết quả gần như trùng bản chính xác (cosine ≥ 0.98 so với embedding chính xác). Số EER cuối cùng thì tính bằng `infer.py`, mặc định chính xác.

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
| `loss.py` | AAM-softmax: kích thước embedding tuỳ chỉnh, đổi được margin, khởi tạo bằng centroid |
| `dataLoader.py` | crop theo batch, MUSAN/RIR (theo repo ECAPA), speed perturb, kênh băng hẹp |
| `trainer.py` | vòng train/eval: freeze, lịch LR, L2-SP, AMP, DataParallel, checkpoint |
| `train.py`, `config.py` | train qua dòng lệnh hoặc YAML |
| `infer.py`, `inference.py`, `tools.py` | chấm điểm, đánh giá, trích embedding, EER/minDCF |
| `probe_memory.py` | tìm batch size vừa GPU |
| `configs/` | `ft_redimnet_m.yaml`, `lmf_redimnet_m.yaml` |
| `tests/` | dữ liệu giả và `smoke_test.sh` (CPU, vài phút) |

## Giấy phép và nguồn

Repo theo giấy phép MIT. `third_party/` giữ nguyên giấy phép MIT của ID R&D, Inc. và Palabra.ai.

Phần augment và cách dùng lấy theo [TaoRuijie/ECAPA-TDNN](https://github.com/TaoRuijie/ECAPA-TDNN) (qua [VietCH57/ECAPA-TDNN](https://github.com/VietCH57/ECAPA-TDNN)). Speed perturb và large-margin fine-tuning theo WeSpeaker và các bài báo ReDimNet.

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
@inproceedings{li2018explicit,
  title     = {Explicit Inductive Bias for Transfer Learning with Convolutional Networks},
  author    = {Li, Xuhong and Grandvalet, Yves and Davoine, Franck},
  booktitle = {ICML 2018},
  year      = {2018}
}
```
