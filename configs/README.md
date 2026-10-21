# Công thức train và nguồn của từng giá trị

## Nguồn

| Mã | Nguồn |
|---|---|
| **[R2-PTN]**, **[R2-LM]** | Recipe chính thức của ReDimNet2-B4 trong repo của tác giả: [PalabraAI/redimnet2](https://github.com/PalabraAI/redimnet2) `applications/speaker_rec/examples/vox2/recipes/redimnet2/b4/{ptn,lm}.yaml`. Checkpoint `b4-vox2-lm` chính là đầu ra của hai recipe này. |
| **[R2]** | Code train chính thức của ReDimNet2, cùng repo trên: `applications/speaker_rec/wespeaker_lite` (`bin/train.py`, `utils/executor.py`, `utils/schedulers.py`, `dataset/processor.py`, `dataset/augs.py`, `models/projections.py`). Đây là bản fork rút gọn của WeSpeaker do chính tác giả ReDimNet (I. Yakovlev) công bố. |
| **[R2-HA]** | Recipe chính thức cho dữ liệu VoxBlink2 + VoxCeleb2 + CN-Celeb2 (augment "hard"): `examples/vb2_vox2_cnc2/recipes/redimnet2/b6/{ptn,lm}.yaml`. |
| **[RD]** | Yakovlev et al., *Reshape Dimensions Network for Speaker Recognition*, Interspeech 2024, [arXiv 2407.18223](https://arxiv.org/abs/2407.18223), mục 3. README của [IDRnD/ReDimNet](https://github.com/IDRnD/ReDimNet) ghi rõ ReDimNet được train bằng WeSpeaker. |
| **[WS-RD]**, **[WS-LM]** | WeSpeaker [wenet-e2e/wespeaker](https://github.com/wenet-e2e/wespeaker) `examples/voxceleb/v2/conf/redimnet.yaml` và `resnet_lm.yaml`. `run.sh` dùng `resnet_lm.yaml` cho stage LM của mọi mô hình. |
| **[WS]** | Code WeSpeaker: `wespeaker/bin/train.py`, `wespeaker/dataset/processor.py`, `local/score.sh`, `wespeaker/bin/score.py`. |

`tests/test_official.py` kiểm tra bằng số rằng loss SphereFace2, AAM, lịch LR, lịch margin và LowPassAug của repo này cho kết quả trùng với code chính thức.

## Hai stage

Cả hai họ mô hình đều được tác giả train theo cùng một quy trình: **pre-train** trên đoạn 2 s với speed perturb và margin tăng dần, sau đó **large-margin fine-tuning (LM)** trên đoạn 6 s với margin lớn hơn, tắt speed perturb, LR nhỏ.

- **`lm.yaml` là recipe LM chính thức, áp nguyên lên dữ liệu Vi-SRED.** Stage LM là stage duy nhất trong quy trình chính thức xuất phát từ một checkpoint đã train xong, nên đây là cách fine-tune có nguồn rõ nhất.
- **`ft.yaml` (tuỳ chọn, chạy trước `lm.yaml`).** Không có recipe chính thức nào cho việc thích nghi mô hình đã train sang một tập speaker mới. `ft.yaml` lấy nguyên `lm.yaml` rồi trả đúng 3 thiết lập mà stage LM thay đổi về lại giá trị lúc pre-train: crop 2,015 s, speed perturb 0.9/1.1, margin 0.2. Tức là dữ liệu và loss giống pre-train, còn optimizer giống stage chính thức tiếp tục từ checkpoint. Tiền lệ cho việc fine-tune mô hình nguồn trên dữ liệu miền đích có speed perturb: hệ thống DKU-MSXF ở track domain adaptation của VoxSRC-23 ([arXiv 2308.08766](https://arxiv.org/abs/2308.08766), mục 4).

Muốn chạy LM sau FT: `--initial_model <ft>/model/last.ckpt`. Khi đó các hàng của head tương ứng với cùng speaker được chép sang, giống `reassemble_projection_head` của [R2] (chỉ lấy hàng tốc độ 1.0).

## ReDimNet2-B4 (`configs/redimnet2_b4/`)

| Tham số | `ft.yaml` | `lm.yaml` | Nguồn |
|---|---|---|---|
| crop | 2,015 s | 6,015 s | [R2-PTN] `num_frms 200`, [R2-LM] `num_frms 600`; độ dài = (num_frms − 1)·10 ms + 25 ms ([R2] `dataset.py`) |
| lọc file ngắn | < 1,0 s | < 1,0 s | [R2] `min_num_frames 100` |
| speed perturb | 0.9 / 1.1, ×3 lớp | tắt | [R2-PTN] / [R2-LM]; [R2] `processor.speed_perturb` |
| augment | `wespeaker_lite`, p 0,6 | như FT | [R2-PTN] = [R2-LM] `aug_setup: wespeaker_augs, aug_prob 0.6, preset default` → [R2] `add_reverb_noise_gain` |
| SpecAugment | theo checkpoint (tắt) | như FT | [R2-PTN] / [R2-LM] `do_spec_aug: false` |
| loss | SphereFace2-C, s 32, λ 0,7, t 3 | như FT | [R2-PTN] / [R2-LM] `projection_args` |
| margin | 0,2 cố định | 0,3 cố định | [R2-PTN] `final_margin 0.2`; [R2-LM] 0,3 → 0,3. Đoạn margin tăng 0 → 0,2 (epoch 20–40) chỉ dành cho train từ đầu, checkpoint đã qua đoạn đó |
| optimizer | SGD, momentum 0,9, Nesterov | như FT | [R2-LM] |
| weight decay | 1e-4, head ×1 | như FT | [R2-LM] (không có `loss_weight_decay_mult` nên ×1) |
| LR | 1e-4 → 2,5e-5, ExponentialDecrease theo từng bước | như FT | [R2-LM] `scheduler_args` |
| nhân LR | × batch hiệu dụng / 64 | như FT | [R2] `train.py`: `scale_ratio = world_size * batch_size / 64` |
| warm-up | 1 epoch, từ 0 | như FT | [R2-LM] `warm_up_epoch 1, warm_from_zero true` |
| số epoch | 5 | 5 | [R2-LM] `num_epochs 5` |
| batch | 128 = 64/GPU × 2 GPU | như FT | [R2-LM] `batch_size 64` mỗi GPU |
| AMP | bật, head luôn fp32 | như FT | [R2-LM] `enable_amp true`; [R2] `executor.run_epoch` |
| grad clip | không | không | [R2] `executor.run_epoch` |

## ReDimNet-M (`configs/redimnet_m/`)

| Tham số | `ft.yaml` | `lm.yaml` | Nguồn |
|---|---|---|---|
| crop | 2,015 s | 6,015 s | [RD] "2-second segments" / "6 seconds"; [WS-RD] `num_frms 200`, [WS-LM] `num_frms 600` |
| speed perturb | 0.9 / 1.1 | tắt | [RD] mục 3.1 / 3.2 |
| augment | `wespeaker` (bản gốc của WeSpeaker), p 0,6 | như FT | [WS-RD] / [WS-LM] `aug_prob 0.6`; [WS] `processor.add_reverb_noise` |
| SpecAugment | theo checkpoint (bật, freq [0,10], time [0,20]) | như FT | `model_config` lưu trong chính checkpoint `M-vb2+vox2+cnc-ft_mix` |
| loss | AAM, s 32 | như FT | [RD] "AAM softmax"; [WS-RD] `arc_margin, scale 32, easy_margin False` |
| margin | 0,2 cố định | 0,5 cố định | [RD]: tăng tới 0,2 khi pre-train; "constant 0.5" khi LM; [WS-LM] |
| optimizer, wd, LR, warm-up, epoch, batch | như B4 | như B4 | [WS-LM]: SGD Nesterov 0,9, wd 1e-4, 1e-4 → 2,5e-5, warm-up 1 epoch từ 0, 5 epoch, batch 64/GPU; [WS] `train.py` có cùng `scale_ratio` |
| AMP | bật | bật | **Lệch:** [WS-LM] `enable_amp: False`. Trên T4 cần fp16 để đủ nhanh; head vẫn chạy fp32 |

## Đánh giá trong lúc train

Giống [R2] (`bin/train.py` + `utils/executor.eval_epoch_simple`, `eval_dur: 8.0`): mỗi file val lấy **8 giây đầu**, file ngắn hơn được lặp cho đủ 8 s (`Vox1TestEvalDatasetSimple`), fp16, batch = batch train / 4 = 16, cosine. Mọi đầu vào cùng một độ dài. Khác [R2] ở một điểm: chỉ GPU 0 đánh giá (GPU 0 giữ mô hình được lưu), các GPU khác chờ. EER cuối cùng tính bằng `infer.py` trên cả câu, fp32.

## Những chỗ lệch so với nguồn (đã ghi `DEVIATION` trong YAML)

1. **Khởi tạo head cho speaker mới.** [R2] chép hàng của speaker đã có trong checkpoint và gán hàng ngẫu nhiên cho speaker mới. Checkpoint release không kèm head, còn speaker Vi-SRED đều là speaker mới. Repo này đặt mỗi hàng bằng embedding trung bình của speaker đó, trích bằng encoder pretrained (*weight imprinting*, Qi, Brown & Lowe, CVPR 2018). Cách này tránh việc gradient từ một head ngẫu nhiên phá encoder ngay ở các bước đầu; hiện tượng đó được phân tích trong Kumar et al., *Fine-Tuning can Distort Pretrained Features*, ICLR 2022 (LP-FT). Muốn dùng đúng cách chính thức (xavier): `--head_init random`.
2. **AMP cho ReDimNet-M** (xem bảng trên).
3. **Chọn checkpoint.** WeSpeaker lấy model cuối của stage LM (`num_avg 1`). Repo này đánh giá trên val sau mỗi epoch và lưu `best.model`, kèm `last.model`. Val gồm các người tách riêng khỏi train (repo `vi-sred-data-cleaning`), nên việc chọn này không dùng tới public test.
4. **Chấm điểm `center`.** WeSpeaker `local/score.sh` dùng `--cal_mean True`: trừ trung bình embedding (thô) của **toàn bộ** tập train rồi tính cosine. Repo này trừ trung bình embedding đã chuẩn hoá L2 của 1000 file train ngẫu nhiên khi đánh giá trong lúc train (cho nhanh). `infer.py --cohort_size` tăng được tới toàn bộ tập.

## ReDimNet-M bền hơn với câu ngắn và băng hẹp (`ft_robust.yaml`, `lm_robust.yaml`)

Biến thể của `ft.yaml` / `lm.yaml` cho ReDimNet-M. Lý do (chỉ từ val): EER của các trial val có câu ngắn hơn 2 s, hoặc có phía băng hẹp / lệch băng thông, cao gấp 4–5 lần trial dài, băng rộng. Mọi giá trị khác `ft.yaml` / `lm.yaml` đều ghi `DEVIATION` trong YAML.

| Thay đổi | `ft_robust.yaml` | `lm_robust.yaml` | Nguồn / lý do |
|---|---|---|---|
| độ dài đoạn train (một độ dài mỗi batch, đều) | {1,015; 2,015; 3,015} s | {2,015; 6,015} s | **tự đặt**. 1,015 s = 100 frame = độ dài tối thiểu chính thức (`min_num_frames 100`); trung bình FT vẫn 2,015 s như [WS-RD] |
| augment | cây `hard` của [R2-HA], p 0,9 | như FT, p 0,7 | [R2-HA] `ptn` 0,9 / `lm` 0,7; ReDimNet-M gốc dùng augment WeSpeaker thường |
| low-pass | p 0,5, cutoff {0,5; 0,6}·Nyquist (4 / 4,8 kHz) | như FT | [R2-HA] p 0,25, cutoff 0,6 (0,5 nằm trong comment của [R2-HA]); **p 0,5 là tự đặt** theo EDA (~23 % file train ≤ 4 kHz) |
| L2-SP | α = 0,01, anchor = checkpoint pretrained, encoder không weight decay | như FT (anchor vẫn là pretrained, không phải kết quả FT) | Li, Grandvalet & Davoine, ICML 2018 (arXiv 1802.01483): Ω = α/2·‖w − w⁰‖² thay cho weight decay trên các lớp pretrained; head giữ weight decay. Bài chọn α bằng cross-validation trong {10⁻³, 10⁻², 10⁻¹, 1}; **ở đây không tune, lấy giữa lưới** |

`--l2sp_anchor <spec>` nạp anchor từ một encoder bất kỳ (mặc định: trọng số lúc bắt đầu run). Khi `--l2sp > 0`, encoder không có weight decay.

## Các tuỳ chọn không thuộc recipe chính thức (mặc định tắt)

`--freeze_epochs`, `--l2sp` (Li et al., ICML 2018), `--layer_decay`, `--bn_mode frozen`, `--grad_clip`, `--wd_skip_1d`, `--optimizer adamw`. Chúng được giữ lại để làm ablation, nhưng không có trong bất kỳ YAML nào.

## Augment `hard` (ablation)

`--aug_chain hard --hard_prob 0.9` (pre-train) hoặc `0.7` (LM) là cây augment của [R2-HA]: MUSAN music / noise / babble (p 0,8), sau đó RIR (p 0,2), sau đó low-pass (p 0,25, cutoff 0,6·Nyquist = 4,8 kHz). Tác giả dùng cây này khi dữ liệu train có CN-Celeb2 (dữ liệu "in the wild" nhiều thể loại). Bộ low-pass gần với hiện tượng EDA thấy trên tập train: khoảng 23 % file có băng thông hiệu dụng ≤ 4 kHz. Trong comment của [R2-HA] có thêm cutoff 0,5 (4 kHz, giả lập điện thoại); muốn dùng thì đặt `--lowpass_cutoffs 0.5,0.6`.
