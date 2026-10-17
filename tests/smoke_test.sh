#!/usr/bin/env bash
# End-to-end smoke test on synthetic data (CPU): for ReDimNet-M and ReDimNet2-B4, FT 1 epoch, resume to 2,
# LM from the FT last.ckpt (head reassembly), exact inference with center scoring.
set -euo pipefail
cd "$(dirname "$0")/.."
R=${1:-/tmp/redimnet_smoke}
rm -rf "$R"; python tests/make_fake_data.py "$R"
K=$R/kaggle/input/datasets; T=$K/hongvitchugo/vi-sred26-train; P=$K/hongvitchugo/vi-sred26-publictest
L=$K/hongvitchugo/vi-sred-cleaned-labels/labels; D=$T/Vi-SRED26-Train/data
SMALL="--device cpu --batch_size 4 --n_cpu 2 --cohort_size 20 --log_interval 100 --eval_dur 2 --eval_batch 4 --max_hours 0
       --musan_path $K/nhattruongdev/musan-noise --rir_path $K/nhattruongdev/rirs-noises
       --eval_list $L/val_trials.txt --eval_path $D --eval_meta $L/val_trials_meta.csv"
for M in redimnet_m redimnet2_b4; do
  E=$R/exp_$M
  python train.py --config configs/$M/ft.yaml $SMALL --train_list $L/V3/train_split.txt --train_path $D \
         --save_path $E/ft --max_epoch 1 --crop_sec 1.015
  python train.py --config configs/$M/ft.yaml $SMALL --train_list $L/V3/train_split.txt --train_path $D \
         --save_path $E/ft --max_epoch 2 --crop_sec 1.015                    # resumes from last.ckpt
  python train.py --config configs/$M/lm.yaml $SMALL --train_list $L/V3/train_split.txt --train_path $D \
         --save_path $E/lm --max_epoch 1 --crop_sec 1.515 --initial_model $E/ft/model/last.ckpt
  python infer.py --mode score --model $E/lm/model/best.model --trials $P/public-test.txt --data_path $P/data \
         --out $E/sub/results.txt --zip --device cpu --score_norm center
  unzip -l $E/sub/results.zip
done
echo SMOKE TEST OK
