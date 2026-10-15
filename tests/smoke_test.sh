#!/usr/bin/env bash
# End-to-end smoke test on synthetic data (CPU, a few minutes): train 2 epochs, resume, infer.
set -euo pipefail
cd "$(dirname "$0")/.."
R=${1:-/tmp/redimnet_smoke}
rm -rf "$R"; python tests/make_fake_data.py "$R"
K=$R/kaggle/input/datasets; T=$K/hongvitchugo/vi-sred26-train; P=$K/hongvitchugo/vi-sred26-publictest
COMMON="--train_list $T/train_split.txt --train_path $T/Vi-SRED26-Train/data --eval_list $T/val_trials.txt --eval_path $T/Vi-SRED26-Train/data
        --musan_path $K/nhattruongdev/musan-noise --rir_path $K/nhattruongdev/rirs-noises --save_path $R/exp
        --device cpu --batch_size 6 --n_cpu 2 --crop_sec 1,1.5 --cohort_size 20 --log_interval 100"
python train.py $COMMON --max_epoch 1
python train.py $COMMON --max_epoch 2          # resumes from last.ckpt
python infer.py --mode score --model $R/exp/model/best.model --trials $P/public-test.txt --data_path $P/data \
                --out $R/sub/results.txt --zip --device cpu --score_norm center
unzip -l $R/sub/results.zip
echo SMOKE TEST OK
