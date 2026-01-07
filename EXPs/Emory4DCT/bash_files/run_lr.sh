#!/usr/bin/env bash
set -euo pipefail

python GCN_SMS/train_gcn_sms.py --lr 0.5 \
  --log-file train_lr_0.5.log


python GCN_SMS/train_gcn_sms.py --lr 1 \
  --log-file train_lr_1.log

  python GCN_SMS/train_gcn_sms.py --lr 0.05 \
  --log-file train_lr_0.05.log