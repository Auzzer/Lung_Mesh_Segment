#!/usr/bin/env bash
set -euo pipefail

# MSE-style: ||u_sim - u_obs|| (no division by ||u_obs||)
python GCN_SMS/train_gcn_sms.py --omega-u 5 --omega-alpha 5 --num-epochs 100 \
  --unnormalized-data --log-file train_RelLoss.log

# Relative-loss: (||u_sim - u_obs|| / ||u_obs||) 

python GCN_SMS/train_gcn_sms.py --omega-u 5 --omega-alpha 5 --num-epochs 100 \
  --log-file train_MSE.log


