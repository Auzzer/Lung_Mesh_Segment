#!/usr/bin/env bash
set -euo pipefail

# Run MSE-style (unnormalized data loss)
python GCN_SMS/train_gcn_sms.py --omega-u 0.5 --omega-alpha 0.5 --num-epochs 100 --unnormalized-data > train_MSE.log 2>&1

# Run relative loss (normalized by ||u_obs||)
python GCN_SMS/train_gcn_sms.py --omega-u 0.5 --omega-alpha 0.5 --num-epochs 100 > train_RelLoss.log 2>&1

# Optional: print tail of each log after completion
echo "Last lines of train_MSE.log:" && tail -n 20 train_MSE.log
echo "Last lines of train_RelLoss.log:" && tail -n 20 train_RelLoss.log
