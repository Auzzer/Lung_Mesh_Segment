#!/usr/bin/env bash
set -euo pipefail

# Sweep all combinations of omega_u and omega_alpha.
OMEGA_VALUES=(0.1 0.5 5)
NUM_EPOCHS=100

for omega_u in "${OMEGA_VALUES[@]}"; do
  for omega_alpha in "${OMEGA_VALUES[@]}"; do
    echo "[MSE-style] omega_u=${omega_u}, omega_alpha=${omega_alpha}"
    python GCN_SMS/train_gcn_sms.py --omega-u "${omega_u}" --omega-alpha "${omega_alpha}" \
      --num-epochs "${NUM_EPOCHS}" --unnormalized-data \
      --log-file "train_u_${omega_u}_alpha_${omega_alpha}_mse.log"

    echo "[Relative-loss] omega_u=${omega_u}, omega_alpha=${omega_alpha}"
    python GCN_SMS/train_gcn_sms.py --omega-u "${omega_u}" --omega-alpha "${omega_alpha}" \
      --num-epochs "${NUM_EPOCHS}" \
      --log-file "train_u_${omega_u}_alpha_${omega_alpha}_rel.log"
  done
done
