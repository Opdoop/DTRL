#!/bin/bash
for lr in 2e-02 5e-02 2e-03 5e-03 1e-04 2e-04 5e-05 1e-05; do
  for epoch in 5 10 15; do
    sh ./run_vibert_32.sh $lr $epoch
  done
done