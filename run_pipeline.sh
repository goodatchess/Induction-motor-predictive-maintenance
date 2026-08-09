#!/usr/bin/env bash
set -e
python prepare_data.py
python train.py
python app.py
