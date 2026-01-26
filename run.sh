#!/bin/bash
set -e
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python preprocess_data.py
python train.py
