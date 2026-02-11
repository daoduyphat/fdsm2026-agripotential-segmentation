#!/bin/bash
set -e
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python train_new.py #chinh ten tuy theo model minh train