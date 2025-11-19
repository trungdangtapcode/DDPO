#!/bin/bash


cd python-api
source venv/bin/activate
conda init && conda deactivate
python main.py