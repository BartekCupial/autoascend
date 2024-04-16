#!/bin/bash
python3 -m heur.run_formatting_shards --input_dir=text_data --output_dir=processed_data --num_workers=8 --seq=128 --nsamples=100
