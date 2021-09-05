#!/bin/bash
timeout 29m python3 heur/run.py 128 4 || ( echo timeout && exit 0 )
