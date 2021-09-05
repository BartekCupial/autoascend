#!/bin/bash
timeout 29m python3 heur/run.py 128 128 || ( echo timeout && exit 0 )
