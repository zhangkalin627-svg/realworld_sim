#!/bin/bash
cd "$(dirname "$0")"
if [ -x .venv/bin/python ]; then .venv/bin/python app.py; else python3 app.py; fi
