# GEVAI

This is the main repository providing the first non-cohesive framework for **General Explainable and Verifiable Artificial Intelligence**. Please look at [this](https://github.com/LogDS/GEVAI/wiki) wiki for more information on this project.

## Install Requirements
Ensure you have `/python/src` as your root directory:
```bash
export PYTHONPATH="${PYTHONPATH}:/python/src"
```

Install Poetry, if you haven't already:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Then:
```bash
pip install .
```

## Example
`examples/RunExperiment.py` contains example pipeline.

## GPU Support
### Ubuntu 22.04 - NVIDIA GPU
Must have CUDA version >= 12.3 installed