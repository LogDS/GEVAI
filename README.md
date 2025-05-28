# <img src="extra/logo.svg" style="height:80px; width: auto;" alt="Logo: Credits to Oliver Robert Fox (2025)"/> GEVAI

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

## Benchmarking
### Compare **metrics** from different ad hoc pipelines
Once results are collected, specify folder with all desired outputs (`/results`) within: `python/src/GEVAI/benchmarking/aggregate_metrics.py`, and run the script. A file (`output.csv`) will be presented comparing: accuracy, F1 score, precision, recall.

### Compare running times for different GEVAI phases
Use `python/src/GEVAI/benchmarking/plot_results.py` to plot how the different results compare in terms of running times, this file should be generated within `results/` as `benchmark_(x)ex_post_explainers.csv`.

## GPU Support
### Ubuntu 22.04 - NVIDIA GPU
Must have CUDA version >= 12.3 installed