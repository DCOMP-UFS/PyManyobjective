# PyManyObjetive Framework

## Setup

### Recommended Python version

- This project targets Python 3.9+. Python 3.10 or 3.11 is recommended.
- It is recommended to manage Python versions with [pyenv](https://github.com/pyenv/pyenv).
- The Python version currently used for this project is stored in the .python-version file at the repository root.
- To install and set the project's version:

```bash
pyenv install "$(cat .python-version)"
pyenv local "$(cat .python-version)"
python3 --version
```

### Virtual Python Environment
On the project's root create a virtual python environment.

```bash
python3 -m venv ./venv
```

Activate the environment

```bash
source ./venv/bin/activate
```

### Install dependencies
Install dependencies using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

## DVL runner

`run_dvl.py` now only runs the experiment, computes the indicators, and writes the CSV files used for analysis. It does not call `gnuplot` automatically.

Example for `DTLZ1` with 10 objectives, `k_problema = 10` (`n = 19`), and the sample size from Table 18:

```bash
python3 run_dvl.py --objectives 10 --evaluations 250 --sample-size 50 --problem-k 10
python3 run_dvl.py --objectives 10 --evaluations 500 --sample-size 112 --problem-k 10
python3 run_dvl.py --objectives 10 --evaluations 1000 --sample-size 200 --problem-k 10
python3 run_dvl.py --objectives 10 --evaluations 1500 --sample-size 200 --problem-k 10
python3 run_dvl.py --objectives 10 --evaluations 10000 --sample-size 300 --problem-k 10
```

Each run writes its datasets under `out/dvl/<run_name>/csv/`, including:

- `sample.csv`
- `estimated.csv`
- `final_front.csv`
- `reference_front.csv`
- `reference_points.csv`

## Notebook

The notebook [DVLFramework.ipynb](/home/gustavo/Documents/pesquisa/maop/PyManyobjective/DVLFramework.ipynb) runs the five `DTLZ1` cases with 10 objectives, consolidates the IGD/HV metrics, and saves a summary CSV in `out/dvl/dtlz1_m10_igd_summary.csv`.

## Optional GNUPlot debug

The framework generates standard CSV files that can be easily plotted using Gnuplot. For instance, to plot a 3D visualization of your 3-objective runs directly from the terminal, you can run:

```bash
gnuplot -persist -e "
set datafile separator ',';
set title 'DVL Front Debug';
set xlabel 'f1'; set ylabel 'f2'; set zlabel 'f3';
splot 'out/dvl/<RUN_DIR>/csv/final_front.csv' using 1:2:3 title 'Final Front' with points pt 7 ps 1.5, \\
      'out/dvl/<RUN_DIR>/csv/reference_front.csv' using 1:2:3 title 'Reference Front' with dots
"
```

Since the project no longer bundles `.plt` scripts or a Python gnuplot wrapper to avoid clutter, you can create your own standard gnuplot commands or scripts pointing to the generated `csv/` directories.
