# imodels: Benchmarks + Experiments

Dieses Repository enthält zwei getrennte Bereiche:

- `benchmarks/`: reproduzierbare UCI-Benchmark-Pipeline (`imodels`, ExSTraCS, Merge-Plots)
- `experiments/`: explorative Skripte und Visualisierungs-Artefakte

## Projektstruktur

```text
imodels/
├─ benchmarks/
│  ├─ uci/
│  │  ├─ run_imodels_benchmark.py
│  │  ├─ run_exstracs_benchmark.py
│  │  └─ merge_benchmark_plots.py
│  └─ outputs/
│     ├─ imodels/
│     ├─ exstracs/
│     └─ merged/
├─ experiments/
│  └─ synthetic_rules/
│     ├─ compare_three_rule_models.py
│     ├─ plot_ground_truth_contours.py
│     └─ assets/
│        ├─ contours.ai
│        ├─ contours.pdf
│        └─ dt.dot
└─ data/
   └─ csv/
```

## Installation

```bash
python -m pip install -r requirements.txt
```

## Benchmark Quickstart

imodels-Benchmark:

```bash
python benchmarks/uci/run_imodels_benchmark.py --no-show
```

ExSTraCS-Benchmark:

```bash
python benchmarks/uci/run_exstracs_benchmark.py --no-show
```

Merged Plot aus beiden Plot-CSV-Dateien:

```bash
python benchmarks/uci/merge_benchmark_plots.py --no-show
```

## Standard-Outputpfade

- `benchmarks/uci/run_imodels_benchmark.py` -> `benchmarks/outputs/imodels/`
- `benchmarks/uci/run_exstracs_benchmark.py` -> `benchmarks/outputs/exstracs/`
- `benchmarks/uci/merge_benchmark_plots.py` -> `benchmarks/outputs/merged/`

Der Merge-Plot exportiert immer PNG + PDF.

## Paper-orientierte Plot-Eigenschaften

- Standardstil: `--plot-style dots` (Dot-Whisker)
- Fehlerbalken-Default: `--error-bars std`
- `combined`-Modus: vertikal gestapelte Subplots + Legende rechts
- Alternierende Hintergrundbänder pro Datensatzzeile
- Vergrößerte Schrift für Titel/Achsenlabels

## Legacy-Entrypoints (kompatibel)

Die bisherigen Dateinamen bleiben als Wrapper erhalten:

- `uciml.py` -> `benchmarks/uci/run_imodels_benchmark.py`
- `exstracs_benchmark.py` -> `benchmarks/uci/run_exstracs_benchmark.py`
- `plot_uciml_csvs.py` -> `benchmarks/uci/merge_benchmark_plots.py`

So funktionieren alte Befehle weiterhin, aber neue Befehle sollten die Pfade unter `benchmarks/uci/` verwenden.

