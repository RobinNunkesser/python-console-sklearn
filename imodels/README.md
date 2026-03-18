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
│  ├─ multiplexer/
│  │  ├─ run_multiplexer_benchmark.py
│  │  ├─ merge_benchmark_plots.py
│  │  └─ multiplexer_plotting.py
│  └─ outputs/
│     ├─ imodels/
│     ├─ exstracs/
│     ├─ merged/
│     └─ multiplexer/
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

Der UCI-Runner nutzt denselben Shared-Plotrenderer wie die Merge-Skripte und unterstützt ebenfalls:

- `--plot-style dots|bars`
- `--plot-mode combined|separate|by_dataset` (Default: `combined`)
- PNG + PDF Export

ExSTraCS-Benchmark:

```bash
python benchmarks/uci/run_exstracs_benchmark.py --no-show
```

Hinweis: Dieser Schritt schreibt nur CSV-Dateien. Die eigentliche Darstellung läuft danach über
`benchmarks/uci/merge_benchmark_plots.py`, also bereits über den gemeinsamen Shared-Plotrenderer.

Merged Plot aus beiden Plot-CSV-Dateien:

```bash
python benchmarks/uci/merge_benchmark_plots.py --no-show
```

Multiplexer-Benchmark (lokale CSVs in `data/csv/Real`):

```bash
python benchmarks/multiplexer/run_multiplexer_benchmark.py --no-show
```

Multiplexer-Plot aus mehreren `multiplexer_plot_data.csv`-Dateien zusammenführen:

```bash
python benchmarks/multiplexer/merge_benchmark_plots.py \
  --input-csvs "run_a/multiplexer_plot_data.csv,run_b/multiplexer_plot_data.csv" \
  --no-show
```

Wichtige Unterschiede beim Multiplexer-Benchmark:

- kein Train/Test-Split (Training und Evaluation auf dem vollen Datensatz)
- Metrik `accuracy` statt `f1`
- gleicher Plot-Look wie beim UCI-Plot (`--plot-style dots|bars`, `--plot-mode combined|separate|by_dataset`, Default: `combined`)
- eigener Plot-Output: je nach `--plot-mode` entweder `multiplexer_combined.{png,pdf}` oder separate Dateien für `accuracy` und `model_size`
- eigener Merge-Plot-Output: je nach `--plot-mode` entweder `merged_multiplexer_combined.{png,pdf}` oder separate Dateien für `accuracy` und `model_size`

## Standard-Outputpfade

- `benchmarks/uci/run_imodels_benchmark.py` -> `benchmarks/outputs/imodels/`
- `benchmarks/uci/run_exstracs_benchmark.py` -> `benchmarks/outputs/exstracs/`
- `benchmarks/uci/merge_benchmark_plots.py` -> `benchmarks/outputs/merged/`
- `benchmarks/multiplexer/run_multiplexer_benchmark.py` -> `benchmarks/outputs/multiplexer/`
- `benchmarks/multiplexer/merge_benchmark_plots.py` -> `benchmarks/outputs/multiplexer/merged/`

Die Plot-Skripte exportieren PNG + PDF; bei `combined` entsteht eine Datei pro Benchmark, bei `separate` je eine Datei pro Metrik.
Optional mit `--plot-mode by_dataset`: eine große Figure mit einer Zeile pro Datensatz, gespeichert als `*_by_dataset.{png,pdf}`.

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
- `multiplexer_benchmark.py` -> `benchmarks/multiplexer/run_multiplexer_benchmark.py`
- `plot_multiplexer_csvs.py` -> `benchmarks/multiplexer/merge_benchmark_plots.py`

So funktionieren alte Befehle weiterhin, aber neue Befehle sollten die Pfade unter `benchmarks/uci/` verwenden.

