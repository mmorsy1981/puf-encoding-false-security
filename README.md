# Encoding-Induced False Security in ML-Resistant PUFs — experiment code

Experiment scripts for the article

> M. Farag, "Encoding-Induced False Security in ML-Resistant PUFs: What Public and Secret Challenge Layers Are Worth to an Informed Attacker," manuscript under review, 2026.

The experiments price encoding layers placed in front of Arbiter PUFs (APUFs) with attackers of graded knowledge:

- uninformed learners on the encoded challenge (T0);
- an informed attacker that inverts every public layer (T1);
- a structured attacker that knows the architecture but not its secret parts (T2).

The main T2 attack fits a secret byte substitution table jointly with the APUF weights. Its cost is compared with a known-table control that runs the same learner with the true table.

## Layout

| Path | Contents |
|---|---|
| `experiments/encoding-induced-false-security-pufs-*.py` | One script per experiment. The pre-specified predictions and decision rules are in each script's docstring. Each script writes its per-run results to a CSV file of the same name. |
| `experiments/encoding-induced-false-security-pufs-{gate1,gate1c-finegrid,e8fpga,e9scale-nscale,e9scale-stress,e15scale30,e18mixladder}.csv` | Results of earlier experiments that later scripts read as inputs: E21 re-analyses E8, E9 and E15; E22 continues E18. |
| `article/stats_cluster.py` | Instance-level bootstrap used by the E21 re-analysis (`e21sweep`). |
| `experiments/e22-mixscale-SPEC.md` | Pre-specification of E22. |
| `experiments/requirements-clean.txt` | Pinned Python environment. |
| `data/fpga/README.md` | How to obtain the measured FPGA data, with their SHA-256 checksums. |

## Experiment map

| Article | Scripts (prefix `encoding-induced-false-security-pufs-` omitted) |
|---|---|
| Public layers, T0/T1 (E5) | `gate1*`, `full1`–`full3` (these also hold repeats for the published designs), `perm`, `pilot-e5*`, `pilot-xgbt0*` |
| Metric validation | `rq3`, `rq3b`, `rq3c-advnull`, `pilot-advnull-24` |
| Secret S-box (E7, E7b) | `pilot-e7sboxlearn`, `pilot-e7gpu`, `pilot-e7b` |
| FPGA replay (E8) and measured XOR (E14) | `e8fpga`, `e14xorfpga` |
| Length, noise, delay model (E9, E15, E20, E21) | `e9scale`, `e15scale30`, `e20noisectl`, `e21sweep` |
| Boundary and learners (E11–E13, E16–E19, E22) | `e11gen`, `e12boundary`, `e13learners`, `e16calperm`, `e17calmore`, `e18mixladder`, `e19xorsim`, `e22mixscale` |
| ROC-SPUF (E2, E3, E10) | `pilot-e2fals`, `pilot-e3grid*`, `pilot-e3npilot`, `roc3*`, `pilot-roc3x`, `pilot-r5agree`, `mlp25`, `noisy-ceiling`, `e10rocnoise` |
| DATE-2018 (E4) | `pilot-e4*` |
| LFSR-APUF | `lfsr-*`, `pilot-lfsr-*` |
| OB-PUF, RSO, CO-PUF (E1, E6) | `gate2`, `t2ext*`, `pilot-ob*`, `pilot-e6*`, `pilot-e1m0`, `rso-parity`, `copuf` |
| Exact-specification replications | `exactA`, `exactB` |

Some scripts reuse code from others by executing them (for example, most E11–E22 scripts build on `e11gen` or `e13learners`). Keep the directory together.

## Environment

The experiments ran under Python 3.12 with pypuf 2.2.0 and PyTorch 2.8. Build the pinned environment as follows:

```
conda create -n puf-clean python=3.12.13 -y
conda run -n puf-clean python -m pip install -r experiments/requirements-clean.txt --extra-index-url https://download.pytorch.org/whl/cu129
conda run -n puf-clean python -m pip install pypuf==2.2.0 --no-deps
```

GPU runs used an RTX 4070 laptop GPU with TensorFloat-32 arithmetic disabled. CPU reruns with the same seeds agree within 0.001 accuracy.

## Running

Run every script from the repository root, for example:

```
python experiments/encoding-induced-false-security-pufs-e9scale.py
```

Scripts that use measured data (E8, E14) need the files described in `data/fpga/README.md`.

## Data

The measured APUF and XOR-APUF data are the public pypuf data collection (Zenodo, DOI 10.5281/zenodo.5221305). They are not redistributed here.

## License

MIT; see `LICENSE`.
