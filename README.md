# Auto-train-xrd

This is a clean OpenI/MMP-compatible single-run XRD training project.

It is intentionally small: the repository validates the platform training skill
end to end by training a deterministic centroid classifier on XRD-like spectra.
The original multi-round agent optimization workflow is out of scope for this
repository shape; it can be rebuilt above this single-run primitive later.

## Local smoke test

```bash
python3 scripts/make_demo_dataset.py --output data/demo_xrd
python3 train.py --dataset data/demo_xrd --task_output outputs/local --epochs 1 --bs 16 --lr 0.001
```

Expected artifacts:

- `xrd_centroid_model.json`
- `metrics.json`
- `run_manifest.json`
- `samples.csv`

## Platform entry command

```bash
python train.py --task_output /result --epochs 1 --bs 16 --lr 0.001
```

The script accepts unknown platform-injected arguments, calls `c2net.context.prepare`
when available, resolves mounted datasets, extracts zip datasets, and writes
artifacts to the platform output path.
