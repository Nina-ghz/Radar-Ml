# Data

Only the split manifest is version-controlled. Recordings are large and must stay local.

```text
data/
├── manifest/
│   ├── split_manifest.csv    # official 80/20 split, SHA256 4ccc4937…afc7 (tracked)
│   └── split_manifest.json   # split algorithm + checksum (tracked)
└── raw/                      # git-ignored
    └── recorded_dataset/     # 660 x (*_recorded.npy + .json), manifest copies, official scripts
```

## Setup

1. Download `recorded_dataset.zip` (~2.54 GB) from the
   [UCL RDR Radar ML - Lite record](https://rdr.ucl.ac.uk/articles/dataset/Radar_ML_-_Lite/33977674)
   and place it at the repository root.
2. Run `python scripts/setup_data.py --zip recorded_dataset.zip`
   (extracts to `data/raw/`, copies the manifest beside the arrays, verifies every SHA256).

Disk: ~4.4 GB extracted. The full official SNR expansion (57.66 GB) is **not needed**:
use `radarml.noise` to generate the requested SNRs on the fly, identical to the official arrays.

**Do not download `blind_test_participants.zip` before the Day 5 announcement.**
