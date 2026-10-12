# Vendored organizer scripts

Copied **unmodified** from [UCLRadarGroup/Radar_ML_lite](https://github.com/UCLRadarGroup/Radar_ML_lite)
at commit `490cb87afd227a9ac4125cc710204af37be2b778` (folder `hackathon/`). GPL-3.0, see `LICENSE`.

| File | Use |
| --- | --- |
| `make_dataset_hackathon.py` | Official `degrade()` (SNR noise), imported by `src/radarml/noise.py` |
| `compact_dataset.py` | Official expansion CLI (used in tests as ground truth) |
| `participant_loader.py` | Loader for the blind-test shards (Day 5) |
| `score_submission.py` | Organizer scorer (reference for `src/radarml/metrics.py`) |

Do not edit these files: re-sync from upstream if the organizers publish changes.
