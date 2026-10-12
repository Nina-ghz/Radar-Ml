# Submission checklist

## Model freeze — Day 4, 17:00 (Europe/London)

- [ ] Final model selected **on validation only**; validation report saved (per-SNR balanced accuracy + confusion matrices)
- [ ] All code committed; freeze tagged: `git tag freeze-track-b && git push --tags`
- [ ] Weights + config + preprocessing settings saved under `models/final/`
- [ ] `python scripts/hash_artifacts.py models/final/* --output freeze_manifest.json`
- [ ] Inference command documented and tested end-to-end on validation data
- [ ] Dependency versions recorded (`pip freeze` is captured by `hash_artifacts.py`)
- [ ] Frozen artifacts sent to the organizers (destination announced at kick-off)

## Blind inference — Day 5, 09:00 → Day 6, 17:00

- [ ] Download `blind_test_participants.zip` **only after the announcement**, into a separate folder
- [ ] Verify SHA256 `7d13639eea4aa936ad43f815d97b9d0a60cbaaf8fade6a472306c75a70223eab`
- [ ] Run the **frozen** pipeline in inference mode — no retraining, no tuning, no test statistics
- [ ] `python scripts/validate_submission.py --predictions predictions.csv --template submission_template.csv`
- [ ] Check artifact hashes still match `freeze_manifest.json`

## Final package — Day 6, 17:00

- [ ] `predictions.csv` (`sample_id,predicted_label`, every ID once, 7 permitted labels)
- [ ] Frozen pipeline identifier (commit / tag) + artifact hashes
- [ ] Method summary: track B, representation, model, training subset & SNRs, seeds
- [ ] Validation scores: mean balanced accuracy, per-SNR curve, confusion matrices
- [ ] Hardware, training time, **inference time per pulse incl. STFT**, model size
- [ ] Attribution of reused code/papers + **disclosure of AI assistance**
- [ ] Limitations; any post-freeze change disclosed (and not presented as frozen result)
- [ ] Team and track identified in the submission folder/message (not in the CSV)
