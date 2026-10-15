# Blind-test inference

Kaggle accepts exactly `sample_id,label` (competition rules); `predict.py` writes that header by default.
The organizers' offline scorer (`third_party/radarml_lite/score_submission.py`) reads
`sample_id,predicted_label` instead: pass `--label-column predicted_label` when scoring a fake blind test.

The blind test is predicted **by a team member, in their own terminal**. Never upload its files to an
AI assistant, Google Drive or Colab, and never use it to train, tune or choose a model: the model is
chosen on the validation split first (docs/EXPERIMENT_LOG.md), then the blind test is predicted once.

Keep the extracted blind test **outside the repository** (for example `C:\Users\<you>\Desktop\blind_test`),
so that no tool reading the project can open it by accident.

## 1. Check the pipeline on a fake blind test (validation pulses only)

```powershell
# 2 validation pulses per configuration x 13 SNRs = 17 160 samples, official format, in data/fake_blind/ (git-ignored)
.\.venv\Scripts\python.exe scripts\make_fake_blind_test.py --pulses-per-file 2

.\.venv\Scripts\python.exe scripts\predict.py --run runs\<chosen run> --test-dir data\fake_blind --output data\fake_blind\predictions.csv --label-column predicted_label --device cpu
.\.venv\Scripts\python.exe scripts\validate_submission.py --predictions data\fake_blind\predictions.csv --template data\fake_blind\submission_template.csv --label-column predicted_label
.\.venv\Scripts\python.exe third_party\radarml_lite\score_submission.py --key data\fake_blind\answer_key.csv --submission data\fake_blind\predictions.csv --output data\fake_blind\fake_score.json
```

The official scorer must agree with the validation score of the same run on the same pulses.

## 2. Predict the real blind test

```powershell
.\.venv\Scripts\python.exe scripts\predict.py --run runs\<chosen run> --test-dir <blind test directory> --output predictions.csv --device cpu
.\.venv\Scripts\python.exe scripts\validate_submission.py --predictions predictions.csv --template <blind test directory>\submission_template.csv
```

`--test-dir` is the directory that contains `index.csv` and the `.npy` shards (`test/` in the extracted
`blind_test_participants.zip`); `--template` defaults to the `submission_template.csv` found in `--test-dir`
or its parent. Check the ZIP first: `Get-FileHash blind_test_participants.zip -Algorithm SHA256` must print
`7D13639EEA4AA936AD43F815D97B9D0A60CBAAF8FADE6A472306C75A70223EAB`. The script prints the SHA256 of `model.pt`, the time per pulse
(reading + preprocessing + STFT + model) and the number of predictions per class: copy them into the
submission notes. `predictions*.csv` is git-ignored.

Only the I/Q rows reach the model; `predict.py` reads `file`, `row` and `sample_id` from `index.csv`
to locate each sample and ignores every other column.
