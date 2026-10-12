"""Fixed acquisition constants and challenge definitions (RadarML Lite)."""

# Exact label strings required by the scorer. Sorted order == scorer's class order.
CLASSES = ("barker", "cw", "fmcw", "fsk", "hyp", "nlfm", "quad")
CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASSES)}

SAMPLE_RATE_HZ = 120_000_000          # complex samples per second
N_COMPLEX = 16_800                    # complex samples per pulse (140 us)
N_INTERLEAVED = 2 * N_COMPLEX         # int16 values per pulse: I0, Q0, I1, Q1, ...

# The 13 requested SNRs used by the official expansion and the blind test.
REQUESTED_SNRS_DB = (30, 25, 20, 15, 10, 5, 0, -5, -10, -15, -20, -25, -30)
# Tie-break: mean balanced accuracy over the three lowest requested SNRs.
TIEBREAK_SNRS_DB = (-20, -25, -30)

# Seed used by the organizers for the split and for the official SNR expansion.
OFFICIAL_SEED = 42
