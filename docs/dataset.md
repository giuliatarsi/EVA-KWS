# Local dataset

Default directory: `<repository>/data_v2/` (excluded from Git).

```
data_v2/
  training/{eva,unknown,noise}/...wav
  test/{eva,unknown,noise}/...wav
```

Optional validation/ and test_finale/ use the same class subdirectories. Filenames must preserve Pxx speaker or Axx noise-environment identifiers. Data must be PCM16 mono at 16kHz, 1100ms within configured tolerance. Group or duplicate-audio overlap across original subsets is rejected.

Three-class training preserves labels; binary merges unknown/noise into non_target in memory. Cross-validation pools the historical training/test subset and splits by group. Final training pools development data for grouped training/validation and excludes test_finale. Read the configuration before running either entry point.

Edge Impulse supports label assignment and cropping. Only complete activation phrases are positive. Isolated EVA/aiutami and confusing expressions are negatives; inspect crops to avoid accidental phrase boundaries across utterances. Keep all recordings of a participant grouped, including the new hard negatives. New dataset versions need matched evaluation when comparing models.
