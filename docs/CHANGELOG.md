# Changelog

## Added

- Added PatchTST wrapper (requires supplying an official checkpoint) and a self-contained PyTorch implementation of TimesNet to serve as high-capacity transformer baselines. Legacy transformer placeholders have been removed.
- Added shared metric helpers (`weighted_absolute_percentage_error`, `weighted_percentage_error`) and surfaced WAPE/WPE throughout comparison and evaluation pipelines.
- Wired new baselines into `scripts/compare_baselines.py` and `scripts/evaluate_etnn.py` for immediate benchmarking.
