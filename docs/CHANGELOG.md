# Changelog

## Added

- Introduced transformer-based baselines (`TemporalTransformerBaseline`, `PretrainedTransformerBaseline`) alongside the existing ETNN options to provide SOTA transformer and foundation-model style comparisons.
- Wired new baselines into `scripts/compare_baselines.py` and `scripts/evaluate_etnn.py` for immediate benchmarking.
