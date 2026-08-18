# Contributing

Contributions should preserve the distinction between structural RTL coverage
and numerically accepted support.

- Preserve the package classification in `ip/catalog.json`: a verified
  operator name is not automatically a standalone core.
- Run `make ip-validate` after changing the Open IP Library and add a focused
  proof mapping for new public packages.
- Do not claim dimensions, quantization formats, models, full-model behavior,
  or deployment targets beyond reproducible evidence.
- Add deterministic, redistribution-safe vectors and an independent oracle for
  numerical changes.
- Run `make demo` before submitting changes.
- Do not add model weights, datasets, credentials, private benchmark inputs,
  proprietary PDK/IP files, internal logs, or machine-specific paths.
- Do not alter generated evidence merely to force a pass.

Contributions are accepted under the Apache License 2.0. By submitting a
contribution, you agree that it may be distributed under that license.
