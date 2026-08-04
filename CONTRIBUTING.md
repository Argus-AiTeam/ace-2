# Contributing

Contributions should preserve the distinction between structural RTL coverage
and numerically accepted support.

- Do not claim support beyond `layer_0.v_proj` without reproducible evidence.
- Add deterministic, redistribution-safe vectors and an independent oracle for
  numerical changes.
- Run `make demo` before submitting changes.
- Do not add model weights, datasets, credentials, private benchmark inputs,
  proprietary PDK/IP files, internal logs, or machine-specific paths.
- Do not alter generated evidence merely to force a pass.

Contributions are accepted under the Apache License 2.0. By submitting a
contribution, you agree that it may be distributed under that license.
