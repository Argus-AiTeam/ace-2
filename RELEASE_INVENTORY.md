# Alpha 2 Release Inventory

## Included

- the exact 23-file certified RTL tree and generated LUT dependencies;
- release-local SDC and SKY130 synthesis/STA scripts;
- deterministic synthetic vectors and RTL testbenches;
- fixed-point operator references and vector generators;
- packed-image and two-token runtime utilities without model weights;
- Verilator runtime harness sources;
- architecture, status, limitations, changelog, and certification summaries;
- SHA-256 manifests.

## Excluded

- model weights and packed model images;
- datasets and private benchmark inputs;
- PDK libraries and proprietary toolchains;
- raw exactly-once PPA and runtime evidence packets;
- build outputs, waveforms, caches, and virtual environments;
- Argus backlog, journal, session, daemon, and operator state;
- credentials, keys, tokens, and machine-specific configuration;
- ongoing local-chat and U280 development introduced after Alpha 2 certification.

The package is prepared by whitelist. Structural source inclusion is not a
claim beyond the scope stated in [CERTIFICATION.md](CERTIFICATION.md).
