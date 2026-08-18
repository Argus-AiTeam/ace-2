# Immutable SKY130 PPA preflight

This package freezes the non-consuming preparation for a future ACE-2
base/candidate SKY130 comparison. It records the pinned ORFS image digest,
absolute in-container Yosys/OpenSTA/liberty paths, source commits and tree
objects, per-file Git and SHA-256 identities, the exact RTL/SDC/flow inputs,
the strict `ace2_shell` benchmark-interface sidecar/schema/validator identities,
the mapped-netlist transform, the preflight implementation hash, and exact
command argument vectors. Preflight first validates the sidecar against public
base `bc0ff4d89341646b948564cd59e2c67307bdea38`, requires the requested base to
equal that commit, and rejects candidate parameter, port, fused-QKV, SDC, or
flow-script drift. It also rejects a revision when its Yosys/OpenSTA scripts
read or write any RTL, SDC, or mapped-netlist path outside the frozen input
closure.
Python's `jsonschema` package is required; every generated manifest is checked
against the bundled Draft 2020-12 schema before it can be returned or written.

The package intentionally has no execution subcommand. `preflight` only reads
Git objects, inspects Docker metadata, and launches a read-only,
network-disabled shell to resolve paths after sourcing the image environment.
It does not mount the repository, invoke Yosys or OpenSTA, create the requested
namespace, or establish area/timing results.

```bash
python3 flow/immutable_ppa/benchmark_interface.py --repo "$PWD"

python3 flow/immutable_ppa/immutable_ppa.py preflight \
  --repo "$PWD" \
  --base-ref <full-base-commit> \
  --candidate-ref <full-candidate-commit> \
  --namespace "$PWD/evidence/immutable_ppa/comparison-<name>" \
  > /tmp/immutable-ppa-manifest.json

PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
  flow/immutable_ppa/test_benchmark_interface.py \
  flow/immutable_ppa/test_immutable_ppa.py
```

Manifest output is exclusive when `--output` is used. An existing output or
namespace is always an error; retries and overwrite are prohibited. Namespace
reservation is a separate, explicit operation for a future authorized runner:

```bash
python3 flow/immutable_ppa/immutable_ppa.py reserve-namespace \
  --namespace-root "$PWD/evidence/immutable_ppa" \
  --namespace "$PWD/evidence/immutable_ppa/comparison-<name>" \
  --manifest-sha256 <sha256> \
  --acknowledge-create
```

The namespace root must already exist, the namespace must be one direct child
matching the configured naming rule, and creation uses exclusive `mkdir`.
Preflight and self-tests never call this command on the official evidence root.
