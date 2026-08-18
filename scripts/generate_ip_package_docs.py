#!/usr/bin/env python3
"""Regenerate per-package manifests and concise READMEs from ip/catalog.json."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "ip" / "catalog.json"


def render_values(values: object) -> str:
    if isinstance(values, (dict, list)):
        return f"`{json.dumps(values, sort_keys=True)}`"
    return f"`{values}`"


def render_readme(package: dict) -> str:
    parameters = "\n".join(
        f"- `{name}`: {render_values(value)}"
        for name, value in package["supported_parameters"].items()
    )
    sources = "\n".join(f"- `{path}`" for path in package["canonical_rtl"])
    dependencies = "\n".join(
        f"- `{path}`" for path in package["dependencies"]["shared_paths"]
    )
    interfaces = "\n".join(f"- {item}" for item in package["interfaces"])
    limitations = "\n".join(
        f"- {item}" for item in package["known_limitations"]
    )
    return f"""# {package["title"]}

> Generated from [`ip/catalog.json`](../catalog.json). Edit the catalog and run
> `python3 scripts/generate_ip_package_docs.py` to keep metadata synchronized.

**Classification:** `{package["classification"]}`  
**Maturity:** {package["maturity"]}

{package["summary"]}

## Canonical RTL (referenced, not copied)

{sources}

## Shared dependencies / integration paths

{dependencies}

## Qwen2.5-0.5B-compatible parameters

{parameters}

## Interfaces

{interfaces}

## Verification

```sh
{package["verification"]["command"]}
```

Proof type: **{package["verification"]["proof_kind"]}**.
{package["verification"]["proof_note"]}
Results are written under `build/ip_library/{package["name"]}/`; PASS is only
reported after every mapped underlying proof passes.

## Known limitations

{limitations}

## License

This package references canonical repository sources and inherits
[{package["license"]["inheritance"]}](../../{package["license"]["source"]}).
"""


def main() -> None:
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    for package in catalog["packages"]:
        package_dir = ROOT / "ip" / package["name"]
        package_dir.mkdir(parents=True, exist_ok=True)
        (package_dir / "manifest.json").write_text(
            json.dumps(package, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (package_dir / "README.md").write_text(
            render_readme(package),
            encoding="utf-8",
        )
    print(f"ACE2_IP_PACKAGE_DOCS_WRITTEN packages={len(catalog['packages'])}")


if __name__ == "__main__":
    main()
