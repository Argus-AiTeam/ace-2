import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_ip_catalog_manifests_and_commands_are_synchronized():
    catalog = json.loads((ROOT / "ip" / "catalog.json").read_text())
    makefile = (ROOT / "Makefile").read_text()

    assert len(catalog["packages"]) == 9
    assert "ip-demo:" in makefile
    assert "ip-demo-all:" in makefile
    for package in catalog["packages"]:
        package_dir = ROOT / "ip" / package["name"]
        manifest = json.loads((package_dir / "manifest.json").read_text())
        assert manifest == package
        assert package["verification"]["command"] == (
            f"make ip-demo IP={package['name']}"
        )
        assert package["canonical_rtl"]
        assert package["verification"]["operators"]
