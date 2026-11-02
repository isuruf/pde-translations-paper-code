import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from ruamel.yaml import YAML
from conda.core.subdir_data import SubdirData
from conda.models.channel import Channel
from conda.models.match_spec import MatchSpec


platforms = {
    "linux-64": "Linux-x86_64",
    "linux-aarch64": "Linux-aarch64",
    "linux-ppc64le": "Linux-ppc64le",
    "osx-64": "Darwin-x86_64",
    "osx-arm64": "Darwin-arm64",
}

platform = sys.argv[1]
kind = sys.argv[2] if len(sys.argv) > 2 else "compute"
root = Path(__file__).resolve().parent
config_file = "plot.yml" if kind == "plot" else "conda.yml"
config = YAML(typ="safe").load((root / config_file).read_text())
env = os.environ.copy()
env["CONDA_SUBDIR"] = platform
if platform.startswith("linux-"):
    env["CONDA_OVERRIDE_GLIBC"] = "2.31"
else:
    env["CONDA_OVERRIDE_OSX"] = "11.0"

with tempfile.TemporaryDirectory() as work:
    result = subprocess.run([
        os.environ.get("CONDA_EXE", "conda"), "create", "--prefix", work + "/paper",
        "--override-channels", "--channel", "conda-forge", "--dry-run", "--json",
        *config["dependencies"],
    ], env=env, capture_output=True, text=True)

if result.returncode:
    sys.exit(result.stdout + result.stderr)
packages = json.loads(result.stdout)["actions"]["LINK"]
lines = ["@EXPLICIT\n"]
for package in packages:
    url = package.get("url")
    if url is None:
        channel = Channel(f"{package['base_url']}/{package['platform']}")
        spec = MatchSpec(f"{package['name']}={package['version']}={package['build_string']}")
        url = next(iter(SubdirData(channel).query(spec))).url
    lines.append(url + "\n")
suffix = "-plot" if kind == "plot" else ""
target = root / f"{platforms[platform]}{suffix}.txt"
target.write_text("".join(lines))
print(f"Generated {target.name}: {len(packages)} packages")
