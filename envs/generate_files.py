import json
import os

m = {
  "linux-64": "Linux-x86_64",
  "linux-aarch64": "Linux-aarch64",
  "linux-ppc64le": "Linux-ppc64le",
  "osx-64": "Darwin-x86_64",
  "osx-arm64": "Darwin-arm64",
}

with open("file.json") as f:
    d = json.load(f)

lines = ["@EXPLICIT\n"]
for dep in d["dependencies"]:
    c = dep.split("::")
    pkg_name = c[1].replace("==", "-").replace("=", "-")
    lines.append(f"https://conda.anaconda.org/{c[0]}/{pkg_name}.tar.bz2\n")

subdir = os.environ["CONDA_SUBDIR"]
file_name = m[subdir]

with open(f"envs/{file_name}.txt", "w") as f:
    f.writelines(lines)
