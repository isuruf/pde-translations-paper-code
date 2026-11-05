import json

import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib


def main(op):
    with open(f"data/HeatKernel_1D_{op}_error.json") as f:
        data = json.load(f)

    plt.clf()
    h_arr = np.asarray(data[0]["h"])

    order_colors = {}
    for ds in data:
        line, = plt.loglog(ds["h"], ds["error"], "o-", label="$p=%d$" % ds["order"])
        order_colors[ds["order"]] = line.get_color()

    ref_orders = [5, data[-1]["order"]]
    for ref_p in ref_orders:
        ds = next(d for d in data if d["order"] == ref_p)
        errs = np.asarray(ds["error"])
        index = 2
        ref_h = h_arr[index:]
        ref = ref_h ** (ref_p + 1) * errs[index] / h_arr[index] ** (ref_p + 1)
        plt.loglog(
            ref_h,
            ref,
            "--",
            color=order_colors[ref_p],
            label="$h^{%d}$" % (ref_p + 1),
        )

    plt.gca().set_ylim(bottom=1e-17)
    plt.grid()
    plt.xlabel(r"$h$")
    plt.ylabel(r"$\varepsilon_{\mathrm{rel}}$")
    plt.legend(loc="upper left", prop={"size": 8})
    plt.tight_layout()

    tex_file_name = f"figures/error-heat-{op.upper()}-1d.pgf"
    tikzplotlib.save(tex_file_name)
    with open(tex_file_name) as f:
        lines = f.readlines()
    skip = False
    with open(tex_file_name, "w") as f:
        for line in lines:
            if line.startswith("\\begin{tikzpicture}") or line.startswith("\\end{tikzpicture}"):
                continue
            if line.startswith("minor ytick"):
                skip = True
            if not skip:
                line = line.replace("$\\mathdefault{", "\\(\\displaystyle {")
                line = line.replace("}$,", "}\\),")
                line = line.replace("}$\n", "}\\)\n")
                f.write(line)
            if line.startswith("},"):
                skip = False


if __name__ == "__main__":
    main("m2m")
    main("l2l")
    main("m2l")
