import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
import numpy as np
import json
import tikzplotlib


def main(kernel_name, dim, R):
    plt.clf()
    fname = f"data/{kernel_name}_{dim}D_p2m2m2p_error_vs_k_R.json"
    with open(fname, "r") as inf:
        data = json.loads(inf.read())

    data = [dataset for dataset in data if dataset["order"] <= 14]

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
            'tab:brown']

    for idataset, dataset in enumerate(data):
        k = np.array(dataset["k"])
        order = dataset["order"]
        error = np.array(dataset["trunc_error_compressed"])
        error_uncompressed = np.array(dataset["trunc_error_uncompressed"])

        plt.loglog(k, error, "o-", color=colors[idataset])
        plt.loglog(k, error_uncompressed, "x--",
                color=colors[idataset])

    kernel_disp_name = kernel_name.replace("Kernel", "")
    kernel_disp_name = kernel_disp_name.replace("let", "")
    kernel_id = kernel_disp_name.lower()

    plt.grid()
    plt.xlabel(r"Wave number $\kappa$")
    if dim == 2:
        plt.ylabel(r"Error")

    plt.tight_layout()

    tex_file_name = f"figures/error-vs-k-{kernel_id}-{dim}d.tex"
    tikzplotlib.save(tex_file_name)
    import re
    with open(tex_file_name, "r") as f:
        lines = f.readlines()
    skip = False
    with open(tex_file_name, "w") as f:
        for line in lines:
            if line.startswith("minor ytick"):
                skip=True
            if not skip and not (line.startswith('\\begin{tikzpicture}') or line.startswith("\\end{tikzpicture")):
                f.write(line)
            if line.startswith("},"):
                skip=False

    if dim == 2:
        legend_file_name = f"figures/error-vs-k-{kernel_id}-legend.tex"
        with open(legend_file_name, "w") as f:
            f.write("\\begin{tikzpicture}[font=\\footnotesize]\n")
            for x, style, name, label in [
                    (0.6, "solid, mark=*", "compressed", "comp"),
                    (5.0, "dashed, mark=x", "uncompressed", "uncomp")]:
                f.write(f"\\draw[semithick, {style}, mark size=2.4pt, "
                        "mark options={solid}, mark indices={2}] "
                        f"plot coordinates {{({x},0.6) ({x + 0.45},0.6) "
                        f"({x + 0.9},0.6)}} node[right] {{{name}, "
                        f"$\\epsilon_{{\\mathrm{{trunc}}}}^{{\\mathrm{{{label}}}}}$}};\n")
            for i, dataset in enumerate(data):
                color = to_hex(colors[i]).lstrip("#")
                f.write(f"\\definecolor{{order{i}}}{{HTML}}{{{color}}}\n")
                f.write(f"\\draw[line width=1.2pt, order{i}] ({1.65*i},0) "
                        f"-- ({1.65*i + 0.45},0) node[right, black] "
                        f"{{$p={dataset['order']}$}};\n")
            f.write("\\end{tikzpicture}\n")


if __name__ == "__main__":
    main("HelmholtzKernel", 2, 0.01)
    main("HelmholtzKernel", 3, 0.01)
