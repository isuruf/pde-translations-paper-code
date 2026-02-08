import matplotlib.pyplot as plt
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
        error = np.array(dataset["rel_error"])
        error_uncompressed = np.array(dataset["trunc_error_uncompressed"])

        plt.loglog(k, error, "o-", label=r"$\epsilon_{\mathrm{rel}}$",
                   color=colors[idataset])
        plt.loglog(k, error_uncompressed, "x--",
                label=fr"$\epsilon_{{\mathrm{{trunc}}}} (p={order})$",
                color=colors[idataset])

    kernel_disp_name = kernel_name.replace("Kernel", "")
    kernel_disp_name = kernel_disp_name.replace("let", "")
    kernel_id = kernel_disp_name.lower()

    plt.grid()
    plt.xlabel(r"Wave number $\kappa$")
    if dim == 2:
        plt.ylabel(r"Error")

    plt.legend(loc="upper left", prop={'size': 6}, ncol=2)
    plt.tight_layout()

    tex_file_name = f"figures/error-vs-k-{kernel_id}-{dim}d-R.tex"
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
            if line.startswith("legend style"):
                f.write("nodes={scale=0.7, transform shape},\n")
            if line.startswith("},"):
                skip=False


if __name__ == "__main__":
    main("HelmholtzKernel", 2, 0.01)
    main("HelmholtzKernel", 3, 0.01)
