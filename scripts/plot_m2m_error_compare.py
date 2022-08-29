import matplotlib.pyplot as plt
import numpy as np
import json
import tikzplotlib


def main(kernel_name, dim):
    plt.clf()
    with open(f"data/{kernel_name}_{dim}D_p2m2m2p_error_no_assumption.json", "r") as inf:
        data = json.loads(inf.read())

    data = [dataset for dataset in data
            if (kernel_name != "StokesletKernel" or dataset["order"] != 10) and dataset["order"]<=14]
    orders = [dataset["order"] for dataset in data]

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
            'tab:brown']

    for idataset, dataset in enumerate(data):
        h = np.array(dataset["r"])
        order = dataset["order"]
        #if order < 12:
        #    continue
        error = np.array(dataset["error"])
        error_uncompressed = np.array(dataset["error_uncompressed"])
        error_compressed = np.array(dataset["error_compressed"])

        if kernel_name == "BiharmonicKernel" and order == 1:
            continue

        plt.loglog(h, error, "o-", label=f"$\epsilon_{{rel}}$", color=colors[idataset])
        plt.loglog(h, error_uncompressed, "x--",
                label=f"$\epsilon_{{trunc}} (p={order})$", color=colors[idataset])
    
    kernel_disp_name = kernel_name.replace("Kernel", "")
    kernel_disp_name = kernel_disp_name.replace("let", "")
    kernel_id = kernel_disp_name.lower()

    plt.grid()
    # plt.title(f"M2M Accuracy {kernel_disp_name} {dim}D")

    plt.xlabel("Geometric parameter $R$")
    if dim == 2:
        plt.ylabel(r"Error")

    plt.legend(loc="lower right", prop={'size': 6}, ncol=2)
    plt.tight_layout()
    #plt.show()

    tex_file_name = f"figures/error-compare-{kernel_id}-{dim}d.tex"
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
    #plt.rc("font", size=16)

    #main("LaplaceKernel", 2)
    #main("LaplaceKernel", 3)
    main("HelmholtzKernel", 2)
    main("HelmholtzKernel", 3)
    #main("BiharmonicKernel", 2)
    #main("BiharmonicKernel", 3)
    #main("StokesletKernel", 2)
    #main("StokesletKernel", 3)
