import matplotlib.pyplot as plt
import numpy as np
import json
import tikzplotlib


def main(kernel_name, dim):
    plt.clf()
    with open(f"data/{kernel_name}_{dim}D_p2m2m2l2lp_error.json", "r") as inf:
        data = json.loads(inf.read())

    orders = [dataset["order"] for dataset in data]
    errors = [dataset["error"] for dataset in data]
    plt.semilogy(orders, errors, "o-")

    kernel_disp_name = kernel_name.replace("Kernel", "")
    kernel_disp_name = kernel_disp_name.replace("let", "")
    kernel_id = kernel_disp_name.lower()

    plt.grid()
    # plt.title(f"M2M Accuracy {kernel_disp_name} {dim}D")

    plt.xlabel("Expansion Order $p$")
    plt.ylabel("Relative Error")

    # plt.legend(loc="best", prop={'size': 10})
    plt.tight_layout()

    tex_file_name = f"figures/error-p2m2m2l2l2p-{kernel_id}-{dim}d.tex"
    tikzplotlib.save(tex_file_name)
    import re
    with open(tex_file_name, "r") as f:
        lines = f.readlines()
    skip = False
    if dim == 1:
        xticks = [2,4,6,8,10,12,14,18,24]
    else:
        xticks = orders
    xticks = [str(xtick) for xtick in xticks]
    with open(tex_file_name, "w") as f:
        for line in lines:
            #if line.startswith("minor xtick"):
            #    skip=True
            if not skip and not (line.startswith('\\begin{tikzpicture}') or line.startswith("\\end{tikzpicture")):
                f.write(line)
            #if line.startswith("},"):
            #    skip=False


if __name__ == "__main__":
    #plt.rc("font", size=16)

    main("HeatKernel", 1)
    main("HeatKernel", 2)
    main("HeatKernel", 3)
    #main("StokesletKernel", 2)
    #main("StokesletKernel", 3)
