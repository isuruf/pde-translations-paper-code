import matplotlib.pyplot as plt
import numpy as np
import json
import tikzplotlib


def main(kernel_name, dim, op):
    plt.clf()
    with open(f"data/{kernel_name}_{dim}D_{op}_error.json", "r") as inf:
        data = json.loads(inf.read())

    for i, dataset in enumerate(data):
        order = dataset["order"]
        errors = dataset["error"]
        h = dataset["h"]
        plt.loglog(h, errors, "o-", label=f"p={order}")

    for i, dataset in enumerate(data):
        order = dataset["order"]
        errors = dataset["error"]
        if i in (0, len(data) - 1):
            index = 1 if op == "m2m" else 0
            plt.loglog(
                h,
                np.array(h) ** (order + 1) * errors[index] / h[index] ** (order + 1),
                "--",
                label=f"$h^{{ {order+1} }}$",
            )

    if op == "m2m":
        lbound = 3e-16
    else:
        lbound = 1e-17
    plt.gca().set_ylim(bottom=lbound)

    kernel_disp_name = kernel_name.replace("Kernel", "")
    kernel_disp_name = kernel_disp_name.replace("let", "")
    kernel_id = kernel_disp_name.lower()

    plt.grid()
    plt.xlabel("Convergence factor $h$")
    plt.ylabel("Relative Error")

    plt.legend(loc="best", prop={"size": 10})
    plt.tight_layout()
    # plt.show()

    tex_file_name = f"figures/error-p2{op}2p-heat-{dim}d.tex"
    tikzplotlib.save(tex_file_name)
    import re

    with open(tex_file_name, "r") as f:
        lines = f.readlines()
    skip = False
    if dim == 1:
        xticks = h  # [2,4,6,8,10,12,14,18,24]
    else:
        xticks = h
    xticks = [str(xtick) for xtick in xticks]
    with open(tex_file_name, "w") as f:
        for line in lines:
            # if line.startswith("minor xtick"):
            #    skip=True
            if not skip and not (
                line.startswith("\\begin{tikzpicture}")
                or line.startswith("\\end{tikzpicture")
            ):
                f.write(line)
            # if line.startswith("},"):
            #    skip=False


if __name__ == "__main__":
    # plt.rc("font", size=16)

    main("HeatKernel", 1, "m2l")
    main("HeatKernel", 1, "l2l")
    main("HeatKernel", 1, "m2m")
