import matplotlib.pyplot as plt
import numpy as np
import json
import tikzplotlib


def main(kernel_name, dim):
    plt.clf()
    with open(f"data/{kernel_name}_{dim}D_p2m2m2p_error.json", "r") as inf:
        data = json.loads(inf.read())

    data = [dataset for dataset in data
            if (kernel_name != "StokesletKernel" or dataset["order"] != 10) and dataset["order"]<=14]
    orders = [dataset["order"] for dataset in data]

    for idataset, dataset in enumerate(data):
        h = np.array(dataset["r"])
        error = np.array(dataset["error"])
        order = dataset["order"]

        if kernel_name == "BiharmonicKernel" and order == 1:
            continue

        kwargs = {}

        kwargs["label"] = "$p=%d$" % order

        plt.loglog(h, error, "o-", **kwargs)

    for idataset, dataset in enumerate(data):
        h = np.array(dataset["r"])
        error = np.array(dataset["error"])
        order = dataset["order"]

        if order in [min(orders), max(orders)] and kernel_name == "HelmholtzKernel":
            ref_point = -1
            ref_h = np.linspace(h[0], h[-1])
            ref_err = error[ref_point]*(ref_h/ref_h[ref_point])**(order+1)
            color = "gray" if order == min(orders) else "red"
            plt.loglog(ref_h, ref_err, "--", color=color, label="$R^{%d}$" % (order+1))
            plt.ylim(bottom=1e-16)

    kernel_disp_name = kernel_name.replace("Kernel", "")
    kernel_disp_name = kernel_disp_name.replace("let", "")
    kernel_id = kernel_disp_name.lower()

    plt.grid()
    # plt.title(f"M2M Accuracy {kernel_disp_name} {dim}D")

    plt.xlabel("Geometric parameter $R$")
    if kernel_disp_name == "Laplace":
        plt.ylabel(r"$\epsilon_{\mathrm{rel}}$")

    plt.legend(loc="upper left", prop={'size': 10})
    plt.tight_layout()

    tex_file_name = f"figures/accuracy-{kernel_id}-{dim}d.tex"
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
                if kernel_disp_name == "Laplace":
                    if dim == 2:
                        points = ["2e-16", "5e-16"]
                    else:
                        points = ["1e-15", "2e-16"]
                    if line.startswith("ytick="):
                        f.write(f"ytick={{{points[0]},{points[1]}}},\n")
                        continue
                    if line.startswith("yticklabels="):
                        f.write(line)
                        f.write(f"\\(\\displaystyle {{{points[0]}}}\\),")
                        f.write("\n")
                        f.write(f"\\(\\displaystyle {{{points[1]}}}\\),")
                        f.write("\n")
                        continue

                f.write(line)
            if line.startswith("},"):
                skip=False


if __name__ == "__main__":
    #plt.rc("font", size=16)

    main("LaplaceKernel", 2)
    main("LaplaceKernel", 3)
    main("HelmholtzKernel", 2)
    main("HelmholtzKernel", 3)
    main("BiharmonicKernel", 2)
    main("BiharmonicKernel", 3)
    #main("StokesletKernel", 2)
    #main("StokesletKernel", 3)
