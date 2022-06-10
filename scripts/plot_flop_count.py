import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import tikzplotlib
import numpy as np
import json


expected = {
  3: {
    "p2m": (3, 3),
    "p2l": (3, 2),
    "m2m": (4, 3),
    "m2l": (3, 2),
    "l2l": (4, 3),
    "l2p": (3, 3),
    "m2p": (3, 2),
  },
  2: {
    "p2m": (2, 2),
    "p2l": (2, 1),
    "m2m": (3, 2),
    "m2l": (2, 1),
    "l2l": (3, 2),
    "l2p": (2, 2),
    "m2p": (2, 1),
  },
}

def main(kernel_name, dim):
    plt.clf()
    #plt.figure(figsize=(6, 4.5))
    # plt.style.use("ggplot")
    with open(f"data/{kernel_name}_{dim}D_flop_count.json", "r") as inf:
        data = json.loads(inf.read())

    orders = np.array(data["order"])

    for op in data.keys():
        if op == "order":
            continue
        full_flops = data[op]["full_flops"]
        compressed_flops = data[op]["compressed_flops"]
        plt.loglog(orders, full_flops, "o-", label=f"{op.upper()} Full")
        plt.loglog(orders, compressed_flops, "o-", label=f"{op.upper()} Compressed")

        kernel_disp_name = kernel_name.replace("Kernel", "")
        kernel_disp_name = kernel_disp_name.replace("let", "")
        kernel_id = kernel_disp_name.lower()

        point = -1
        ref_flops = orders**(expected[dim][op][1])
        ref_bad_flops = orders**(expected[dim][op][0])

        if expected[dim][op][1] == 1:
            ref_label = "$p"
        else:
            ref_label = f"$p^{expected[dim][op][1]}"
        if expected[dim][op][0] == 1:
            ref_bad_label = "$p"
        else:
            ref_bad_label = f"$p^{expected[dim][op][0]}"
        if op == "m2l":
            ref_flops = ref_flops * np.log2(ref_flops)
            ref_bad_flops = ref_bad_flops * np.log2(ref_bad_flops)
            ref_label += "\log(p)"
            ref_bad_label += "\log(p)"
        ref_label += "$"
        ref_bad_label += "$"
        ref_flops = ref_flops * compressed_flops[point]/ref_flops[point]
        ref_bad_flops = ref_bad_flops * full_flops[point] /ref_bad_flops[point]
        plt.loglog(orders, ref_flops, "--", color="gray", label=ref_label)
        if expected[dim][op][0] != expected[dim][op][1]:
            plt.loglog(orders, ref_bad_flops, "--", color="red", label=ref_bad_label)

        plt.xlabel("Order $p$", fontsize=15)
        plt.ylabel("FLOP Count", fontsize=15)
        ax=plt.gca()
        #ax.set_xticks(orders)
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(mticker.ScalarFormatter())
        ax.xaxis.get_major_formatter().set_scientific(False)
        ax.xaxis.get_minor_formatter().set_scientific(False)
        # ax.xaxis.set_tick_params(which='major', pad=15)
        #ax.set_xticks([])
        xticks = [2, 4, 6, 8, 10, 20, 30, 40]
        ax.set_xticks([2, 4, 6, 8, 10, 20, 30, 40], minor=True)
        # if dim == 2:
        #     plt.ylim([2.5, 10**5])

        plt.legend(loc="best", prop={'size': 15})
        # plt.title(f"FLOP count {op.upper()} {kernel_disp_name} {dim}D", fontsize=15)

        plt.grid()

        plt.tight_layout()
        #plt.savefig(f"figures/flops-{kernel_id}-{op.upper()}-{dim}d.pdf")
        tex_file_name = f"figures/flops-{kernel_id}-{op.upper()}-{dim}d.tex"
        tikzplotlib.save(tex_file_name)
        import re
        with open(tex_file_name, "r") as f:
            lines = f.readlines()
        with open(tex_file_name, "w") as f:
            for line in lines:
                if line.startswith("ytick={"):
                    f.write("xtick={"+ ",".join([str(xtick) for xtick in xticks]) + "},\n")
                    f.write("xticklabels={" + ",\n".join([f"\(\displaystyle {{ {xtick} }}\)" for xtick in xticks]) + "},\n")
                if not line.startswith(("\\begin{tikzpicture}", "\\end{tikzpicture}")):
                    f.write(line)
        plt.clf()


if __name__ == "__main__":
    plt.rc("font", size=16)

    main("LaplaceKernel", 2)
    main("LaplaceKernel", 3)
