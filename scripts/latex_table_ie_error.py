import json

def main():
    with open("data/Biharmonic_IE_error.json") as f:
        data = json.load(f)
    
    orders = data["fmm_order"]
    full = data["full"]
    comp = data["compressed"]
    fft = data["compressed_fft"]

    text = "  \\begin{tabular}{lcccc} \\toprule\n"
    text += "    Order  & Taylor Series Error & Compressed Taylor Error & Compressed Taylor with FFT Error \\\\ \\midrule\n"
    for i, order in enumerate(orders):
        text += f"    {order} & {full[i]:.5e} & {comp[i]:.5e} & {fft[i]:.5e} \\\\\n"
    text = text[:-1]
    text += "  \\bottomrule\n"
    text += "  \\end{tabular}\n"
    with open("figures/ie_error_table.tex", "w") as f:
        f.write(text)

if __name__ == "__main__":
    main()
