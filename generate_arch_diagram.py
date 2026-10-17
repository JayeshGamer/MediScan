import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_dualpathnet_architecture():
    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 42)
    ax.axis('off')

    # Styles
    box_blue = dict(boxstyle="square,pad=0.3", fc="#e1f5fe", ec="#0288d1", lw=1.5)
    box_green = dict(boxstyle="square,pad=0.3", fc="#e8f5e9", ec="#388e3c", lw=1.5)
    box_amber = dict(boxstyle="square,pad=0.3", fc="#fff8e1", ec="#f57f17", lw=1.5)
    box_purple = dict(boxstyle="square,pad=0.3", fc="#f3e5f5", ec="#7b1fa2", lw=1.5)
    box_red = dict(boxstyle="square,pad=0.3", fc="#ffebee", ec="#d32f2f", lw=1.5)
    box_teal = dict(boxstyle="square,pad=0.3", fc="#e0f2f1", ec="#00796b", lw=1.5)
    box_gray = dict(boxstyle="square,pad=0.3", fc="#f5f5f5", ec="#616161", lw=1.5)

    arrow = dict(arrowstyle="->", lw=1.5, color="#37474f")
    arrow_red = dict(arrowstyle="->", lw=1.5, color="#d32f2f", ls="--")

    # 1. Input Scan
    ax.text(6, 21, "Input Scan\n$3 \\times 224 \\times 224$\n(MRI / CXR / Endo)", ha="center", va="center", bbox=box_gray, fontsize=7.5, weight="bold")

    # 2. Stem
    ax.annotate("", xy=(13.5, 21), xytext=(10.5, 21), arrowprops=arrow)
    ax.text(18, 21, "Stem Block\nConv $7\\times 7$, s2\nBN + SiLU\nMaxPool $3\\times 3$, s2\n($64 \\times 56 \\times 56$)", ha="center", va="center", bbox=box_blue, fontsize=7)

    # Fork arrows
    ax.annotate("", xy=(26, 29), xytext=(22.5, 23), arrowprops=arrow)
    ax.annotate("", xy=(26, 13), xytext=(22.5, 19), arrowprops=arrow)

    # 3. Path A & Path B
    ax.text(32.5, 29, "Path A ($3 \\times 3$)\n3x Conv Blocks\n$64 \\to 128 \\to 256$\nSpatial Dropout (0.08)", ha="center", va="center", bbox=box_green, fontsize=7)
    ax.text(32.5, 13, "Path B ($5 \\times 5$)\n3x Conv Blocks\n$64 \\to 128 \\to 256$\nSpatial Dropout (0.08)", ha="center", va="center", bbox=box_green, fontsize=7)

    # Join arrows
    ax.annotate("", xy=(42.5, 23), xytext=(39, 29), arrowprops=arrow)
    ax.annotate("", xy=(42.5, 19), xytext=(39, 13), arrowprops=arrow)

    # 4. Concat + 1x1 Fusion
    ax.text(47, 21, "Concat & Fuse\n$1\\times 1$ Conv, BN, SiLU\n($256 \\times 28 \\times 28$)", ha="center", va="center", bbox=box_amber, fontsize=7)

    # CBAM
    ax.annotate("", xy=(54.5, 21), xytext=(51.5, 21), arrowprops=arrow)
    ax.text(59, 21, "CBAM Attention\nChannel Attention\n(GAP+GMP $\\to$ MLP)\n+\nSpatial ($7\\times 7$ Conv)", ha="center", va="center", bbox=box_purple, fontsize=7)

    # Refinement Blocks
    ax.annotate("", xy=(66.5, 21), xytext=(63.5, 21), arrowprops=arrow)
    ax.text(72, 21, "3x Residual Refine\nBlock 1: $256 \\to 512$\nBlock 2: $512 \\to 512$\nBlock 3: $512 \\to 512$\n($512 \\times 7 \\times 7$)", ha="center", va="center", bbox=box_teal, fontsize=7)

    # Grad-CAM Hook callout from refine[2] (Block 3)
    ax.annotate("", xy=(72, 33), xytext=(72, 27), arrowprops=arrow_red)
    ax.text(72, 36.5, "Grad-CAM Hook\nPenultimate layer: refine[2]\n($512 \\times 7 \\times 7$ $\\to$ $224 \\times 224$)", ha="center", va="center", bbox=box_red, fontsize=7, weight="bold")

    # Dual Pooling
    ax.annotate("", xy=(80.5, 21), xytext=(77.5, 21), arrowprops=arrow)
    ax.text(84.5, 21, "Dual Pooling\n[GAP ; GMP]\n$1024$-d vector", ha="center", va="center", bbox=box_amber, fontsize=7)

    # Classifier Head
    ax.annotate("", xy=(91.5, 21), xytext=(88.5, 21), arrowprops=arrow)
    ax.text(95.5, 21, "MLP Head\nLinear $1024 \\to 256$\nSiLU + Dropout(0.35)\nLinear $256 \\to C$\nDropout(0.20)\n+ 4-Pass TTA", ha="center", va="center", bbox=box_blue, fontsize=7)

    # Bottom labels / Triage output
    ax.text(95.5, 7, "Output Logits\nCalibrated $\\tau$\nRisk Tiers (Low/Mod/High)", ha="center", va="center", bbox=box_gray, fontsize=7)
    ax.annotate("", xy=(95.5, 11), xytext=(95.5, 15), arrowprops=arrow)

    plt.tight_layout()
    plt.savefig("paper_figures/fig_dualpathnet_architecture.png", dpi=300, bbox_inches='tight')
    plt.savefig("paper_figures/fig_dualpathnet_architecture.pdf", bbox_inches='tight')
    plt.close()
    print("DualPathNet architecture figure created successfully.")

if __name__ == "__main__":
    draw_dualpathnet_architecture()
