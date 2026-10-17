from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.titlesize": 8.5
})

# Load representative Grad-CAM samples
img_tumor = Image.open("results/TumorNet_gradcam_glioma_sample1.png")
img_pneumo = Image.open("results/PneumoNet_gradcam_PNEUMONIA_sample1.png")
img_gastro = Image.open("results/GastroNet_gradcam_Ulcer_sample1.png")

fig, axes = plt.subplots(3, 1, figsize=(7.1, 6.0), dpi=300)

axes[0].imshow(img_tumor)
axes[0].axis("off")
axes[0].set_title("(a) TumorNet: Glial Neoplasm (7x7 Feature Map Upsampled to 224x224 Overlay)", fontweight="bold", pad=4)

axes[1].imshow(img_pneumo)
axes[1].axis("off")
axes[1].set_title("(b) PneumoNet: Pediatric Parenchymal Consolidation with Airspace Infiltrate", fontweight="bold", pad=4)

axes[2].imshow(img_gastro)
axes[2].axis("off")
axes[2].set_title("(c) GastroNet: Mucosal Ulceration Localized to Ulcer Crater and Margin", fontweight="bold", pad=4)

plt.tight_layout()
plt.savefig("paper_figures/fig_gradcam_multipanel.png", dpi=300, bbox_inches="tight")
plt.savefig("paper_figures/fig_gradcam_multipanel.pdf", bbox_inches="tight")
plt.close()
print("Saved multi-panel Grad-CAM visualization -> paper_figures/fig_gradcam_multipanel.png")
