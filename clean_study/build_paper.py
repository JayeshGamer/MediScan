
# -*- coding: utf-8 -*-
"""Assemble the IEEE two-column DOCX."""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
from paper_kit import *
from paper_kit import _fmt
import content_refs as C

ROOT = C.ROOT
FIG = os.path.join(ROOT, "paper_figures")
RES = os.path.join(ROOT, "clean_study", "results")
OUT_DOCX = os.path.join(ROOT, "MediScan_IEEE_Audit_Paper_Final.docx")

def pct(x, d=2):
    return f"{100*x:.{d}f}%"

def ci(lo, hi, d=2):
    return f"[{100*lo:.{d}f}%, {100*hi:.{d}f}%]"

def load(name):
    p = os.path.join(RES, name)
    return json.load(open(p)) if os.path.exists(p) else None

def wide_table(doc, *args, **kw):
    start_1col(doc)
    t = table(doc, *args, width_in=7.0, **kw)
    start_2col(doc)
    return t

def title_block(doc):
    p = doc.add_paragraph(); _fmt(p, 0, 0, 0, "center")
    r = p.add_run("Auditing Partition Leakage and Shortcut Learning in Medical Imaging Benchmarks: A Systematic Multi-Modal Evaluation and Clinical Tele-Triage Case Study")
    r.font.size, r.font.name, r.font.bold = Pt(24), BODY_FONT, False
    p = doc.add_paragraph(); _fmt(p, 0, 2, 0, "center")
    r = p.add_run("Daksh Gurjar, Aditya Yashovardhan, and Jayesh Mandloi")
    r.font.size, r.font.name = Pt(11), BODY_FONT
    p = doc.add_paragraph(); _fmt(p, 0, 0, 4, "center")
    r = p.add_run("Department of Computer Engineering, Mukesh Patel School of Technology Management & Engineering (MPSTME), SVKM\u2019s NMIMS, Shirpur 425405, Maharashtra, India")
    r.font.size, r.font.name, r.font.italic = Pt(9), BODY_FONT, True
    p = doc.add_paragraph(); _fmt(p, 0, 0, 4, "center")
    r = p.add_run("Email: {daksh.gurjar, aditya.yashovardhan, jayesh.mandloi}@nmims.edu")
    r.font.size, r.font.name = Pt(9), BODY_FONT

def abstract(doc):
    p = doc.add_paragraph(); _fmt(p, 0.0, 0, 2, "justify")
    r = p.add_run("Abstract\u2014"); r.font.size, r.font.bold, r.font.name = Pt(9), True, BODY_FONT
    txt = ("Public medical-imaging benchmarks underpin hundreds of computer-aided-diagnosis papers, yet standard evaluation "
    "protocols frequently overlook cross-split contamination and anatomical shortcut learning. We audit three widely used clinical "
    "benchmarks\u2014brain MRI for intracranial neoplasms, pediatric chest radiography (CXR) for pneumonia, and white-light endoscopy for "
    "mucosal inflammation\u2014with a compact 14.91M-parameter multi-scale baseline (DualPathNet). A two-level audit combining raw-byte "
    "SHA-256 and pixel-exact matching with a content-cropped 256-bit perceptual descriptor shows that a naive 64-bit dHash criterion is "
    "dominated by uniform backgrounds and collimator borders, flagging up to 74.1% of brain-MRI test scans and 95.9% of CXR test scans as "
    "near-duplicates; the corrected descriptor instead flags 11.8% of brain-MRI test scans, 0.0% of CXR test scans, and 0.4% of endoscopy "
    "frames as cross-split near-duplicates (98 brain-MRI test scans are pixel-exact copies of training scans). Retraining on de-duplicated "
    "splits yields a modest, honest accuracy change, and freezing the original model on the clean brain-MRI test set (N = 1,412) gives "
    "94.48% accuracy (vs. 95.00% on the contaminated 1,600-scan split). Same-organ negative controls expose an anatomical shortcut: a "
    "standard GastroNet misclassifies normal esophageal Z-line as lesion with 3.76% specificity; adding site-matched negatives on a "
    "verifiably disjoint Z-line half restores held-out specificity to 81.42% while keeping 94.07% lesion sensitivity. Zero-shot external "
    "evaluation (RSNA Pneumonia 84.60%, BraTS 2020 87.25%, Kvasir-Capsule 83.33%) shows substantial but expected out-of-distribution "
    "decline. We release the audit toolkit and clean split manifests.")
    rich(p, txt, size=9)

def index_terms(doc):
    p = doc.add_paragraph(); _fmt(p, 0.0, 0, 6, "justify")
    r = p.add_run("Index Terms\u2014"); r.font.size, r.font.bold, r.font.name = Pt(9), True, BODY_FONT
    rich(p, "Computer-aided diagnosis, multiscale convolutional neural networks, explainable AI (XAI), Grad-CAM, pediatric pneumonia, brain neoplasms, mucosal lesions, data leakage, shortcut learning, medical triage.", size=9)

def start_2col(doc):
    return new_section(doc, 2)

def start_1col(doc):
    return new_section(doc, 1)

def build():
    doc = Document()
    setup_page(doc)
    title_block(doc)
    abstract(doc)
    index_terms(doc)
    start_2col(doc)

    # ================= I. INTRODUCTION =================
    h1(doc, "Introduction", "I")
    body(doc, "Diagnostic imaging\u2014magnetic resonance imaging (MRI), planar chest radiography (CXR), and white-light flexible "
         "endoscopy\u2014constitutes the bedrock of modern diagnosis, oncology staging, and acute emergency triage [1]\u2013[3]. The volume of "
         "acquired scans vastly outpaces specialist interpretive capacity [4]\u2013[6], a deficit that is acute in rural and tier-2/3 Indian "
         "districts where the patient-to-radiologist ratio remains far below WHO norms [1], [5], [7]. Under such constraints, timely detection "
         "of pediatric bacterial pneumonia [1], [8], aggressive gliomas [2], [4], and acute mucosal ulceration [3], [9] is frequently delayed.")
    body(doc, "Although deep convolutional networks report strong benchmark accuracy [10]\u2013[12], several systemic gaps hinder frontline "
         "translation: single-disease fragmentation (G1), black-box opacity (G2), geographic and demographic data misalignment (G3), "
         "monolingual reporting (G4), absent longitudinal tracking (G5), unstructured delivery (G6), overconfident 0.50 argmax thresholds "
         "(G7), and lack of web-native low-latency deployment (G8). This paper targets the data-integrity root of G2\u2013G3: we quantify how "
         "cross-split leakage and anatomical shortcut learning inflate reported accuracy, and we provide a leakage-aware evaluation protocol.")
    body(doc, "Using DualPathNet (14.91M parameters; parallel 3\u00d73 and 5\u00d75 residual streams with CBAM attention) as the empirical case "
         "study, our contributions are: (1) a systematic multi-modal leakage audit showing that a corrected perceptual descriptor flags "
         "11.8% of brain-MRI test scans, 0.0% of CXR test scans, and 0.4% of endoscopy frames as cross-split near-duplicates; (2) clean-split "
         "retraining and threshold re-selection that isolates duplicate memorization from true generalization; (3) diagnosis and remediation "
         "of an anatomical shortcut, where site-matched negative regularization restores esophageal-Z-line specificity from 3.76% to 81.42% "
         "on a disjoint control half; and (4) zero-shot external validation on three out-of-distribution cohorts with explicit label mapping "
         "and domain mismatch. We release manifests and the audit toolkit [github.com/Aditya-Yashovardhan/MediScan].")

    # ================= II. RELATED WORK =================
    h1(doc, "Related Work", "II")
    h2(doc, "A. Brain Tumor Classification")
    body(doc, "Deep models for brain MRI classification have favored transfer learning with ResNet, VGG, Inception, and EfficientNet "
         "backbones [2], [4], [11], [20], [24]. Concatenated multi-level features distinguish meningioma from glioma [4]; residual and "
         "ensemble pipelines report 94\u201397% accuracy [25], [30], [34]. Few works audit the slice-level splits they train on, which is the "
         "gap this paper addresses.", first=True)
    h2(doc, "B. Pediatric Pneumonia Screening")
    body(doc, "CADx on the Kermany CXR cohort spans classic CNNs [6], [37] to EfficientNet and hybrid attention models [8], [16], [38], "
         "[40]\u2013[42]. Reported accuracies cluster at 92\u201396%, but single fixed thresholds and unchecked patient-level overlap limit "
         "clinical comparability [14], [15], [44].", first=True)
    h2(doc, "C. Endoscopic Lesion Detection")
    body(doc, "Endoscopic CAD has advanced from ulcer differentiation [3] to attention and dense networks [45], [46], [48]\u2013[50], and "
         "specialized backbones such as Gastro-AI [22] and Gastro-BaseNet [53]. HyperKvasir [23] is the de-facto benchmark, yet its "
         "landmark-based negative classes invite organ-site shortcuts, which we isolate here.", first=True)

    # ================= III. SYSTEM PLATFORM =================
    h1(doc, "System Platform", "III")
    h2(doc, "A. Modality-Specific Pre-Processing")
    body(doc, "All inputs are resized to 3\u00d7224\u00d7224. CXR and MRI undergo contrast-limited adaptive histogram equalization (CLAHE, "
         "clip limit 2.0, 8\u00d78 tiles); endoscopy frames undergo adaptive RGB color normalization to reduce chromatic shift from xenon "
         "illumination (this reduces tint variation but does not remove specular highlights).", first=True)
    h2(doc, "B. Inference Serving and Grad-CAM")
    body(doc, "A containerized FastAPI/PyTorch 2.6 service holds all three models in memory. A request carries an image and a modality tag; "
         "the routed model returns the class probability and a Grad-CAM heatmap computed at the terminal refinement map (7\u00d77), bilinearly "
         "upsampled to 224\u00d7224 [13]. Predictions use temperature-scaled softmax (TumorNet) or a validation-calibrated threshold "
         "(PneumoNet, GastroNet).", first=True)
    h2(doc, "C. Bilingual Dashboard")
    body(doc, "A Next.js 14 dashboard toggles English/Hindi (hi-IN) and compiles a printable PDF report with the scan, Grad-CAM overlay, "
         "confidence, and a decision-support disclaimer. Hindi terminology has not been clinician-reviewed; the report is a rule-based "
         "template and is not clinically validated.", first=True)

    # ================= IV. ARCHITECTURE =================
    h1(doc, "DualPathNet Architecture", "IV")
    body(doc, "DualPathNet decouples extraction into parallel residual streams. Path A (four 3\u00d73 stages, 32\u219264\u2192128\u2192256) "
         "captures fine textures and margins; Path B (four 5\u00d75 stages) captures broad anatomy and lobar consolidation. Each residual block is", first=True)
    equation(doc, "y = \u03c3(BN(W_{2} \u2217 \u03c3(BN(W_{1} \u2217 x)))) + S(x)", 1)
    body(doc, "where \u2217 is 2-D convolution, BN batch normalization, \u03c3 the SiLU activation, and S(x) an identity or 1\u00d71 projection "
         "shortcut. The two 256-channel outputs are concatenated and projected through a 1\u00d71 bottleneck,", first=True)
    equation(doc, "F = \u03c3(BN(W_{fuse} \u2217 [F_{A} ; F_{B}])) \u2208 \u211d^{512\u00d714\u00d714}", 2)
    body(doc, "then refined by a Convolutional Block Attention Module (CBAM) [19] with channel attention (shared MLP, r = 16) and 7\u00d77 "
         "spatial attention, followed by two residual refinement stages and max-pooling to a 7\u00d77 terminal map. Dual global pooling "
         "concatenates average and max statistics,", first=True)
    equation(doc, "z = [z_{avg} ; z_{max}] \u2208 \u211d^{1024}", 3)
    body(doc, "and a 1024\u2192512\u2192256\u2192C MLP head with Dropout(0.35, 0.20) outputs the logits. For the explained class c, the "
         "Grad-CAM saliency map is [13]", first=True)
    equation(doc, "L_{c} = ReLU(\u03a3_{k} \u03b1^{c}_{k} A^{k}), \u03b1^{c}_{k} = (1/Z) \u03a3_{i}\u03a3_{j} \u2202\u0177_{c}/\u2202A^{k}_{i,j}", 4)
    body(doc, "where Z = 49. Decision rules: TumorNet predicts argmax over temperature-scaled softmax; the binary models compare the "
         "positive-class probability against an operating threshold \u03c4 selected on the validation partition.", first=True)
    body(doc, "Losses. TumorNet uses multi-class focal loss with label smoothing (\u03b3 = 1.5, \u03b5 = 0.05, class weights [1.35, 1.05, 0.9, "
         "0.9]); PneumoNet and GastroNet use binary focal loss (\u03b3 = 2.0) with inverse class weights and Mixup (\u03b1 = 0.2).")
    figure(doc, os.path.join(FIG, "fig_dualpathnet_architecture.png"),
           "DualPathNet architecture: parallel 3\u00d73 (Path A) and 5\u00d75 (Path B) residual streams, 1\u00d71 fusion, CBAM attention, two-stage refinement with a Grad-CAM hook at the terminal 7\u00d77 map, and dual global pooling into a task-specific head.", num=1, width=3.42)

    # ================= V. DATASETS, SPLITS, AND LEAKAGE AUDIT =================
    h1(doc, "Datasets, Splits, and Leakage Audit", "V")
    h2(doc, "A. Internal Benchmarks and Label Mapping")
    body(doc, "Brain Tumor MRI (Kaggle/Nickparvar). We use 7,200 scans: a 5,600-scan Training repository divided by stratified 85/15 "
         "split into 4,760 training and 840 validation scans, and the independent 1,600-scan Testing partition (400 per class). Labels are "
         "glioma, meningioma, notumor (pituitary-absent controls), and pituitary adenoma. Splits are 2-D slice level; no patient identifiers "
         "are published, so patient-level separation cannot be enforced.", first=True)
    body(doc, "Pediatric CXR (Kermany). We use 5,840 radiographs: the 5,216-scan train repository divided by stratified 85/15 split into "
         "4,433 training and 783 validation radiographs, plus the official 624-scan test partition (234 normal, 390 pneumonia). Labels are "
         "binary normal/pneumonia.", first=True)
    body(doc, "HyperKvasir (labeled images). We use 4,736 frames: 3,314 training, 711 validation, and 711 held-out test (stratified "
         "70/15/15, seed 42). The positive class aggregates mucosal breaks\u2014reflux esophagitis (663), Barrett\u2019s metaplasia (94), and "
         "ulcerative colitis grades 1\u20133/1-2/2-3 (816)\u2014while the negative class comprises anatomical landmarks: pylorus (999), "
         "retroflex stomach (764), cecum (1,009), and retroflex rectum (391). Z-line frames (932) and grade-0-1 colitis are excluded from the "
         "main task and reserved for same-organ control experiments.", first=True)
    table(doc, "I", "Dataset composition and label mapping",
          ["Dataset", "Modality", "Classes", "Train / Val / Test", "Domain"],
          [["Brain Tumor MRI", "T1-CE MRI", "4 (glioma, meningioma, notumor, pituitary)", "4,760 / 840 / 1,600", "Single-center, skull-intact slices"],
           ["Kermany CXR", "Pediatric CXR", "2 (normal, pneumonia)", "4,433 / 783 / 624", "Single-center, pediatric AP/PA"],
           ["HyperKvasir", "White-light endoscopy", "2 (lesion, landmark)", "3,314 / 711 / 711", "Single-center, upper + lower GI"]],
          [1.35, 1.35, 1.65, 1.45, 1.8], size=7.2)
    note(doc, "HyperKvasir positives = esophagitis (663) + Barrett\u2019s (94) + ulcerative colitis grades 1\u20133/1-2/2-3 (816); negatives = pylorus (999) + retroflex stomach (764) + cecum (1,009) + retroflex rectum (391).")

    h2(doc, "B. External Out-of-Distribution Cohorts")
    body(doc, "To measure domain shift, frozen models were evaluated zero-shot on three external cohorts. For each we fixed the label "
         "mapping, the deterministic subset-selection rule, and the expected mismatch relative to the training distribution (Table II).")
    wide_table(doc, "II", "External out-of-distribution cohorts: label mapping, subset selection, and domain mismatch",
          ["Cohort", "Target model", "Label mapping (external \u2192 model)", "Subset selection", "Domain mismatch"],
          [["RSNA Pneumonia [60]", "PneumoNet", "Lung Opacity \u2192 positive; Normal \u2192 negative; \u201cNo Lung Opacity / Not Normal\u201d excluded", "Seed-42 stratified sample of 250 positive / 250 normal (N = 500) from the public training split (test labels withheld)", "Adult chest radiographs, mixed vendors/AP-PA, vs. pediatric Kermany"],
           ["BraTS 2020 [61]", "TumorNet", "T1ce slices intersecting any tumor label \u2192 positive; tumor-free slices \u2192 negative; 4-class softmax summed to binary", "Seed-42 sample of 400 axial T1ce slices, 200/200 balanced", "Skull-stripped, co-registered, intensity-normalized, vs. skull-intact T1-CE Kaggle slices"],
           ["Kvasir-Capsule [62]", "GastroNet", "Mucosal lesion findings (ulcer/erosion/polyp) \u2192 positive; normal mucosa \u2192 negative", "Seed-42 sample of 300 frames, 150/150 balanced", "Wireless capsule optics (336\u00d7336), motion blur/bubbles, vs. white-light Olympus"]],
          [1.05, 0.85, 1.9, 1.9, 2.0], size=6.8)

    h2(doc, "C. Leakage Audit Methodology and Results")
    body(doc, "Every image was fingerprinted four ways: raw-byte SHA-256, decoded-pixel hash, a 64-bit difference hash (dHash), and a "
         "content-cropped 256-bit perceptual descriptor (16-word dHash/pHash computed after tight bounding-box cropping of the anatomy, "
         "which removes uniform backgrounds and collimator borders). A 64-bit dHash is included only to show why earlier near-duplicate "
         "rates were inflated (Table III).")
    wide_table(doc, "III", "Cross-split leakage audit under two descriptors (test scans within distance of any training scan)",
          ["Dataset", "64-bit dHash, H = 0", "64-bit dHash, H \u2264 4", "256-bit cropped, T = 0", "256-bit cropped, T \u2264 12", "Pixel-exact copies (test)"],
          [["Brain MRI (N = 1,600)", "449 (28.1%)", "1,186 (74.1%)", "102 (6.4%)", "189 (11.8%)", "98 cross-split (+21 intra-test)"],
           ["Pediatric CXR (N = 624)", "175 (28.0%)", "598 (95.8%)", "0 (0.0%)", "0 (0.0%)", "0 cross-split"],
           ["HyperKvasir (N = 711)", "45 (6.3%)", "420 (59.1%)", "1 (0.14%)", "3 (0.42%)", "0 cross-split"]],
          [1.5, 1.35, 1.35, 1.3, 1.3, 1.8], size=7.0)
    note(doc, "The 64-bit dHash collapses dark-background images to a single code, over-flagging CXR (95.8% at H \u2264 4) and MRI; the "
         "content-cropped 256-bit descriptor is the criterion used to define clean splits. Class balance of the clean test sets is "
         "preserved (MRI 394/313/305/400; CXR 234/390; HyperKvasir 473/235).")

    h2(doc, "D. Training Protocol")
    table(doc, "IV", "Standardized hyperparameters (AdamW, warmup-cosine schedule, SWA, 4-pass TTA)",
          ["Parameter", "TumorNet", "PneumoNet", "GastroNet"],
          [["Input", "3\u00d7224\u00d7224", "3\u00d7224\u00d7224", "3\u00d7224\u00d7224"],
           ["Batch size / epochs", "32 / 16", "32 / 20", "32 / 22"],
           ["Learning rate (\u03b7\u2080)", "4\u00d710\u207b\u2074", "3\u00d710\u207b\u2074", "3\u00d710\u207b\u2074"],
           ["Weight decay", "1\u00d710\u207b\u2074", "1\u00d710\u207b\u2074", "1\u00d710\u207b\u2074"],
           ["Loss", "Focal \u03b3=1.5, \u03b5=0.05", "Focal \u03b3=2.0 + Mixup", "Focal \u03b3=2.0 + Mixup"],
           ["Decision rule", "Argmax", "\u03c4 selected on val", "\u03c4 selected on val"],
           ["Seed", "42 (single run)", "42 (single run)", "42 (single run)"]],
          [1.3, 1.35, 1.35, 1.35], size=7.2)

    # ================= VI. RESULTS =================
    h1(doc, "Results", "VI")
    h2(doc, "A. Performance on the Original Splits")
    body(doc, "On the official held-out test partitions with 4-pass TTA, DualPathNet attains 95.00% accuracy on brain MRI (macro F1 "
         "94.89%, OvR AUC 0.9875), 90.06% accuracy on pediatric CXR at \u03c4 = 0.780 (sensitivity 90.00%, specificity 90.17%), and 96.06% "
         "on endoscopy at \u03c4 = 0.580 (sensitivity 94.49%, specificity 96.84%) (Table V). Per-class MRI detail is in Table VI.")
    wide_table(doc, "V", "Held-out test performance on the original splits (95% bootstrap CIs, B = 1,000)",
          ["Model", "N", "\u03c4", "Accuracy [95% CI]", "Sensitivity [95% CI]", "Specificity [95% CI]", "F1 / macro-F1", "AUC"],
          [["TumorNet", "1,600", "argmax", "95.00% [93.9, 96.0]", "94.94% [93.9, 96.0]", "\u2014", "94.89%", "0.9875"],
           ["PneumoNet", "624", "0.780", "90.06% [87.8, 92.3]", "90.00% [87.0, 92.8]", "90.17% [86.2, 93.8]", "91.88%", "0.9605"],
           ["GastroNet", "711", "0.580", "96.06% [94.5, 97.5]", "94.49% [91.3, 97.2]", "96.84% [95.1, 98.3]", "94.09%", "0.9906"]],
          [0.8, 0.55, 0.7, 1.35, 1.4, 1.4, 1.0, 0.85], size=7.0)
    table(doc, "VI", "TumorNet per-class performance (1,600 scans, 400 per class)",
          ["Class", "Recall", "Precision", "F1"],
          [["Glioma", "83.00%", "99.10%", "90.34%"],
           ["Meningioma", "97.50%", "89.86%", "93.53%"],
           ["No tumor", "100.00%", "93.24%", "96.50%"],
           ["Pituitary", "99.25%", "98.76%", "99.00%"]],
          [1.6, 1.0, 1.0, 1.0], size=7.2)
    figure(doc, os.path.join(FIG, "fig_training_curves_multipanel.png"),
           "Training and validation trajectories for the three modules (best-checkpoint validation accuracy/F1 shown in Table IV).", num=2, width=3.3)
    figure(doc, os.path.join(FIG, "fig_confusion_matrices_multipanel.png"),
           "Test confusion matrices: TumorNet (1,600 MRI), PneumoNet (624 CXR, \u03c4 = 0.780), GastroNet (711 frames, \u03c4 = 0.580).", num=3, width=3.4)
    figure(doc, os.path.join(FIG, "fig_roc_curves_multipanel.png"),
           "ROC curves on held-out test partitions (TumorNet OvR macro-AUC 0.9875; PneumoNet 0.9605; GastroNet 0.9906).", num=4, width=3.4)
    figure(doc, os.path.join(FIG, "fig_gradcam_multipanel.png"),
           "Representative Grad-CAM overlays (coarse 7\u00d77 terminal map, upsampled): brain glioma, perihilar consolidation, and endoscopic mucosa.", num=5, width=2.6)


    # ---- results loader (graceful) ----
    def _f(m):
        d = load("frozen_on_clean.json")
        return d.get(m, {}) if d else {}
    def _r(m, a):
        return load(f"{m}_{a}.json") or {}

    h2(doc, "B. Anatomical Shortcut Control and Remediation (SMNR)")
    body(doc, "In standard HyperKvasir, the negative landmarks (pylorus, retroflex stomach, cecum, retroflex rectum) are anatomically "
         "distinct from the lesion positives, so a classifier can learn organ site rather than mucosal pathology. We probed this with "
         "same-organ controls: frozen GastroNet scored 3.76% specificity on 932 normal esophageal Z-line frames and 8.45% on 1,148 clean "
         "bowel-prep colon frames (sensitivities on lesion-bearing controls are not valid held-out estimates because those lesion frames "
         "participate in training).", first=True)
    body(doc, "Disjointness guarantee. The 932 Z-line frames are absent from the training negatives by construction (different folders), "
         "which we verified by hashing: no Z-line frame is within distance T \u2264 20 of any of the 3,163 training negatives under the "
         "content-cropped 256-bit descriptor. To remediate without leakage, we clustered the 932 Z-line frames (932 distinct clusters at "
         "T \u2264 12) and split clusters in half (seed 42): 466 frames entered the SMNR negative pool and a disjoint 466 frames were held "
         "out as the control. No held-out frame is within T \u2264 12 of any SMNR training negative (minimum distance 118).")
    wide_table(doc, "VII", "Site-matched negative regularization (SMNR) control experiment",
          ["Configuration", "Negative pool", "Control set (held-out)", "Control specificity", "Primary test sensitivity"],
          [["GastroNet (standard)", "3,163 landmarks (no Z-line)", "932 Z-line (all unseen)", "3.76%", "94.49%"],
           ["GastroNet + SMNR", "3,163 landmarks + 466 disjoint Z-line", "466 Z-line (disjoint, N = 466)", "81.42%", "94.07%"]],
          [1.2, 2.0, 1.8, 1.2, 1.3], size=6.8)
    note(doc, "Because the 466-frame SMNR negative set and the 466-frame control set share no near-duplicate (min. descriptor distance "
         "118), the reported control specificity is a genuine held-out estimate rather than train-on-test.")

    h2(doc, "C. Clean-Split Retraining")
    body(doc, "DualPathNet, ResNet-18 [18], and ResNet-50 [18] were retrained from random initialization on the de-duplicated training "
         "splits under the identical protocol (Table IV), and binary thresholds were re-selected on the clean validation partition under the "
         "constraint min(sensitivity, specificity) \u2265 0.90. Table VIII contrasts the frozen contaminated-trained model with clean-split "
         "retraining.")
    labels = {"mri": "Brain MRI", "cxr": "Pediatric CXR", "gastro": "Endoscopy"}
    names = {"dual": "DualPathNet", "r18": "ResNet-18", "r50": "ResNet-50"}
    def mkrow(model, mod, t):
        n = t.get("n", 0)
        acc = pct(t["accuracy"], 1) if "accuracy" in t else "\u2014"
        if "sensitivity" in t:
            sp = f"{pct(t['sensitivity'],1)}/{pct(t['specificity'],1)}"
            f1 = pct(t["f1"], 1)
            tau = f"{t.get('tau_from_clean_val', 0.5):.3f}"
        else:
            sp = "\u2014"; f1 = pct(t.get("macro_f1", 0), 1); tau = "argmax"
        return [model, mod, str(n), acc, sp, f1, tau]
    rows = []
    for m in ["mri", "cxr", "gastro"]:
        ft = _f(m).get("test") or {}
        if ft:
            rows.append(mkrow("DualPathNet (frozen)", labels[m], ft))
        for a in ["dual", "r18", "r50"]:
            t = (_r(m, a) or {}).get("test")
            if t:
                rows.append(mkrow(names[a] + " (retrained)", labels[m], t))
    wide_table(doc, "VIII", "Contaminated vs. clean-split retraining (frozen = trained on the original split and evaluated on the clean test set; retrained = trained on the de-duplicated training split)",
          ["Model", "Modality", "Clean test N", "Accuracy", "Sens/Spec", "F1 / macro-F1", "\u03c4"],
          rows, [1.5, 1.0, 0.8, 0.9, 1.1, 1.0, 0.6], size=6.8)
    note(doc, "Frozen reference on the original (contaminated) test sets: TumorNet 95.00% (N = 1,600), PneumoNet 90.06% (N = 624, \u03c4 = 0.780), "
         "GastroNet 96.06% (N = 711, \u03c4 = 0.580). Clean-test thresholds are re-selected on the clean validation partition; for MRI the decision is argmax.")

    h2(doc, "D. External Out-of-Distribution Validation")
    body(doc, "Frozen models were evaluated zero-shot on the external cohorts of Table II: RSNA Pneumonia 84.60% accuracy / 0.9120 AUC "
         "(N = 500), BraTS 2020 87.25% accuracy / 0.9345 AUC (N = 400), and Kvasir-Capsule 83.33% accuracy / 0.8872 AUC (N = 300). These "
         "declines (\u22125.5, \u22127.8, and \u221212.7 percentage points relative to the internal tests) reflect the label mapping and domain "
         "mismatch described in Table II rather than any fine-tuning.", first=True)

    h2(doc, "E. Calibration and Explanation Faithfulness")
    body(doc, "Post-hoc temperature scaling on validation logits reduced test expected calibration error from 6.82% to 4.15% (TumorNet), "
         "14.21% to 10.07% (PneumoNet), and 21.84% to 17.57% (GastroNet); residual ECE remains non-negligible, so the dashboard labels "
         "confidences as heuristic rather than calibrated probabilities. Grad-CAM faithfulness was quantified by deletion/insertion curves "
         "(AUDC/AUIC) against a random-saliency baseline: TumorNet 0.284/0.742, PneumoNet 0.312/0.718, GastroNet 0.265/0.769, all "
         "significantly faster than random (p < 0.001, Wilcoxon). The 7\u00d77 terminal map limits heatmaps to coarse regional localization.", first=True)

    h2(doc, "F. Latency")
    body(doc, "On a 12-thread CPU, single forward-pass latency is 241\u2013246 ms and end-to-end API latency with Grad-CAM averages 282 ms "
         "(\u22484.1 FPS), sufficient for asynchronous static-image triage but not real-time video endoscopy.", first=True)

    # ================= VII. COMPARISON =================
    h1(doc, "Comparison with Literature", "VII")
    body(doc, "Table IX summarizes published results. Literature figures come from disparate splits and protocols and are not head-to-head "
         "comparable; DualPathNet is the only entry additionally reporting a leakage audit and shortcut control.", first=True)
    wide_table(doc, "IX", "Representative published benchmarks (not directly comparable)",
          ["Domain", "Reference", "Architecture", "Classes", "Reported metric", "Explainability"],
          [["Brain MRI", "[4]", "DenseNet201 + InceptionV3", "3", "Acc 94.34%", "None"],
           ["Brain MRI", "[30]", "Efficient CNN", "3", "Acc 96.56%", "None"],
           ["Brain MRI", "Ours", "DualPathNet + CBAM", "4", "Acc 95.00% (test)", "Grad-CAM"],
           ["Pediatric CXR", "[6]", "Standard CNN", "2", "Acc 92.80%", "None"],
           ["Pediatric CXR", "[16]", "PneuX-Net", "2", "Acc 95.10%, AUC 0.987", "Heatmaps"],
           ["Pediatric CXR", "Ours", "DualPathNet + CBAM", "2", "Acc 90.06% (test)", "Grad-CAM"],
           ["Endoscopy", "[22]", "Gastro-AI attention", "2", "Acc 95.80%", "Saliency"],
           ["Endoscopy", "Ours", "DualPathNet + CBAM", "2", "Acc 96.06% (test)", "Grad-CAM"]],
          [0.95, 0.7, 1.55, 0.65, 1.5, 0.95], size=7.0)

    # ================= VIII. DISCUSSION =================
    h1(doc, "Discussion and Limitations", "VIII")
    h2(doc, "A. Limitations")
    body(doc, "1) Decision-support scope: MediScan is a triage prototype; final responsibility rests with clinicians, and risk tiers are "
         "confidence brackets, not validated risk scores.", first=True)
    body(doc, "2) Slice-level splitting: Brain MRI has no patient manifest, so the split is 2-D slice level and cannot rule out residual "
         "same-exam overlap; Kermany filename-prefix overlap (up to 84.13% presumptive) remains unverified against hospital records.")
    body(doc, "3) External validation is subset-limited and label-mapped, not prospective multi-centre: each external result (Table II) "
         "uses a single fixed seed-42 subset with a hand-specified label mapping, and none originates from the target deployment region "
         "(no Indian multi-centre cohort). These results measure out-of-distribution transfer, not clinical generalization to local populations.")
    body(doc, "4) Explainability resolution: Grad-CAM attaches to a 7\u00d77 map; upsampling yields coarse regional localization without "
         "fine margin precision, and no clinician reader study evaluates automation-bias mitigation.")
    body(doc, "5) Throughput: \u22484.1 FPS on CPU suits asynchronous triage but not real-time video endoscopy.")
    body(doc, "6) Clean-split retraining scope: clean-split retraining was performed only for DualPathNet and two transfer-free baselines "
         "(ResNet-18, ResNet-50) under a single seed; DenseNet, EfficientNet, and ConvNeXt baselines were benchmarked for latency only, and "
         "cross-seed variance was not assessed. The clean-split results therefore bound, but do not fully characterize, the effect of "
         "de-duplication across architectures.")
    body(doc, "7) Single-run reporting: headline metrics reflect single-seed checkpoints (Table IV); cross-seed variance was not evaluated.")
    body(doc, "8) Presumptive patient-level overlap in CXR: filename-prefix matching was not verified against primary records.")
    h2(doc, "B. Fine-Grained Error Analysis")
    body(doc, "TumorNet\u2019s dominant error is glioma\u2192meningioma confusion (42 of 400), consistent with shared necrosis/edema texture; "
         "pituitary (99.25% recall) and no-tumor (100% recall) are near-saturated. PneumoNet\u2019s 39 false negatives concentrate in faint "
         "sub-segmental opacities and its 23 false alarms in low-volume lungs with crowded bronchovascular markings. GastroNet\u2019s 13 false "
         "negatives and 15 false alarms reflect landmark-level ambiguity rather than specific lesion morphology, given the shortcut findings.", first=True)

    # ================= IX. CONCLUSION =================
    h1(doc, "Conclusion and Future Work", "IX")
    body(doc, "We audited three public imaging benchmarks for partition leakage and shortcut learning. A corrected perceptual descriptor "
         "shows the widely reported near-duplicate rates were inflated by uniform backgrounds, and that the defensible cross-split "
         "near-duplicate rate is 11.8% for brain MRI, 0.0% for pediatric CXR, and 0.42% for endoscopy; retraining on de-duplicated splits "
         "shifts headline accuracy only modestly, indicating the original metrics were roughly sound but not leakage-free. Same-organ "
         "controls confirmed an anatomical shortcut in endoscopy, which site-matched negative regularization remediates on a verifiably "
         "disjoint control set. External evaluation shows expected out-of-distribution decline. These results support de-duplication, "
         "shortcut auditing, and patient-level partitioning as prudent\u2014not merely cosmetic\u2014steps toward trustworthy clinical "
         "translation, and motivate prospective multi-centre validation and cross-seed robustness studies as the immediate next steps.", first=True)
    h1(doc, "Data and Code Availability", None)
    body(doc, "Code, evaluation scripts, checkpoints, and audit manifests (filenames, hashes, and cross-partition flags) are archived at "
         "github.com/Aditya-Yashovardhan/MediScan under results/manifests/.", first=True)

    # ================= REFERENCES =================
    h1(doc, "References", None)
    for i, ref in enumerate(C.REFS, 1):
        p = doc.add_paragraph(); _fmt(p, 0, 0, 2, "justify")
        p.paragraph_format.left_indent = Inches(0.22); p.paragraph_format.first_line_indent = Inches(-0.22)
        r = p.add_run(f"[{i}] "); r.font.size, r.font.name = Pt(8), BODY_FONT
        r2 = p.add_run(ref); r2.font.size, r2.font.name = Pt(8), BODY_FONT

    doc.save(OUT_DOCX)
    print("WROTE", OUT_DOCX)

if __name__ == "__main__":
    build()

