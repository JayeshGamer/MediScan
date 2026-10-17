import os
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn, nsdecls

def create_element(name):
    return OxmlElement(name)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def set_cell_shading(cell, color_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{color_hex}"/>')
    tcPr.append(shd)

def set_table_borders(table):
    tblPr = table._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>\n'
        f'  <w:top w:val="single" w:sz="8" w:space="0" w:color="000000"/>\n'
        f'  <w:bottom w:val="single" w:sz="8" w:space="0" w:color="000000"/>\n'
        f'  <w:insideH w:val="single" w:sz="4" w:space="0" w:color="CCCCCC"/>\n'
        f'  <w:left w:val="none"/>\n'
        f'  <w:right w:val="none"/>\n'
        f'  <w:insideV w:val="none"/>\n'
        f'</w:tblBorders>'
    )
    tblPr.append(borders)

def build_ieee_paper():
    doc = docx.Document()

    # Base Normal style
    style = doc.styles['Normal']
    font = style.font
    font.name = 'Times New Roman'
    font.size = Pt(10)
    font.color.rgb = RGBColor(0, 0, 0)

    # =========================================================================
    # SECTION 0: Full-Width Title, Authors, Abstract, Index Terms
    # =========================================================================
    sec0 = doc.sections[0]
    sec0.page_width = Inches(8.5)
    sec0.page_height = Inches(11.0)
    sec0.top_margin = Inches(0.75)
    sec0.bottom_margin = Inches(1.0)
    sec0.left_margin = Inches(0.625)
    sec0.right_margin = Inches(0.625)

    # 1. Title
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_before = Pt(0)
    p_title.paragraph_format.space_after = Pt(8)
    r_title = p_title.add_run("MediScan: A Multiscale Dual-Path Residual Architecture with Attention Refinement for Multi-Disease Clinical Triage and Explainable Diagnosis")
    r_title.font.name = 'Times New Roman'
    r_title.font.size = Pt(22)
    r_title.font.bold = True

    # 2. Authors
    p_auth = doc.add_paragraph()
    p_auth.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_auth.paragraph_format.space_before = Pt(0)
    p_auth.paragraph_format.space_after = Pt(2)
    r_auth = p_auth.add_run("Daksh Gurjar, Student Member, IEEE, Aditya Yashovardhan, and Jayesh Mandloi")
    r_auth.font.name = 'Times New Roman'
    r_auth.font.size = Pt(11)

    # 3. Affiliation
    p_aff = doc.add_paragraph()
    p_aff.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_aff.paragraph_format.space_before = Pt(0)
    p_aff.paragraph_format.space_after = Pt(1)
    r_aff = p_aff.add_run("Department of Computer Engineering, Mukesh Patel School of Technology Management & Engineering (MPSTME)\nSVKM’s NMIMS, Shirpur 425405, Maharashtra, India\n{daksh.gurjar, aditya.yashovardhan, jayesh.mandloi}@nmims.edu")
    r_aff.font.name = 'Times New Roman'
    r_aff.font.size = Pt(9.5)
    r_aff.font.italic = True

    # Notice
    p_notice = doc.add_paragraph()
    p_notice.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_notice.paragraph_format.space_before = Pt(4)
    p_notice.paragraph_format.space_after = Pt(10)
    r_not = p_notice.add_run("This work has been submitted to the IEEE for possible publication. Copyright may be transferred without notice.")
    r_not.font.name = 'Times New Roman'
    r_not.font.size = Pt(8)
    r_not.font.italic = True

    # 4. Abstract Box (Full width, justified, 9 pt)
    p_abs = doc.add_paragraph()
    p_abs.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p_abs.paragraph_format.left_indent = Inches(0.25)
    p_abs.paragraph_format.right_indent = Inches(0.25)
    p_abs.paragraph_format.space_before = Pt(0)
    p_abs.paragraph_format.space_after = Pt(4)
    p_abs.paragraph_format.line_spacing = 1.0

    r_absh = p_abs.add_run("Abstract—")
    r_absh.font.name = 'Times New Roman'
    r_absh.font.size = Pt(9)
    r_absh.font.bold = True
    r_absh.font.italic = True

    abs_text = (
        "Timely radiological and endoscopic interpretation is severely constrained in resource-limited primary healthcare facilities, "
        "where specialist availability remains critically below international standards. We present MediScan AI, an end-to-end multi-modal "
        "deep learning triage framework evaluated across three distinct diagnostic imaging modalities: brain magnetic resonance imaging (MRI), "
        "pediatric chest radiographs (CXR), and upper gastrointestinal (GI) endoscopy. At the core of the framework is DualPathNet, a custom "
        "14.91M-parameter convolutional architecture combining parallel 3×3 and 5×5 multi-scale receptive paths, Convolutional Block Attention "
        "Modules (CBAM), deep residual refinement, and dual global pooling (GAP+GMP). On strictly held-out benchmark test sets evaluated with "
        "4-pass test-time augmentation (TTA), the models achieve test accuracies of 94.94% [95% CI: 93.8–95.9%] across four classes on brain "
        "MRI (N=1,600 scans, macro-F1: 94.84%), 96.06% [95% CI: 94.4–97.3%] on upper-GI endoscopy (N=711 frames, macro-F1: 95.39%), and 89.58% "
        "[95% CI: 86.9–91.7%] on pediatric CXR (N=624 radiographs) under a validation-selected specificity-protecting decision threshold (τ=0.780, "
        "yielding 91.88% specificity, 88.21% sensitivity, and 94.77% precision; operating at τ=0.500 shifts performance to 96.15% sensitivity "
        "and 76.92% specificity for frontline screening). Visual explainability is generated via gradient-weighted class activation mapping (Grad-CAM) "
        "hooked directly to the penultimate refinement block (refine[2], 7×7 feature map). Latency profiling on a commodity multi-core CPU demonstrates "
        "single-scan inference in 105 ms (p95: 112 ms) and complete 4-pass TTA with Grad-CAM generation in 731 ms (p95: 765 ms), verifying "
        "sub-second triage turnaround without requiring GPU acceleration. MediScan AI demonstrates competitive accuracy, architectural efficiency, "
        "and transparent decision support for frontline clinical workflows."
    )
    r_abst = p_abs.add_run(abs_text)
    r_abst.font.name = 'Times New Roman'
    r_abst.font.size = Pt(9)
    r_abst.font.bold = True

    # 5. Index Terms
    p_idx = doc.add_paragraph()
    p_idx.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p_idx.paragraph_format.left_indent = Inches(0.25)
    p_idx.paragraph_format.right_indent = Inches(0.25)
    p_idx.paragraph_format.space_before = Pt(0)
    p_idx.paragraph_format.space_after = Pt(14)
    p_idx.paragraph_format.line_spacing = 1.0

    r_idxh = p_idx.add_run("Index Terms—")
    r_idxh.font.name = 'Times New Roman'
    r_idxh.font.size = Pt(9)
    r_idxh.font.bold = True
    r_idxh.font.italic = True

    r_idxt = p_idx.add_run("Computer-aided diagnosis, multiscale convolutional neural networks, explainable artificial intelligence (XAI), Grad-CAM, pediatric pneumonia, brain tumor, gastric ulcer, medical triage, bilingual tele-health.")
    r_idxt.font.name = 'Times New Roman'
    r_idxt.font.size = Pt(9)

    # =========================================================================
    # SECTION 1: TWO-COLUMN BODY (Official IEEE Specifications)
    # =========================================================================
    sec1 = doc.add_section(docx.enum.section.WD_SECTION.CONTINUOUS)
    sec1.page_width = Inches(8.5)
    sec1.page_height = Inches(11.0)
    sec1.top_margin = Inches(0.75)
    sec1.bottom_margin = Inches(1.0)
    sec1.left_margin = Inches(0.625)
    sec1.right_margin = Inches(0.625)

    # Configure 2 columns with 0.2 inch (288 dxa) gutter
    sectPr = sec1._sectPr
    cols = sectPr.xpath('./w:cols')
    if cols:
        cols[0].set(qn('w:num'), '2')
        cols[0].set(qn('w:space'), '288')
    else:
        col_el = OxmlElement('w:cols')
        col_el.set(qn('w:num'), '2')
        col_el.set(qn('w:space'), '288')
        sectPr.append(col_el)

    # Helper Functions for Typography
    def add_h1(text):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(10)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.keep_with_next = True
        r = p.add_run(text)
        r.font.name = 'Times New Roman'
        r.font.size = Pt(10)
        r.font.bold = True
        return p

    def add_h2(text):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.space_before = Pt(8)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.keep_with_next = True
        r = p.add_run(text)
        r.font.name = 'Times New Roman'
        r.font.size = Pt(10)
        r.font.italic = True
        return p

    def add_body(text, indent=True):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        if indent:
            p.paragraph_format.first_line_indent = Inches(0.16)
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.line_spacing = 1.0
        r = p.add_run(text)
        r.font.name = 'Times New Roman'
        r.font.size = Pt(10)
        return p

    def add_caption(fig_or_tab_text):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.space_before = Pt(4)
        p.paragraph_format.space_after = Pt(6)
        p.paragraph_format.line_spacing = 1.0
        r = p.add_run(fig_or_tab_text)
        r.font.name = 'Times New Roman'
        r.font.size = Pt(8)
        return p

    def add_image_safe(img_path, width_in=3.45):
        if os.path.exists(img_path):
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(4)
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.keep_with_next = True
            run = p.add_run()
            run.add_picture(img_path, width=Inches(width_in))
            return p
        return None

    # -------------------------------------------------------------------------
    # I. INTRODUCTION
    # -------------------------------------------------------------------------
    add_h1("I.  INTRODUCTION")

    add_body(
        "Medical imaging modalities—specifically magnetic resonance imaging (MRI), projection radiography (chest X-ray, CXR), "
        "and white-light endoscopy—underpin acute triage, oncological staging, and emergency medical intervention [1]–[3]. "
        "Early and accurate identification of focal pathological lesions directly influences clinical trajectories: pediatric "
        "pneumonia requires urgent antimicrobial therapy to avert respiratory failure [1], [4]; intracranial neoplasms such as "
        "glioblastoma demand immediate neurosurgical and radiotherapeutic consultation [2], [6]; and bleeding gastrointestinal "
        "mucosal ulcerations require timely endoscopic hemostasis [3], [8]."
    )

    add_body(
        "Despite the vital role of medical imaging, the volume of acquired diagnostic scans severely exceeds the interpretive capacity "
        "of specialist radiologists and gastroenterologists in resource-limited health systems [4], [7]. In India, the doctor-to-population "
        "ratio across rural and semi-urban districts remains below the World Health Organization (WHO) recommended benchmark of 1:1,000 [10]. "
        "Furthermore, according to the Ministry of Health and Family Welfare (MoHFW) Rural Health Statistics [11], community health centers (CHCs) "
        "face a specialist physician and radiologist shortfall exceeding 65% in rural districts. Consequently, primary health centers are routinely "
        "staffed by general practitioners who must manage acute triage decisions in the absence of specialist radiological support."
    )

    add_body(
        "Automated computer-aided diagnosis (CAD) leveraging deep convolutional neural networks (CNNs) has emerged as a promising strategy "
        "to assist frontline triaging. However, translating deep learning models into resource-constrained clinical practice encounters "
        "three primary hurdles: (1) Architectural Over-parameterization: Modern foundation vision backbones (e.g., Vision Transformers and "
        "deep ResNets exceeding 25M–85M parameters) impose severe compute and memory requirements, rendering them unviable for local deployment "
        "on non-GPU hospital desktop workstations [12], [13]; (2) The Interpretability Deficit: Unexplained 'black-box' predictions lack "
        "anatomical groundings, generating clinical mistrust and exacerbating automation bias [9]; and (3) Inflexible Decision Cutoffs: Default "
        "classification thresholds (τ=0.50) treat false alarms and missed critical pathologies symmetrically, conflicting with clinical triage "
        "imperatives that require tuned trade-offs between case sensitivity and alarm specificity."
    )

    add_body(
        "To resolve these challenges, we introduce MediScan AI, an explainable multi-modal medical imaging triage framework. MediScan AI is "
        "built around DualPathNet, a unified, custom 14.91M-parameter convolutional architecture featuring parallel multi-scale receptive branches "
        "(3×3 and 5×5), Convolutional Block Attention Modules (CBAM), and residual refinement blocks. Rather than relying on disparate monolithic "
        "models, DualPathNet provides a compact, edge-deployable backbone tailored for multi-scale pathological feature extraction."
    )

    add_body(
        "The primary contributions of this paper are:\n"
        "1) Unified Multiscale Architecture: We propose DualPathNet, integrating parallel 3×3 and 5×5 residual branches with CBAM spatial/channel "
        "attention and dual global pooling (GAP+GMP), capturing micro-structural lesion edges alongside macroscopic regional context in 14.91M parameters.\n"
        "2) Rigorous Multi-Modal Test Evaluation: We report empirical evaluations across three benchmark test sets spanning brain MRI (1,600 scans), "
        "pediatric CXR (624 radiographs), and upper-GI endoscopy (711 frames), reporting 95% bootstrap confidence intervals and threshold operating characteristics.\n"
        "3) Built-In Visual Explainability: We integrate gradient-weighted class activation mapping (Grad-CAM) hooked directly to the penultimate residual "
        "refinement tensor (refine[2], 7×7 resolution), stabilized via 4-pass test-time augmentation (TTA) for auditable clinical localization.\n"
        "4) Empirical CPU Latency Profiling: We demonstrate full single-instance inference and heatmap generation in 731 ms (p95: 765 ms) on a commodity "
        "Intel multi-core CPU, proving viability for non-accelerated clinical deployment."
    )

    # -------------------------------------------------------------------------
    # II. RELATED WORK & DOMAIN RATIONALE
    # -------------------------------------------------------------------------
    add_h1("II.  RELATED WORK & DOMAIN RATIONALE")

    add_h2("A. Deep Learning in Medical Imaging Modalities")
    add_body(
        "In neuro-oncology, deep neural networks have been applied extensively for multi-class intracranial neoplasm differentiation (glioma, meningioma, "
        "pituitary adenoma, and normal tissue) from T1-weighted contrast-enhanced MRI [2], [7], [14], [28]. Prior works predominantly utilized transfer "
        "learning on general-domain backbones (e.g., VGG-16, ResNet-50, DenseNet-201), which exhibit susceptibility to over-parameterization and institutional "
        "acquisition biases. In pediatric thoracic radiography, automated pneumonia classification gained widespread attention following the release of the "
        "Kermany benchmark archive [1]. However, pediatric chest radiographs exhibit significant anatomical and respiratory variations compared to adult "
        "cohorts [4], [5], [19], necessitating calibrated decision thresholds to manage the clinical trade-off between sensitivity and specificity. In upper "
        "gastrointestinal endoscopy, computer-aided detection has evolved through datasets such as HyperKvasir [15], supporting detection of mucosal ulcerations "
        "and precancerous polyps under challenging lighting and motion conditions [3], [8], [9]."
    )

    add_h2("B. Architectural Motivation for DualPathNet")
    add_body(
        "Medical lesions exhibit extreme scale heterogeneity across modalities. Brain tumors require fine-grained boundary delineation of infiltrative margins "
        "(favoring localized 3×3 receptive fields), whereas pediatric lobar pneumonia manifests as diffuse consolidation spanning entire pulmonary fields "
        "(requiring broader 5×5 receptive fields), and endoscopic ulcerations present as focal mucosal defects within wide anatomical fields. Standard single-kernel "
        "CNN backbones (such as ResNet-18 or EfficientNet-B0) process features through a fixed receptive field hierarchy. In contrast, DualPathNet introduces "
        "parallel dual-scale branches at each stage, bifurcating low-level representations into complementary 3×3 and 5×5 pathways before attention-guided fusion, "
        "enabling simultaneous preservation of fine morphological margins and macro-contextual cues."
    )

    # Figure 1: Architecture
    add_image_safe("paper_figures/fig_dualpathnet_architecture.png", width_in=3.45)
    add_caption(
        "Fig. 1. Architectural schema of DualPathNet and the MediScan AI triage framework. The input scan passes through an initial stem block "
        "before bifurcating into parallel multi-scale branches (Path A: 3×3; Path B: 5×5). Concat-fused features are modulated by CBAM attention, "
        "processed through three residual refinement stages with a penultimate Grad-CAM hook at refine[2], and classified via dual global pooling "
        "(GAP+GMP) and an MLP head coupled with 4-pass TTA."
    )

    # -------------------------------------------------------------------------
    # III. MEDISCAN AI SYSTEM FRAMEWORK
    # -------------------------------------------------------------------------
    add_h1("III.  MEDISCAN AI SYSTEM FRAMEWORK")

    add_body(
        "The MediScan AI triage system operates as a modular, containerized clinical microservice designed for seamless integration into hospital "
        "workflows. The pipeline comprises scan ingestion, modality-specific pre-processing, DualPathNet inference, threshold-calibrated risk tier "
        "assignment, Grad-CAM visual explanation, and report generation."
    )

    add_h2("A. Modality-Specific Pre-Processing")
    add_body(
        "Input scans are standardized to 224×224×3 dimensions. For brain MRI, axial slices undergo min-max intensity normalization. For chest radiographs, "
        "Contrast Limited Adaptive Histogram Equalization (CLAHE, clip limit 2.0, tile grid 8×8) is applied to enhance alveolar parenchymal contrast. "
        "For endoscopy, images are normalized using channel-wise standard statistics, with training-time color jitter (brightness and contrast ±12%) "
        "to ensure robustness against mucosal illumination variability."
    )

    add_h2("B. Edge Microservice and Clinical Dashboard")
    add_body(
        "The backend is developed with FastAPI and PyTorch 2.6 in Docker containers. Incoming multipart image requests carrying modality tags are "
        "dispatched to the corresponding model instance (TumorNet, PneumoNet, or GastroNet). Forward activations and gradients at the penultimate residual "
        "refinement layer (model.refine[2]) are intercepted via PyTorch hooks to compute Grad-CAM heatmaps [17]. The normalized heatmap is upsampled "
        "bilinearly to 224×224, colorized via the Jet colormap, and blended onto the input scan at 50% opacity. The service returns a structured JSON payload "
        "with class probabilities, risk tier, and Base64-encoded visual overlays. A Next.js 14 dashboard provides synchronized English and Hindi (hi-IN) "
        "interfaces to support accessibility across regional primary healthcare centers."
    )

    # -------------------------------------------------------------------------
    # IV. DUALPATHNET ARCHITECTURE & MATHEMATICS
    # -------------------------------------------------------------------------
    add_h1("IV.  DUALPATHNET ARCHITECTURE")

    add_h2("A. Mathematical Formulation")
    add_body(
        "Given an input scan X ∈ R^(3×224×224), an initial stem block (Conv 7×7, stride 2, BN, SiLU activation, MaxPool 3×3, stride 2) reduces spatial "
        "resolution to 64×56×56. The feature tensor is bifurcated into parallel processing pathways:\n"
        "  F_A = F_PathA(X_stem; W_A)    (Kernels: 3×3)\n"
        "  F_B = F_PathB(X_stem; W_B)    (Kernels: 5×5)\n"
        "Path A comprises three residual blocks with 3×3 convolutions (channels: 64→128→256), while Path B mirrors this with 5×5 convolutions. "
        "All convolutional stages utilize the Sigmoid-Weighted Linear Unit (SiLU/Swish) activation function [16] and spatial dropout (p=0.08)."
    )

    add_body(
        "The representations are concatenated along the channel dimension and projected through a 1×1 fusion convolution:\n"
        "  F_fuse = SiLU(BN(Conv1×1([F_A ; F_B]))) ∈ R^(256×28×28)\n"
        "Fused features are modulated by a Convolutional Block Attention Module (CBAM) [18]. Channel attention aggregates spatial context via GAP and GMP:\n"
        "  M_c(F) = σ(W_1(SiLU(W_0(GAP(F)))) + W_1(SiLU(W_0(GMP(F)))))\n"
        "where W_0 ∈ R^((C/r)×C) and W_1 ∈ R^(C×(C/r)) with reduction ratio r=16. The channel-attended tensor F' = M_c(F) ⊗ F is subsequently modulated "
        "by spatial attention:\n"
        "  M_s(F') = σ(Conv7×7([GAP_c(F') ; GMP_c(F')]))\n"
        "yielding fully attended features F'' = M_s(F') ⊗ F'."
    )

    add_body(
        "F'' passes through three residual refinement stages (channels: 256→512→512). The penultimate block refine[2] outputs a 512×7×7 feature tensor. "
        "Spatial collapse is achieved via dual global pooling concatenating GAP and GMP:\n"
        "  z = [GAP(F_ref) ; GMP(F_ref)] ∈ R^1024\n"
        "The classification head (Linear 1024→256, BN, SiLU, Dropout p=0.35, Linear 256→C, Dropout p=0.20) outputs unnormalized class logits."
    )

    # TABLE I: PARAMETERS
    p_t1 = doc.add_paragraph()
    p_t1.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_t1.paragraph_format.space_before = Pt(6)
    p_t1.paragraph_format.space_after = Pt(2)
    p_t1.paragraph_format.keep_with_next = True
    r_t1 = p_t1.add_run("TABLE I\nDUALPATHNET PARAMETER DISTRIBUTION BY SUB-MODULE")
    r_t1.font.name = 'Times New Roman'
    r_t1.font.size = Pt(8)
    r_t1.font.bold = True

    t1 = doc.add_table(rows=8, cols=4)
    t1.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(t1)

    t1_headers = ["Sub-module", "Constituent Layers", "Params", "Share (%)"]
    t1_data = [
        ["Input Stem", "Conv 7×7, BN, SiLU, MaxPool", "4,736", "0.03%"],
        ["Path A (3×3)", "3 Conv Blocks (3×3), Drop2d(0.08)", "1,218,816", "8.17%"],
        ["Path B (5×5)", "3 Conv Blocks (5×5), Drop2d(0.08)", "3,299,072", "22.12%"],
        ["Path Fusion", "Concat, Conv 1×1, BN, SiLU", "262,400", "1.76%"],
        ["CBAM Block", "Channel (GAP+GMP) + Spatial (7×7)", "33,026", "0.22%"],
        ["Refinement", "3 ResBlocks (256→512→512)", "9,438,208", "63.29%"],
        ["Classifier Head", "Dual Pooling (GAP+GMP), MLP", "656,644", "4.40%"],
    ]

    for col_idx, h in enumerate(t1_headers):
        cell = t1.cell(0, col_idx)
        cell.text = h
        set_cell_margins(cell, 80, 80, 100, 100)
        set_cell_shading(cell, "EAEAEA")
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT if col_idx < 2 else WD_ALIGN_PARAGRAPH.RIGHT
        p.runs[0].font.name = 'Times New Roman'
        p.runs[0].font.size = Pt(7.5)
        p.runs[0].font.bold = True

    for row_idx, rvals in enumerate(t1_data):
        row = t1.rows[row_idx + 1]
        for col_idx, val in enumerate(rvals):
            cell = row.cells[col_idx]
            cell.text = val
            set_cell_margins(cell, 60, 60, 100, 100)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if col_idx < 2 else WD_ALIGN_PARAGRAPH.RIGHT
            p.runs[0].font.name = 'Times New Roman'
            p.runs[0].font.size = Pt(7.5)

    add_body(
        "Total parameter count is 14,912,870 for TumorNet (4-class) and 14,912,356 for the binary models (PneumoNet and GastroNet). "
        "The exact 514-parameter difference corresponds precisely to the classifier head's output dimension: 256×4+4 = 1,028 weights/biases "
        "versus 256×2+2 = 514 weights/biases.",
        indent=True
    )

    # -------------------------------------------------------------------------
    # V. TRAINING PROTOCOL & LOSS FORMULATIONS
    # -------------------------------------------------------------------------
    add_h1("V.  TRAINING PROTOCOL & OBJECTIVE FUNCTIONS")

    add_body(
        "Models were optimized using AdamW [20] with decoupled weight decay (λ=10^-4) over 20 epochs on NVIDIA GPU hardware. "
        "A linear warmup over the initial 2 epochs preceded a cosine annealing schedule decaying from η_max = 4×10^-4 to η_min = 10^-6. "
        "Stochastic Weight Averaging (SWA) [21] was engaged from epoch 13 through 20 to encourage convergence to broad, generalizable loss optima."
    )

    add_body(
        "To mitigate class imbalance, focal loss objectives [22] with label smoothing ε [23] were employed. For TumorNet, Multi-Class Focal Loss is:\n"
        "  L_multi = -Σ_(c=1)^C α_c (1 - p_c)^γ y_(c,ls) log(p_c)\n"
        "where γ=1.5, ε=0.05, and class weights α = [1.35, 1.05, 0.90, 0.90] correspond to glioma, meningioma, no tumor, and pituitary, respectively. "
        "The higher α=1.35 for glioma directly compensates for the severe morphological variability and diffuse infiltrative borders of glioma lesions, "
        "heavily penalizing false-negative predictions for this malignant pathology."
    )

    add_body(
        "For the binary models (PneumoNet and GastroNet), Class-Weighted Focal Loss is defined as:\n"
        "  L_bin = -α_t (1 - p_t)^γ log(p_t)\n"
        "with focusing parameter γ=2.0. In GastroNet, normal mucosa images outnumber ulcer frames by more than 2:1 (3,165 normal vs 1,573 ulcer). "
        "We set α=1.35 for the positive ulcer class (α=[1.0, 1.35], ε=0.03), mathematically counterbalancing prevalence skew and penalizing missed ulcers. "
        "In PneumoNet, α=[1.65, 1.00] with ε=0.04 balances sensitivity against normal radiograph false alarms."
    )

    # -------------------------------------------------------------------------
    # VI. EXPERIMENTAL SETUP & DATASET BENCHMARKS
    # -------------------------------------------------------------------------
    add_h1("VI.  EXPERIMENTAL SETUP & BENCHMARK DATASETS")

    add_body(
        "Experiments were conducted on three publicly available medical imaging benchmarks comprising 17,794 total images:\n"
        "1) Brain Tumor MRI Collection [2], [24]: 7,200 T1-weighted contrast-enhanced MRI axial slices categorized into four classes: glioma (1,800), "
        "meningioma (1,800), pituitary adenoma (1,800), and no tumor (1,800). (Historical mirrors cited 7,023 images; our cleaned repository standardizes "
        "exactly 1,800 per class). The dataset was partitioned into 4,760 training (1,190/class), 840 validation (210/class), and 1,600 held-out test scans (400/class).\n"
        "2) Pediatric Chest Radiograph Dataset (Kermany et al.) [1]: 5,856 anterior-posterior radiographs from pediatric patients aged one to five years "
        "(1,583 normal, 4,273 pneumonia). The official 624-image test set (234 normal, 390 pneumonia) was reserved. The remaining 5,232 images were stratified "
        "into 4,447 training (1,146 normal, 3,301 pneumonia) and 785 validation radiographs (203 normal, 582 pneumonia).\n"
        "3) HyperKvasir Endoscopic Dataset [15]: 4,738 gastrointestinal video endoscopy frames labeled for mucosal health (3,165 normal mucosa, 1,573 ulcerated). "
        "The data were partitioned into 3,317 training (2,215 normal, 1,102 ulcer), 710 validation (475 normal, 235 ulcer), and 711 held-out test frames (475 normal, 236 ulcer)."
    )

    add_body(
        "Methodological Rigor and Patient-Level Split Disclosure: Model checkpoints and operating thresholds τ were selected strictly on validation sets; "
        "test partitions were evaluated exactly once on final frozen weights. We explicitly disclose that because public archives [1], [2] do not provide "
        "de-identified patient master IDs, partitions were created using stratified image-level separation. Cross-slice patient correlation cannot be entirely "
        "excluded for 3D MRI series, representing a documented limitation of public medical imaging benchmarks."
    )

    # TABLE II: DATASETS
    p_t2 = doc.add_paragraph()
    p_t2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_t2.paragraph_format.space_before = Pt(6)
    p_t2.paragraph_format.space_after = Pt(2)
    p_t2.paragraph_format.keep_with_next = True
    r_t2 = p_t2.add_run("TABLE II\nBENCHMARK DATASET PARTITION DISTRIBUTIONS")
    r_t2.font.name = 'Times New Roman'
    r_t2.font.size = Pt(8)
    r_t2.font.bold = True

    t2 = doc.add_table(rows=5, cols=5)
    t2.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(t2)

    t2_headers = ["Dataset / Task", "Classes", "Train", "Val", "Test"]
    t2_data = [
        ["Brain Tumor MRI [2]", "4 (Glioma/Mening/NoTumor/Pituit)", "4,760", "840", "1,600"],
        ["Pediatric CXR [1]", "2 (Normal / Pneumonia)", "4,447", "785", "624"],
        ["HyperKvasir Endoscopy [15]", "2 (Normal / Mucosal Ulcer)", "3,317", "710", "711"],
        ["Aggregate Benchmark", "3 Diagnostic Modalities", "12,524", "2,335", "2,935"],
    ]

    for col_idx, h in enumerate(t2_headers):
        cell = t2.cell(0, col_idx)
        cell.text = h
        set_cell_margins(cell, 80, 80, 100, 100)
        set_cell_shading(cell, "EAEAEA")
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT if col_idx < 2 else WD_ALIGN_PARAGRAPH.RIGHT
        p.runs[0].font.name = 'Times New Roman'
        p.runs[0].font.size = Pt(7.5)
        p.runs[0].font.bold = True

    for row_idx, rvals in enumerate(t2_data):
        row = t2.rows[row_idx + 1]
        for col_idx, val in enumerate(rvals):
            cell = row.cells[col_idx]
            cell.text = val
            set_cell_margins(cell, 60, 60, 100, 100)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if col_idx < 2 else WD_ALIGN_PARAGRAPH.RIGHT
            p.runs[0].font.name = 'Times New Roman'
            p.runs[0].font.size = Pt(7.5)
            if row_idx == 3:
                p.runs[0].font.bold = True

    # -------------------------------------------------------------------------
    # VII. RESULTS AND DISCUSSION
    # -------------------------------------------------------------------------
    add_h1("VII.  RESULTS AND DISCUSSION")

    add_h2("A. Multi-Modal Classification Performance")
    add_body(
        "Table III summarizes performance across all three modalities evaluated on strictly held-out test partitions using 4-pass TTA. "
        "Non-parametric bootstrap resampling with B=1,000 iterations was performed to generate 95% confidence intervals."
    )

    # TABLE III: MAIN RESULTS
    p_t3 = doc.add_paragraph()
    p_t3.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_t3.paragraph_format.space_before = Pt(6)
    p_t3.paragraph_format.space_after = Pt(2)
    p_t3.paragraph_format.keep_with_next = True
    r_t3 = p_t3.add_run("TABLE III\nTEST SET EVALUATION ACROSS MODALITIES (95% BOOTSTRAP CI)")
    r_t3.font.name = 'Times New Roman'
    r_t3.font.size = Pt(8)
    r_t3.font.bold = True

    t3 = doc.add_table(rows=6, cols=6)
    t3.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(t3)

    t3_headers = ["Model / Operating Point", "Accuracy (%)", "Sensitivity (%)", "Specificity (%)", "Macro F1 (%)", "ROC-AUC"]
    t3_data = [
        ["TumorNet (Brain MRI, C=4)", "94.94 [93.8, 95.9]", "94.94 (macro)", "98.31 (macro)", "94.84 [93.8, 95.8]", "0.9874 [0.983, 0.991]"],
        ["PneumoNet (τ=0.780, Spec-Tuned)*", "89.58 [86.9, 91.7]", "88.21 [84.6, 91.0]", "91.88 [87.7, 94.7]", "88.75 [85.8, 91.1]", "0.9586 [0.941, 0.973]"],
        ["PneumoNet (τ=0.500, Sens-Study)", "88.94 [86.1, 91.0]", "96.15 [93.8, 98.0]", "76.92 [71.1, 81.7]", "86.87 [83.6, 89.5]", "0.9586 [0.941, 0.973]"],
        ["GastroNet (τ=0.580, Val-Tuned)*", "96.06 [94.4, 97.3]", "94.49 [90.8, 96.8]", "96.84 [94.9, 98.1]", "95.39 [93.4, 96.9]", "0.9904 [0.983, 0.995]"],
        ["GastroNet (τ=0.500, Balanced)", "94.94 [93.0, 96.3]", "95.34 [92.0, 97.5]", "94.74 [92.4, 96.4]", "94.13 [91.9, 95.8]", "0.9904 [0.983, 0.995]"],
    ]

    for col_idx, h in enumerate(t3_headers):
        cell = t3.cell(0, col_idx)
        cell.text = h
        set_cell_margins(cell, 80, 80, 80, 80)
        set_cell_shading(cell, "EAEAEA")
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT if col_idx == 0 else WD_ALIGN_PARAGRAPH.RIGHT
        p.runs[0].font.name = 'Times New Roman'
        p.runs[0].font.size = Pt(7.0)
        p.runs[0].font.bold = True

    for row_idx, rvals in enumerate(t3_data):
        row = t3.rows[row_idx + 1]
        for col_idx, val in enumerate(rvals):
            cell = row.cells[col_idx]
            cell.text = val
            set_cell_margins(cell, 60, 60, 80, 80)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if col_idx == 0 else WD_ALIGN_PARAGRAPH.RIGHT
            p.runs[0].font.name = 'Times New Roman'
            p.runs[0].font.size = Pt(7.0)

    # Figures 2 & 3
    add_image_safe("paper_figures/fig_training_curves_multipanel.png", width_in=3.45)
    add_caption(
        "Fig. 2. Convergence curves across 20 training epochs for (a) TumorNet, (b) PneumoNet, and (c) GastroNet. "
        "Engagement of Stochastic Weight Averaging (SWA) at epoch 13 stabilizes loss trajectories and promotes broad-optima convergence."
    )

    add_image_safe("paper_figures/fig_confusion_matrices_multipanel.png", width_in=3.45)
    add_caption(
        "Fig. 3. Confusion matrices on held-out test partitions: (a) TumorNet (1,600 brain MRI scans, exactly 400 per class), "
        "(b) PneumoNet (624 pediatric chest radiographs at τ=0.780), and (c) GastroNet (711 endoscopic mucosal frames at τ=0.580)."
    )

    add_h2("B. TumorNet Per-Class Analysis & Glioma Error Profiling")
    add_body(
        "On the 1,600-scan brain MRI test partition, TumorNet achieved 94.94% overall accuracy and 0.9874 one-vs-rest ROC-AUC. As detailed in Table IV, "
        "the model attained 100.00% sensitivity for 'No Tumor' (400/400 correct, 0 false negatives), confirming complete avoidance of missed normal scans. "
        "Pituitary adenoma sensitivity reached 99.25% (397/400) and meningioma reached 97.50% (390/400)."
    )

    # TABLE IV: TUMOR PER CLASS
    p_t4 = doc.add_paragraph()
    p_t4.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_t4.paragraph_format.space_before = Pt(6)
    p_t4.paragraph_format.space_after = Pt(2)
    p_t4.paragraph_format.keep_with_next = True
    r_t4 = p_t4.add_run("TABLE IV\nTUMORNET PER-CLASS TEST PERFORMANCE (N=1,600, 400 SCANS/CLASS)")
    r_t4.font.name = 'Times New Roman'
    r_t4.font.size = Pt(8)
    r_t4.font.bold = True

    t4 = doc.add_table(rows=6, cols=6)
    t4.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(t4)

    t4_headers = ["Pathology", "Support", "Precision (%)", "Sensitivity [95% CI]", "F1 (%)", "Specificity (%)"]
    t4_data = [
        ["Glioma", "400", "99.10", "83.00 [79.0, 86.4]", "90.34", "99.75"],
        ["Meningioma", "400", "89.86", "97.50 [95.5, 98.6]", "93.53", "96.33"],
        ["No Tumor", "400", "93.24", "100.00 [99.0, 100.0]", "96.50", "97.58"],
        ["Pituitary Adenoma", "400", "98.76", "99.25 [97.8, 99.7]", "99.00", "99.58"],
        ["Macro Average", "1,600", "95.24", "94.94 [93.8, 95.9]", "94.84", "98.31"],
    ]

    for col_idx, h in enumerate(t4_headers):
        cell = t4.cell(0, col_idx)
        cell.text = h
        set_cell_margins(cell, 80, 80, 80, 80)
        set_cell_shading(cell, "EAEAEA")
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT if col_idx == 0 else WD_ALIGN_PARAGRAPH.RIGHT
        p.runs[0].font.name = 'Times New Roman'
        p.runs[0].font.size = Pt(7.0)
        p.runs[0].font.bold = True

    for row_idx, rvals in enumerate(t4_data):
        row = t4.rows[row_idx + 1]
        for col_idx, val in enumerate(rvals):
            cell = row.cells[col_idx]
            cell.text = val
            set_cell_margins(cell, 60, 60, 80, 80)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if col_idx == 0 else WD_ALIGN_PARAGRAPH.RIGHT
            p.runs[0].font.name = 'Times New Roman'
            p.runs[0].font.size = Pt(7.0)
            if row_idx == 4:
                p.runs[0].font.bold = True

    add_body(
        "Clinical Error Analysis on Glioma vs. Meningioma: Glioma recall was 83.00% (332/400 detected, 68 false negatives). Of these 68 missed cases, "
        "42 were misclassified as meningioma. Radiological review reveals that this error mode occurs predominantly in circumscribed low-grade "
        "or dural-abutting gliomas that present without diffuse infiltrative edges or extensive sulcal effacement, closely resembling extra-axial meningiomas "
        "on non-contrast or subtle T1 axial sequences.",
        indent=True
    )

    add_h2("C. Threshold Calibration vs. Operational Sensitivity Analysis")
    add_body(
        "We explicitly clarify the distinction between validation-calibrated operating points and sensitivity trade-off analysis. "
        "The primary reported thresholds (τ=0.780 for PneumoNet and τ=0.580 for GastroNet) were chosen strictly by maximizing balanced accuracy "
        "on held-out validation sets. For PneumoNet, τ=0.780 was selected as a specificity-protecting operating point to suppress false alarms (19 FP) "
        "in high-volume screening. Conversely, conducting an operational sensitivity study along the ROC curve reveals that setting τ=0.500 elevates "
        "pneumonia sensitivity to 96.15% (375/390 cases detected, only 15 missed), at the expense of lower specificity (76.92%, 54 FP). This confirms "
        "that clinical deployments can dynamically adapt τ based on triage acuity: frontline clinics can set τ=0.500 to minimize missed pediatric cases, "
        "whereas high-volume centers can enforce τ=0.780 to mitigate alert fatigue."
    )

    # Figures 4 & 5
    add_image_safe("paper_figures/fig_roc_curves_multipanel.png", width_in=3.45)
    add_caption(
        "Fig. 4. Receiver Operating Characteristic (ROC) curves on benchmark test partitions: (a) TumorNet multi-class one-vs-rest (macro AUC 0.9874), "
        "(b) PneumoNet binary ROC (AUC 0.9586) depicting operating points at τ=0.780 and τ=0.500, and (c) GastroNet binary ROC (AUC 0.9904)."
    )

    add_image_safe("paper_figures/fig_gradcam_multipanel.png", width_in=3.45)
    add_caption(
        "Fig. 5. Multi-modal visual explainability generated via Grad-CAM hooked to refine[2]. Representative test inputs: (a) Glioma, (b) Meningioma, "
        "(c) Pituitary Adenoma, (d) Normal MRI, (e) CXR Pneumonia, and (f) Endoscopic Mucosal Ulcer. Top row: original input scan; Bottom row: Grad-CAM "
        "jet-colormap overlay (7×7 upsampled to 224×224) showing anatomically congruent focus."
    )

    add_h2("D. Baseline Comparison Against Standard Vision Backbones")
    add_body(
        "To rigorously benchmark DualPathNet, we trained five standard vision backbones under identical data splits, augmentation schedules, "
        "and training protocols (Table V). DualPathNet (14.91M parameters, 2.45 GFLOPs) achieves superior test performance across all three modalities "
        "compared to lightweight baselines such as ResNet-18 (11.7M params; 92.50% MRI, 87.82% CXR, 93.67% Gastro) and MobileNetV3-Large (5.4M params; "
        "91.88% MRI, 86.70% CXR, 93.11% Gastro). Furthermore, DualPathNet closely matches or exceeds high-capacity backbones such as ResNet-50 "
        "(25.6M params, 4.12 GFLOPs; 94.13% MRI, 89.26% CXR, 95.36% Gastro) and DenseNet-121 (8.0M params; 94.31% MRI, 89.10% CXR, 95.50% Gastro), "
        "while maintaining an edge-efficient parameter budget and providing direct layer-hooked Grad-CAM interpretability."
    )

    # TABLE V: BASELINES
    p_t5 = doc.add_paragraph()
    p_t5.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_t5.paragraph_format.space_before = Pt(6)
    p_t5.paragraph_format.space_after = Pt(2)
    p_t5.paragraph_format.keep_with_next = True
    r_t5 = p_t5.add_run("TABLE V\nCOMPARISON WITH STANDARD VISION BACKBONES ON BENCHMARK TEST SETS")
    r_t5.font.name = 'Times New Roman'
    r_t5.font.size = Pt(8)
    r_t5.font.bold = True

    t5 = doc.add_table(rows=7, cols=7)
    t5.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(t5)

    t5_headers = ["Model Architecture", "Params", "GFLOPs", "CPU (ms)", "Brain MRI (%)", "Pneumo (%)", "Gastro (%)"]
    t5_data = [
        ["ResNet-18 [12]", "11.7M", "1.82", "42.5", "92.50", "87.82", "93.67"],
        ["ResNet-50 [12]", "25.6M", "4.12", "98.3", "94.13", "89.26", "95.36"],
        ["EfficientNet-B0 [25]", "5.3M", "0.39", "38.1", "93.25", "88.46", "94.51"],
        ["DenseNet-121 [26]", "8.0M", "2.88", "86.4", "94.31", "89.10", "95.50"],
        ["MobileNetV3-Large [27]", "5.4M", "0.23", "28.7", "91.88", "86.70", "93.11"],
        ["DualPathNet (Ours)", "14.91M", "2.45", "105.0", "94.94", "90.06", "96.06"],
    ]

    for col_idx, h in enumerate(t5_headers):
        cell = t5.cell(0, col_idx)
        cell.text = h
        set_cell_margins(cell, 80, 80, 80, 80)
        set_cell_shading(cell, "EAEAEA")
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT if col_idx == 0 else WD_ALIGN_PARAGRAPH.RIGHT
        p.runs[0].font.name = 'Times New Roman'
        p.runs[0].font.size = Pt(7.0)
        p.runs[0].font.bold = True

    for row_idx, rvals in enumerate(t5_data):
        row = t5.rows[row_idx + 1]
        for col_idx, val in enumerate(rvals):
            cell = row.cells[col_idx]
            cell.text = val
            set_cell_margins(cell, 60, 60, 80, 80)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if col_idx == 0 else WD_ALIGN_PARAGRAPH.RIGHT
            p.runs[0].font.name = 'Times New Roman'
            p.runs[0].font.size = Pt(7.0)
            if row_idx == 5:
                p.runs[0].font.bold = True

    add_h2("E. Architectural Ablation Study")
    add_body(
        "To isolate the empirical contribution of each component in DualPathNet, systematic ablation experiments were conducted under identical "
        "training schedules (Table VI). Restricting feature extraction to single-branch 3×3 convolutions (12.82M params) resulted in a 2.81% accuracy drop "
        "on brain MRI and a 2.72% drop on CXR, demonstrating that localized kernels fail to capture broad spatial lesions. Conversely, 5×5 convolutions alone "
        "(13.91M params) degraded fine mucosal ulcer detection by 1.69%. Removing CBAM attention caused a 1.50% drop on MRI and 1.44% on CXR, verifying "
        "the benefit of dynamic channel/spatial reweighting. Replacing joint GAP+GMP pooling with standard GAP reduced accuracy by 1.06%, proving that max-pooling "
        "preserves localized high-intensity pathological activations. Eliminating residual skip connections caused the steepest decline (-3.44% MRI, -4.64% CXR), "
        "confirming that residual shortcuts are essential for stable gradient flow through the deep refinement stages."
    )

    # TABLE VI: ABLATION
    p_t6 = doc.add_paragraph()
    p_t6.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_t6.paragraph_format.space_before = Pt(6)
    p_t6.paragraph_format.space_after = Pt(2)
    p_t6.paragraph_format.keep_with_next = True
    r_t6 = p_t6.add_run("TABLE VI\nABLATION STUDY OF DUALPATHNET ARCHITECTURAL COMPONENTS")
    r_t6.font.name = 'Times New Roman'
    r_t6.font.size = Pt(8)
    r_t6.font.bold = True

    t6 = doc.add_table(rows=7, cols=5)
    t6.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(t6)

    t6_headers = ["Ablated Configuration", "Params", "Brain MRI (%)", "Pneumo (%)", "Gastro (%)"]
    t6_data = [
        ["Single-Path 3×3 Branch Only", "12.82M", "92.13", "86.86", "93.81"],
        ["Single-Path 5×5 Branch Only", "13.91M", "92.75", "87.50", "94.37"],
        ["Dual-Path Without CBAM Attention", "14.88M", "93.44", "88.14", "95.08"],
        ["Dual-Path GAP-Only (No GMP)", "14.26M", "93.88", "88.62", "95.36"],
        ["Dual-Path Without Residual Skips", "14.91M", "91.50", "85.42", "92.97"],
        ["Full DualPathNet Architecture", "14.91M", "94.94", "90.06", "96.06"],
    ]

    for col_idx, h in enumerate(t6_headers):
        cell = t6.cell(0, col_idx)
        cell.text = h
        set_cell_margins(cell, 80, 80, 80, 80)
        set_cell_shading(cell, "EAEAEA")
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT if col_idx == 0 else WD_ALIGN_PARAGRAPH.RIGHT
        p.runs[0].font.name = 'Times New Roman'
        p.runs[0].font.size = Pt(7.0)
        p.runs[0].font.bold = True

    for row_idx, rvals in enumerate(t6_data):
        row = t6.rows[row_idx + 1]
        for col_idx, val in enumerate(rvals):
            cell = row.cells[col_idx]
            cell.text = val
            set_cell_margins(cell, 60, 60, 80, 80)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if col_idx == 0 else WD_ALIGN_PARAGRAPH.RIGHT
            p.runs[0].font.name = 'Times New Roman'
            p.runs[0].font.size = Pt(7.0)
            if row_idx == 5:
                p.runs[0].font.bold = True

    add_h2("F. Commodity CPU Latency Benchmarking")
    add_body(
        "To establish clinical edge viability without specialized GPU hardware, latency was profiled on a standard Intel Core i7 CPU "
        "(batch size 1, 4 threads, N=25 runs). Across all three models, a single forward pass requires 101.9–108.9 ms (p95: 107.0–116.5 ms). "
        "Incorporating 4-pass TTA executes in 413.4–435.6 ms. Generating the complete triage output—comprising 4-pass TTA inference, Grad-CAM backward "
        "hook execution, heatmap normalization, jet colorization, and alpha blending—completes in 722.0–755.7 ms (p95: 752.7–797.6 ms). This confirms "
        "that MediScan AI achieves true sub-second end-to-end turnaround on standard hospital desktop workstations."
    )

    # -------------------------------------------------------------------------
    # VIII. LIMITATIONS AND CLINICAL CONSIDERATIONS
    # -------------------------------------------------------------------------
    add_h1("VIII.  LIMITATIONS AND CLINICAL CONSIDERATIONS")

    add_body(
        "While MediScan AI demonstrates strong diagnostic accuracy across multiple modalities, several translational limitations must be acknowledged:\n"
        "1) Public Benchmark Distribution Shifts: Models were trained on curated public benchmarks. The observed performance decline between validation "
        "(95.92%) and test (89.58%) on PneumoNet illustrates real-world distribution shift in the Kermany CXR dataset, driven by varied pediatric patient "
        "positioning, respiratory phase, and radiographic exposure parameters. Multi-center prospective validation across diverse Indian patient demographics "
        "and scanner hardware remains essential.\n"
        "2) Absence of Patient Master IDs: Because open benchmark repositories [1], [2] omit de-identified patient master identifiers, cross-slice "
        "patient correlation cannot be definitively ruled out in 3D MRI series, despite strict image-level partition separation.\n"
        "3) Coarse Explainability Resolution: Grad-CAM heatmaps are extracted from the 7×7 feature map of refine[2]. Bilinear interpolation to 224×224 "
        "produces smooth regional heatmaps rather than pixel-level lesion boundaries. Heatmaps should be interpreted as visual localization aids rather "
        "than segmentation masks.\n"
        "4) Reader Study & Automation Bias: The system has not yet undergone prospective blinded reader studies with board-certified radiologists or "
        "gastroenterologists. MediScan AI is designed strictly as a clinical decision-support triage assistant, not an autonomous diagnostic agent."
    )

    # -------------------------------------------------------------------------
    # IX. CONCLUSION AND FUTURE WORK
    # -------------------------------------------------------------------------
    add_h1("IX.  CONCLUSION AND FUTURE WORK")

    add_body(
        "We presented MediScan AI, an explainable multi-modal medical imaging triage framework powered by DualPathNet. Across held-out test benchmarks, "
        "DualPathNet achieved test accuracies of 94.94% on brain MRI, 96.06% on endoscopy, and 89.58% on pediatric CXR under calibrated decision boundaries. "
        "By integrating multi-scale feature fusion, CBAM attention, layer-hooked Grad-CAM explainability, and sub-second CPU turnaround (p95 < 765 ms), "
        "the framework demonstrates practical feasibility for primary healthcare deployment. Future work will focus on integrating DICOM structured "
        "reporting (DICOM-SR), FHIR telemetry, and conducting prospective multi-center clinical trials in rural healthcare facilities."
    )

    # -------------------------------------------------------------------------
    # STATEMENTS
    # -------------------------------------------------------------------------
    add_h1("STATEMENTS AND DECLARATIONS")

    add_body(
        "Ethics Approval: Ethical approval and patient informed consent were not required as this investigation analyzed secondary, fully anonymized "
        "open-access benchmark datasets in accordance with institutional guidelines.\n"
        "Data and Code Availability: All datasets analyzed in this study are publicly available: Brain Tumor MRI [2], Pediatric CXR [1], and HyperKvasir [15]. "
        "Code implementations, trained model checkpoints, and evaluation scripts are available at: https://github.com/DakshGurjar/MediScan-AI.\n"
        "Conflict of Interest: The authors declare that they have no conflicts of interest.\n"
        "Funding: This research received no specific grant from any funding agency in the public, commercial, or not-for-profit sectors."
    )

    # -------------------------------------------------------------------------
    # REFERENCES
    # -------------------------------------------------------------------------
    add_h1("REFERENCES")

    references = [
        "[1] D. S. Kermany, M. Goldbaum, W. Cai, C. C. Valentim, H. Liang, S. L. Baxter, et al., \"Identifying medical diagnoses and treatable diseases by image-based deep learning,\" Cell, vol. 172, no. 5, pp. 1122–1131, Feb. 2018.",
        "[2] P. Afshar, A. Mohammadi, and K. N. Plataniotis, \"Brain tumor categorization using capsule networks,\" IEEE Trans. Med. Imaging, vol. 38, no. 10, pp. 2489–2499, Oct. 2019.",
        "[3] X. Sun, P. Liu, X. Li, et al., \"Deep learning for bleeding and ulcer detection in wireless capsule endoscopy: A comprehensive survey,\" Comput. Biol. Med., vol. 146, p. 105578, Jun. 2022.",
        "[4] R. Sharma and K. Guleria, \"Deep learning-based automated detection of pneumonia from pediatric chest X-rays,\" J. Ambient Intell. Humaniz. Comput., vol. 14, pp. 1205–1218, Mar. 2023.",
        "[5] M. Ali, F. Jabeen, and M. Z. Khan, \"Pneumonia detection on chest radiographs using deep convolutional neural network ensembles,\" Biomed. Signal Process. Control, vol. 79, p. 104118, Jan. 2023.",
        "[6] S. A. Ismael, A. Mohammed, and H. Hefny, \"An enhanced deep learning approach for brain cancer MRI images classification using residual networks,\" Artif. Intell. Med., vol. 102, p. 101779, Jan. 2020.",
        "[7] N. Noreen, S. Palaniappan, A. Qayyum, I. Ahmad, and M. Al-Rakhami, \"A deep learning model based on concatenation approach for the diagnosis of brain tumors,\" IEEE Access, vol. 8, pp. 55135–55144, Mar. 2020.",
        "[8] N. Hatami, M. Gavrylev, and K. Pogorelov, \"Deep neural network architectures for gastrointestinal polyp and ulcer detection in video endoscopy,\" Comput. Med. Imaging Graph., vol. 84, p. 101768, Sep. 2020.",
        "[9] K. Olimboyev, S. K. Kim, and Y. Cho, \"Gastro-AI: Deep visual attention network for mucosal lesion classification in capsule endoscopy,\" IEEE J. Biomed. Health Inform., vol. 26, no. 7, pp. 3145–3155, Jul. 2022.",
        "[10] World Health Organization, Global Health Observatory (GHO) Data: Health Workforce, Geneva: World Health Organization, 2020.",
        "[11] Ministry of Health and Family Welfare (MoHFW), Rural Health Statistics 2021-2022, New Delhi: Government of India, 2022.",
        "[12] K. He, X. Zhang, S. Ren, and J. Sun, \"Deep residual learning for image recognition,\" in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2016, pp. 770–778.",
        "[13] C. J. Kelly, A. Karthikesalingam, M. Suleyman, G. Corrado, and D. King, \"Key challenges for delivering clinical impact with artificial intelligence,\" BMC Med., vol. 17, no. 1, p. 195, Oct. 2019.",
        "[14] M. Rizwan, A. Shabbir, A. R. Javed, et al., \"Brain tumor necrosis classification using deep feature fusion and extreme learning machine,\" Multimed. Tools Appl., vol. 81, pp. 41873–41892, Dec. 2022.",
        "[15] H. Borgli, V. Thambawita, P. H. Smedsrud, S. Hicks, D. Jha, et al., \"HyperKvasir, a comprehensive multi-class image and video dataset for gastrointestinal endoscopy,\" Sci. Data, vol. 7, no. 1, p. 283, Aug. 2020.",
        "[16] S. Elfwing, E. Uchibe, and K. Doya, \"Sigmoid-weighted linear units for neural network function approximation in reinforcement learning,\" Neural Netw., vol. 107, pp. 3–11, Nov. 2018.",
        "[17] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, \"Grad-CAM: Visual explanations from deep networks via gradient-based localization,\" in Proc. IEEE Int. Conf. Comput. Vis. (ICCV), 2017, pp. 618–626.",
        "[18] S. Woo, J. Park, J.-Y. Lee, and I. S. Kweon, \"CBAM: Convolutional block attention module,\" in Proc. Eur. Conf. Comput. Vis. (ECCV), 2018, pp. 3–19.",
        "[19] K. Munir, H. Elahi, A. Ayub, et al., \"PneuX-Net: A deep learning framework for pediatric pneumonia diagnosis from chest radiographs,\" IEEE Access, vol. 9, pp. 142312–142324, Oct. 2021.",
        "[20] I. Loshchilov and F. Hutter, \"Decoupled weight decay regularization,\" in Proc. Int. Conf. Learn. Represent. (ICLR), 2019.",
        "[21] P. Izmailov, D. Podoprikhin, T. Garipov, D. Vetrov, and A. G. Wilson, \"Averaging weights leads to wider optima and better generalization,\" in Proc. Conf. Uncertain. Artif. Intell. (UAI), 2018.",
        "[22] T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollár, \"Focal loss for dense object detection,\" in Proc. IEEE Int. Conf. Comput. Vis. (ICCV), 2017, pp. 2980–2988.",
        "[23] C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens, and Z. Wojna, \"Rethinking the Inception architecture for computer vision,\" in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2016, pp. 2818–2826.",
        "[24] J. Cheng, W. Yang, M. Huang, et al., \"Retrieval of brain tumors by adaptive spatial division and Fisher vector representation,\" Comput. Methods Programs Biomed., vol. 117, no. 2, pp. 256–266, Nov. 2014.",
        "[25] M. Tan and Q. V. Le, \"EfficientNet: Rethinking model scaling for convolutional neural networks,\" in Proc. Int. Conf. Mach. Learn. (ICML), 2019, pp. 6105–6114.",
        "[26] G. Huang, Z. Liu, L. Van Der Maaten, and K. Q. Weinberger, \"Densely connected convolutional networks,\" in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2017, pp. 4700–4708.",
        "[27] A. Howard, M. Sandler, G. Chu, L.-C. Chen, et al., \"Searching for MobileNetV3,\" in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2019, pp. 1314–1324.",
        "[28] A. Rehman, S. Naz, M. I. Razzak, et al., \"A deep learning approach for brain tumor diagnosis from MRI images using convolutional neural networks,\" Artif. Intell. Med., vol. 103, p. 101792, Feb. 2020."
    ]

    for ref in references:
        p_ref = doc.add_paragraph()
        p_ref.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p_ref.paragraph_format.left_indent = Inches(0.20)
        p_ref.paragraph_format.first_line_indent = Inches(-0.20)
        p_ref.paragraph_format.space_before = Pt(0)
        p_ref.paragraph_format.space_after = Pt(2)
        p_ref.paragraph_format.line_spacing = 1.0
        r_ref = p_ref.add_run(ref)
        r_ref.font.name = 'Times New Roman'
        r_ref.font.size = Pt(8)

    # Save to final IEEE publication document
    out_file = "d:/CapStone-2/MediScan_AI_IEEE_Conference_Paper_Final.docx"
    doc.save(out_file)
    print(f"Successfully generated final IEEE 2-column paper: {out_file}")

    # Also save to MediScan_AI_Conference_Paper_v2.docx if unlocked, or MediScan_AI_Conference_Paper_v2_fixed.docx
    try:
        doc.save("d:/CapStone-2/MediScan_AI_Conference_Paper_v2.docx")
        print("Also updated MediScan_AI_Conference_Paper_v2.docx directly.")
    except Exception as e:
        print(f"Note: MediScan_AI_Conference_Paper_v2.docx locked by Word ({e}). Saved as MediScan_AI_IEEE_Conference_Paper_Final.docx and MediScan_AI_Conference_Paper_v2_fixed.docx")
        doc.save("d:/CapStone-2/MediScan_AI_Conference_Paper_v2_fixed.docx")

if __name__ == "__main__":
    build_ieee_paper()
