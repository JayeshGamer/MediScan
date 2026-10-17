import os
import shutil
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn, nsdecls

def set_cell_margins(cell, top=80, bottom=80, left=100, right=100):
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
        f'  <w:insideH w:val="single" w:sz="4" w:space="0" w:color="DDDDDD"/>\n'
        f'  <w:left w:val="none"/>\n'
        f'  <w:right w:val="none"/>\n'
        f'  <w:insideV w:val="none"/>\n'
        f'</w:tblBorders>'
    )
    tblPr.append(borders)

def build_paper():
    doc = docx.Document()

    # Base style
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

    # Title
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_before = Pt(0)
    p_title.paragraph_format.space_after = Pt(8)
    r_title = p_title.add_run("MediScan: A Multiscale Dual-Path Residual Architecture with Attention Refinement for Multi-Disease Clinical Triage and Explainable Diagnosis")
    r_title.font.name = 'Times New Roman'
    r_title.font.size = Pt(22)
    r_title.font.bold = True

    # Authors
    p_auth = doc.add_paragraph()
    p_auth.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_auth.paragraph_format.space_before = Pt(0)
    p_auth.paragraph_format.space_after = Pt(2)
    r_auth = p_auth.add_run("Daksh Gurjar, Student Member, IEEE, Aditya Yashovardhan, and Jayesh Mandloi")
    r_auth.font.name = 'Times New Roman'
    r_auth.font.size = Pt(11)

    # Affiliation
    p_aff = doc.add_paragraph()
    p_aff.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_aff.paragraph_format.space_before = Pt(0)
    p_aff.paragraph_format.space_after = Pt(2)
    r_aff = p_aff.add_run("Department of Computer Engineering, Mukesh Patel School of Technology Management & Engineering (MPSTME)\nSVKM’s NMIMS, Shirpur 425405, Maharashtra, India\n{daksh.gurjar, aditya.yashovardhan, jayesh.mandloi}@nmims.edu")
    r_aff.font.name = 'Times New Roman'
    r_aff.font.size = Pt(9.5)
    r_aff.font.italic = True

    # Notice
    p_not = doc.add_paragraph()
    p_not.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_not.paragraph_format.space_before = Pt(4)
    p_not.paragraph_format.space_after = Pt(10)
    r_not = p_not.add_run("This work has been submitted to the IEEE for possible publication. Copyright may be transferred without notice.")
    r_not.font.name = 'Times New Roman'
    r_not.font.size = Pt(8)
    r_not.font.italic = True

    # Abstract Box
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
        "Access to timely radiological and endoscopic interpretation is severely restricted in resource-constrained primary healthcare "
        "facilities, where specialist-to-patient availability remains drastically below international standards. We present MediScan AI, "
        "an end-to-end multi-modal deep learning triage framework evaluated across three distinct clinical imaging modalities: brain magnetic "
        "resonance imaging (MRI), pediatric chest radiography (CXR), and upper gastrointestinal (GI) endoscopy. At the core of the framework "
        "is DualPathNet, a custom 14.91M-parameter convolutional architecture combining parallel 3×3 and 5×5 multi-scale receptive paths, "
        "Convolutional Block Attention Modules (CBAM), deep residual refinement, and dual global pooling (GAP+GMP). On strictly held-out "
        "benchmark test sets evaluated with 4-pass test-time augmentation (TTA), the models achieve test accuracies of 94.94% [95% CI: 93.8–95.9%] "
        "across four classes on brain MRI (N=1,600 scans, macro-F1: 94.84%, OvR ROC-AUC: 0.9874), 96.06% [95% CI: 94.4–97.3%] on upper-GI endoscopy "
        "(N=711 frames, macro-F1: 95.39%, ROC-AUC: 0.9904), and 89.58% [95% CI: 86.9–91.7%] on pediatric CXR (N=624 radiographs, ROC-AUC: 0.9586) "
        "under a validation-calibrated specificity-protecting decision threshold (τ=0.780, yielding 91.88% specificity, 88.21% sensitivity, and 94.77% precision; "
        "operating at τ=0.500 shifts performance to 96.15% sensitivity and 76.92% specificity for frontline screening). Visual interpretability is "
        "provided via gradient-weighted class activation mapping (Grad-CAM) hooked directly to the penultimate refinement block (refine[2], 7×7 feature map). "
        "Latency profiling on a commodity multi-core CPU demonstrates single-scan inference in 105 ms (p95: 112 ms) and complete 4-pass TTA with "
        "Grad-CAM generation in 731 ms (p95: 765 ms), verifying sub-second clinical triage without requiring GPU acceleration. MediScan AI establishes "
        "a computationally efficient, clinically auditable paradigm for point-of-care medical decision support."
    )
    r_abst = p_abs.add_run(abs_text)
    r_abst.font.name = 'Times New Roman'
    r_abst.font.size = Pt(9)
    r_abst.font.bold = True

    # Index Terms
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

    def add_h1(text):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(11)
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

    def add_eq(eq_text, eq_num):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        p.paragraph_format.space_before = Pt(3)
        p.paragraph_format.space_after = Pt(3)
        r1 = p.add_run(f"\t{eq_text}\t")
        r1.font.name = 'Times New Roman'
        r1.font.size = Pt(9.5)
        r1.font.italic = True
        r2 = p.add_run(f"({eq_num})")
        r2.font.name = 'Times New Roman'
        r2.font.size = Pt(9.5)
        return p

    def add_caption(text):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.space_before = Pt(4)
        p.paragraph_format.space_after = Pt(6)
        p.paragraph_format.line_spacing = 1.0
        r = p.add_run(text)
        r.font.name = 'Times New Roman'
        r.font.size = Pt(8)
        return p

    def add_img(path, width=3.45):
        if os.path.exists(path):
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(4)
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.keep_with_next = True
            r = p.add_run()
            r.add_picture(path, width=Inches(width))
            return p
        return None

    # -------------------------------------------------------------------------
    # I. INTRODUCTION
    # -------------------------------------------------------------------------
    add_h1("I.  INTRODUCTION")

    add_body(
        "Diagnostic medical imaging—spanning magnetic resonance imaging (MRI), projection radiography (chest X-ray, CXR), "
        "and white-light endoscopy—constitutes the cornerstone of acute clinical triaging, oncological staging, and emergency "
        "intervention [1]–[3]. Timely and precise identification of focal pathological lesions is of decisive clinical consequence. "
        "Pediatric pneumonia represents a leading cause of childhood mortality worldwide, demanding prompt initiation of targeted antimicrobial "
        "therapy to avert respiratory distress syndrome [1], [4], [5]. In neuro-oncology, malignant intracranial neoplasms such as high-grade "
        "gliomas necessitate immediate neurosurgical resection planning and adjuvant radiochemotherapy to mitigate catastrophic mass-effect "
        "edema [2], [6], [7]. In gastroenterology, actively bleeding peptic ulcerations require urgent endoscopic intervention to forestall "
        "hypovolemic shock and fatal gastrointestinal hemorrhage [3], [8], [9]."
    )

    add_body(
        "Notwithstanding the critical importance of prompt interpretation, the volume of acquired diagnostic scans vastly outstrips the "
        "specialist workforce available in primary and secondary healthcare institutions worldwide [4], [7]. In India, the doctor-to-population "
        "ratio in rural and semi-urban districts remains stubbornly below the World Health Organization (WHO) minimum threshold of 1:1,000 [10]. "
        "More critically, official data from the Ministry of Health and Family Welfare (MoHFW) Rural Health Statistics [11] reveal that community "
        "health centers (CHCs) in rural regions face a specialist physician and radiologist shortfall exceeding 65%. Consequently, primary health "
        "facilities are predominantly manned by medical officers and general practitioners who are compelled to make urgent clinical triage decisions "
        "without specialized radiological consultation."
    )

    add_body(
        "Automated computer-aided diagnosis (CAD) systems driven by deep convolutional neural networks (CNNs) present an attractive solution "
        "for bridging this interpretive shortfall. However, clinical translation of modern deep learning models into resource-constrained frontline "
        "environments is severely hindered by three fundamental challenges:"
    )

    add_body(
        "1) Severe Model Over-parameterization: Contemporary vision transformers and ultra-deep convolutional architectures (e.g., ViT, ResNet-101, "
        "DenseNet-201) frequently contain 25M to 85M parameters, requiring dedicated graphics processing units (GPUs) and excessive memory bandwidth [12], [13]. "
        "Such infrastructure is overwhelmingly absent in rural diagnostic centers and peripheral clinics.",
        indent=True
    )

    add_body(
        "2) The Black-Box Trust Deficit: Opaque decision-making without verifiable spatial groundings prevents frontline clinicians from validating "
        "whether network predictions are anchored to genuine pathological anatomy or to spurious image acquisition artifacts [9], [17].",
        indent=True
    )

    add_body(
        "3) Inflexible Operational Decision Cutoffs: Default argmax classification rules (τ = 0.50) treat false alarms and missed critical pathologies "
        "symmetrically. In acute clinical triage, the cost of a missed intracranial malignancy or untreated pneumonia far exceeds that of a false positive, "
        "necessitating adjustable, calibrated decision boundaries tailored to specific clinical workflows.",
        indent=True
    )

    add_body(
        "To systematically address these clinical constraints, we introduce MediScan AI, an explainable multi-modal medical imaging triage framework. "
        "MediScan AI is powered by DualPathNet, a custom 14.91M-parameter convolutional architecture engineered to extract multi-scale pathological features "
        "without reliance on proprietary foundation models. The framework is evaluated across three primary diagnostic tasks: 4-class brain tumor MRI "
        "(TumorNet), binary pediatric pneumonia CXR (PneumoNet), and binary upper-GI ulcer endoscopy (GastroNet)."
    )

    add_body(
        "The primary contributions of this paper are fourfold:\n"
        "• Architectural Innovation: We propose DualPathNet, a custom convolutional backbone integrating parallel 3×3 and 5×5 residual branches, "
        "Convolutional Block Attention Modules (CBAM), deep residual refinement, and joint global average and max pooling (GAP+GMP), capturing fine "
        "lesion boundaries and broad organ context in 14.91M parameters.\n"
        "• Empirical Multi-Modal Validation: We report rigorous test-first evaluations across three publicly available medical benchmarks comprising "
        "17,794 total images (1,600 test brain MRIs, 624 test chest radiographs, and 711 test endoscopy frames), documenting 95% bootstrap confidence intervals.\n"
        "• Clinically Auditable Explainability: We hook gradient-weighted class activation mapping (Grad-CAM) directly to the penultimate residual refinement "
        "tensor (refine[2], 7×7 resolution), stabilized via 4-pass test-time augmentation (TTA) to generate anatomically congruent saliency overlays.\n"
        "• Edge Deployment Feasibility: We profile execution latencies on commodity multi-core CPU hardware, achieving end-to-end inference and Grad-CAM "
        "visualization within 731 ms (p95: 765 ms), verifying practical edge viability without GPU acceleration."
    )

    # -------------------------------------------------------------------------
    # II. RELATED WORK & DOMAIN RATIONALE
    # -------------------------------------------------------------------------
    add_h1("II.  RELATED WORK & DOMAIN RATIONALE")

    add_h2("A. Deep Learning in Diagnostic Medical Imaging")
    add_body(
        "Over the past decade, deep learning has achieved remarkable diagnostic efficacy across medical imaging sub-specialties. In neuro-oncology, "
        "the classification of intracranial neoplasms—specifically glioma, meningioma, and pituitary adenoma—from T1-weighted contrast-enhanced MRI "
        "has predominantly utilized transfer learning with general-domain backbones such as VGG-16, ResNet-50, and InceptionV3 [2], [6], [7], [14], [28]. "
        "Noreen et al. [7] explored concatenation ensembles of InceptionV3 and DenseNet201, reporting 94.34% accuracy across three tumor categories. "
        "Rizwan et al. [14] combined deep feature extractors with extreme learning machines, achieving 95.40% accuracy. Similarly, Rehman et al. [28] "
        "investigated transfer learning across multiple architectures, reporting 96.56% accuracy. However, existing works rarely evaluate multi-scale "
        "receptive fields designed specifically to address the irregular morphology of infiltrative glial margins."
    )

    add_body(
        "In thoracic imaging, automated detection of pediatric pneumonia was catalyzed by the benchmark archive curated by Kermany et al. [1]. "
        "Subsequent investigations demonstrated that pediatric chest radiographs exhibit distinct anatomical proportions and respiratory motion compared "
        "to adult cohorts [4], [5], [19]. Sharma & Guleria [4] evaluated VGG-16 backbones, reporting 92.80% accuracy. Ali et al. [5] developed an ensemble "
        "utilizing EfficientNetV2L, reaching 94.60% accuracy. Munir et al. [19] introduced PneuX-Net, reporting 95.10% accuracy and 0.987 ROC-AUC with saliency "
        "visualizations. Nevertheless, these studies evaluate single decision thresholds, overlooking the severe clinical disparity between screening and triage."
    )

    add_body(
        "In gastrointestinal endoscopy, computer-aided detection has evolved with the release of comprehensive video endoscopy databases such as "
        "HyperKvasir [15]. Sun et al. [3] surveyed deep learning methods for bleeding and mucosal ulcer detection in capsule endoscopy, noting an average "
        "accuracy of 92.40%. Hatami et al. [8] applied ResNet-50 transfer learning, achieving 93.80% accuracy on endoscopic mucosal frames. Olimboyev et al. [9] "
        "developed Gastro-AI, incorporating visual attention mechanisms to achieve 95.80% accuracy on mucosal lesions."
    )

    add_h2("B. Architectural Motivation for DualPathNet")
    add_body(
        "A critical limitation of off-the-shelf CNN backbones (e.g., standard ResNet or MobileNet) is their reliance on uniform convolutional kernel sizes "
        "(predominantly 3×3) across individual stages. While effective for natural imagery where semantic objects occupy standardized spatial hierarchies, "
        "medical pathologies exhibit profound scale divergence across diagnostic modalities:\n"
        "• Micro-structural Pathologies: Brain tumor margins require localized 3×3 receptive fields to resolve infiltrative dural edges and vascular displacement, "
        "while mucosal ulcers in endoscopy present as focal, sharp epithelial disruptions.\n"
        "• Macro-structural Pathologies: Pediatric lobar pneumonia manifests as diffuse parenchymal opacification spanning extensive pulmonary lobes, "
        "demanding wide 5×5 receptive fields to capture bilateral thoracic symmetry and broad parenchymal context."
    )

    add_body(
        "DualPathNet resolves this conflict by bifurcating feature extraction into parallel 3×3 (Path A) and 5×5 (Path B) residual pathways. "
        "By fusing localized and contextual representations through 1×1 projection and CBAM attention, the model preserves high-frequency lesion margins "
        "while simultaneously integrating broad anatomical context, achieving superior representation without ballooning parameter count."
    )

    # FIGURE 1: Architecture
    add_img("paper_figures/fig_dualpathnet_architecture.png", width=3.45)
    add_caption(
        "Fig. 1. Architectural schema of DualPathNet and the MediScan AI clinical triage pipeline. The input scan passes through an initial stem block "
        "before splitting into parallel multi-scale branches (Path A: 3×3 kernels; Path B: 5×5 kernels). Fused features are modulated by CBAM attention, "
        "processed through three residual refinement stages with a penultimate Grad-CAM hook at refine[2], and classified via dual global pooling "
        "(GAP+GMP) and an MLP head coupled with 4-pass TTA."
    )

    # -------------------------------------------------------------------------
    # III. MEDISCAN AI SYSTEM PLATFORM
    # -------------------------------------------------------------------------
    add_h1("III.  MEDISCAN AI SYSTEM PLATFORM")

    add_body(
        "MediScan AI is structured as a modular, containerized clinical microservice architecture designed for edge deployment within local hospital "
        "networks. The pipeline comprises six integrated stages: scan ingestion, modality-specific pre-processing, DualPathNet inference, "
        "threshold-calibrated triage, Grad-CAM visual explanation, and bilingual clinical report compilation."
    )

    add_h2("A. Modality-Specific Pre-Processing")
    add_body(
        "All diagnostic scans are standardized to 224×224×3 RGB tensors. To account for acquisition disparities across physical modalities, "
        "tailored normalization protocols are applied:\n"
        "• Brain MRI: Axial T1-weighted contrast-enhanced slices undergo min-max intensity scaling to normalize cranial parenchymal contrast.\n"
        "• Pediatric CXR: Radiographs undergo Contrast Limited Adaptive Histogram Equalization (CLAHE, clip limit 2.0, tile grid 8×8) to accentuate "
        "subtle alveolar consolidation and air bronchograms against cardiac shadows.\n"
        "• Endoscopy: Endoscopic mucosal frames are normalized using channel-wise standard statistics, coupled with training-time color jitter "
        "(brightness and contrast ±12%) to provide robust invariance against mucosal reflections and endoscope illumination variations."
    )

    add_h2("B. Microservice Architecture & Clinical Dashboard")
    add_body(
        "The production backend is developed using FastAPI and PyTorch 2.6 in Docker containers. Models are loaded into system memory during start-up. "
        "Incoming multipart requests carrying an image and modality tag are routed to the corresponding model instance (TumorNet, PneumoNet, or GastroNet). "
        "Inference is augmented via a 4-pass Test-Time Augmentation (TTA) scheme, averaging predictions across original, horizontally flipped, and center-cropped "
        "variants to stabilize confidence scores. Forward activations and backward gradients at the penultimate residual block (model.refine[2]) are captured "
        "via PyTorch hooks to generate Grad-CAM heatmaps. The service returns a structured JSON payload containing predicted diagnosis, calibrated risk tier, "
        "confidence probability, and a Base64-encoded visual heatmap overlay."
    )

    add_body(
        "The clinician dashboard is engineered with Next.js 14, React, and Tailwind CSS, providing a synchronized English and Hindi (hi-IN) interface. "
        "A clinical report engine compiles patient demographic fields, raw scan imagery, Grad-CAM heatmap overlays, and severity-graded diagnostic "
        "recommendations into an auditable, printable medical document."
    )

    # -------------------------------------------------------------------------
    # IV. DUALPATHNET ARCHITECTURAL FORMULATION
    # -------------------------------------------------------------------------
    add_h1("IV.  DUALPATHNET ARCHITECTURAL FORMULATION")

    add_body(
        "DualPathNet is formulated as a dual-branch residual convolutional network. Given an input medical scan X ∈ R^(3×224×224), an initial stem block "
        "(Conv 7×7, stride 2, padding 3, BN, SiLU activation, MaxPool 3×3, stride 2) reduces spatial resolution to 64×56×56. The resulting feature tensor "
        "is bifurcated into parallel processing pathways:"
    )

    add_eq("F_A = F_PathA(X_stem; W_A)    (Kernels: 3×3)", 1)
    add_eq("F_B = F_PathB(X_stem; W_B)    (Kernels: 5×5)", 2)

    add_body(
        "Each path comprises four sequential stages with channel depths 32→64→128→256, built using ResidualConvBlock instances. Each residual block "
        "is defined mathematically as:"
    )

    add_eq("y = SiLU(BN(Conv_2(SiLU(BN(Conv_1(x)))))) + Shortcut(x)", 3)

    add_body(
        "where Shortcut(x) represents an identity skip connection when channel dimensions match, or a 1×1 convolution with BatchNorm when channel "
        "transition occurs. Spatial dropout (p=0.08) is applied between stages to prevent co-adaptation of convolutional feature maps."
    )

    add_body(
        "At the conclusion of the fourth stage, Path A and Path B output feature tensors F_A, F_B ∈ R^(256×14×14). These representations are concatenated "
        "along the channel dimension and projected through a 1×1 fusion convolution with batch normalization and SiLU activation:"
    )

    add_eq("F_fuse = SiLU(BN(Conv1×1([F_A ; F_B]))) ∈ R^(512×14×14)", 4)

    add_body(
        "The fused representation is subsequently modulated by a Convolutional Block Attention Module (CBAM) [18]. Channel attention aggregates spatial "
        "context through both global average pooling (GAP) and global max pooling (GMP):"
    )

    add_eq("M_c(F) = σ(W_1(SiLU(W_0(GAP(F)))) + W_1(SiLU(W_0(GMP(F)))))", 5)

    add_body(
        "where W_0 ∈ R^((C/r)×C) and W_1 ∈ R^(C×(C/r)) with reduction ratio r=16, and σ denotes the sigmoid function. The channel-refined tensor "
        "F' = M_c(F) ⊗ F is then processed by spatial attention:"
    )

    add_eq("M_s(F') = σ(Conv7×7([GAP_c(F') ; GMP_c(F')]))", 6)

    add_body(
        "yielding the fully attended feature tensor F'' = M_s(F') ⊗ F'. The attended tensor passes through a deep refinement block comprising three "
        "residual stages with a max-pooling step, reducing spatial resolution to 512×7×7. The penultimate residual layer refine[2] serves as the direct "
        "gradient hook for Grad-CAM generation."
    )

    add_body(
        "Spatial collapse is achieved via dual global pooling concatenating GAP and GMP across the 512 channels:"
    )

    add_eq("z = [GAP(F_ref) ; GMP(F_ref)] ∈ R^1024", 7)

    add_body(
        "The 1,024-dimensional diagnostic representation is classified via an MLP head (Linear 1024→512, BN, SiLU, Dropout p=0.35, Linear 512→256, "
        "BN, SiLU, Dropout p=0.20, Linear 256→C) to yield raw unnormalized class logits."
    )

    # TABLE I: PARAMETERS
    p_t1 = doc.add_paragraph()
    p_t1.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_t1.paragraph_format.space_before = Pt(6)
    p_t1.paragraph_format.space_after = Pt(2)
    p_t1.paragraph_format.keep_with_next = True
    r_t1 = p_t1.add_run("TABLE I\nDUALPATHNET LAYER-BY-LAYER PARAMETER DISTRIBUTION")
    r_t1.font.name = 'Times New Roman'
    r_t1.font.size = Pt(8)
    r_t1.font.bold = True

    t1 = doc.add_table(rows=8, cols=4)
    t1.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(t1)

    t1_headers = ["Architectural Sub-module", "Constituent Layers & Ops", "Parameters", "Parameter Share"]
    t1_data = [
        ["Input Stem", "Conv 7×7, stride 2, BN, SiLU, MaxPool", "4,736", "0.03%"],
        ["Path A (3×3 Kernels)", "4 Stages (32→64→128→256), Drop2d(0.08)", "1,218,816", "8.17%"],
        ["Path B (5×5 Kernels)", "4 Stages (32→64→128→256), Drop2d(0.08)", "3,299,072", "22.12%"],
        ["Feature Fusion", "Concat (512-ch), Conv 1×1, BN, SiLU", "262,400", "1.76%"],
        ["CBAM Attention", "Channel MLP (r=16) + Spatial Conv 7×7", "33,026", "0.22%"],
        ["Deep Refinement", "3 ResBlocks (512-ch) + MaxPool (7×7)", "9,438,208", "63.29%"],
        ["Classification Head", "Dual Pool (1024-d), MLP (512→256→C)", "656,644", "4.40%"],
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
        "As documented in Table I, DualPathNet comprises exactly 14,912,870 parameters for TumorNet (4-class) and 14,912,356 parameters for PneumoNet "
        "and GastroNet (binary). The exact 514-parameter difference corresponds to the final projection layer: 256×4+4 = 1,028 weights/biases for "
        "TumorNet versus 256×2+2 = 514 weights/biases for the binary models.",
        indent=True
    )

    # -------------------------------------------------------------------------
    # V. TRAINING METHODOLOGY & OPTIMIZATION PILLARS
    # -------------------------------------------------------------------------
    add_h1("V.  TRAINING METHODOLOGY & OPTIMIZATION PILLARS")

    add_body(
        "To maximize diagnostic accuracy without expanding model capacity, training was conducted under an eight-pillar optimization methodology "
        "designed to stabilize loss gradients, counter prevalence imbalances, and promote generalizable minima:"
    )

    add_body(
        "1) Full Dataset Exposure: All artificial sample-per-epoch caps were eliminated, allowing the network to observe every distinct patient scan "
        "in every epoch (4,760 brain MRI, 4,447 chest radiographs, and 3,317 endoscopy frames).",
        indent=True
    )

    add_body(
        "2) Dynamic Class-Weighted Sampling: In medical cohorts, pathology distribution is heavily skewed (e.g., pediatric pneumonia outnumbers normal "
        "CXR by nearly 3:1). We calculated inverse class frequency weights w_c = 1 / N_c and utilized PyTorch WeightedRandomSampler with replacement, "
        "ensuring that every mini-batch presents an expected 50/50 balance between pathological and normal cases during gradient calculation.",
        indent=True
    )

    add_body(
        "3) Class-Weighted Focal Loss with Label Smoothing: Standard cross-entropy loss is dominated by easily classified background regions. "
        "We implemented focal loss objectives [22] with focusing parameter γ=2.0 and label smoothing ε [23]:",
        indent=True
    )

    add_eq("L_Focal = -α_t (1 - p_t)^γ log(p_t)", 8)

    add_body(
        "For TumorNet, Multi-Class Focal Loss was applied with γ=1.5, ε=0.05, and class weights α = [1.35, 1.05, 0.90, 0.90] for glioma, meningioma, "
        "no tumor, and pituitary, respectively. The α=1.35 weighting on glioma penalizes missed infiltrative malignancies. For GastroNet, normal mucosa "
        "outnumbers ulcer frames by 2:1; we assigned α=1.35 to the positive ulcer class (α=[1.0, 1.35], ε=0.03), penalizing missed ulcerations.",
        indent=True
    )

    add_body(
        "4) Warmup Cosine Annealing: Using the AdamW optimizer (weight decay λ=10^-4), a linear warmup over epochs 1–2 ramped learning rate from 0 to "
        "η_max = 4×10^-4, followed by cosine decay to η_min = 1×10^-6, preventing early divergence of attention weights.",
        indent=True
    )

    add_body(
        "5) Stochastic Weight Averaging (SWA): From epoch 13 to 20, we maintained a running average of model parameters: "
        "W_SWA = (n · W_SWA + W_cur) / (n + 1). SWA guides the model to the geometric center of flat loss valleys, promoting robustness against "
        "scanner domain shifts [21].",
        indent=True
    )

    add_body(
        "6) Test-Time Augmentation (TTA): During inference, four augmented forward passes are averaged:\n"
        "  p_TTA = 0.40 p_orig + 0.30 p_flip + 0.15 p_crop + 0.15 p_crop-flip\n"
        "This multi-pass averaging mitigates acquisition orientation bias and stabilizes borderline prediction confidences.",
        indent=True
    )

    add_body(
        "7) Validation-Tuned Operating Thresholds: In binary triage, standard argmax cutoffs (τ=0.50) are decoupled from clinical priorities. "
        "Decision thresholds τ were tuned strictly on validation data to maximize balanced accuracy: τ=0.780 for PneumoNet (enforcing high specificity "
        "against pediatric false alarms) and τ=0.580 for GastroNet.",
        indent=True
    )

    add_body(
        "8) Multi-Tier Clinical Risk Stratification: Predictions are mapped into three actionable clinical urgency tiers:\n"
        "• High Risk (Urgent Consultation, 12–24h): Brain MRI classified as Glioma or Pituitary tumor; CXR with p_pneumonia ≥ 0.780.\n"
        "• Moderate Risk (Specialist Referral): Brain MRI classified as Meningioma; Endoscopy with p_ulcer ≥ 0.580.\n"
        "• Low Risk (Routine Outpatient Review): Scans classified as No Tumor or with pathology probabilities strictly below τ.",
        indent=True
    )

    # -------------------------------------------------------------------------
    # VI. EXPERIMENTAL SETUP & DATASETS
    # -------------------------------------------------------------------------
    add_h1("VI.  EXPERIMENTAL SETUP & BENCHMARK DATASETS")

    add_body(
        "Experiments were conducted across three publicly available medical imaging benchmarks comprising 17,794 total images:\n"
        "1) Brain Tumor MRI Benchmark [2], [24]: 7,200 T1-weighted contrast-enhanced axial slices categorized into four classes: glioma (1,800), "
        "meningioma (1,800), pituitary adenoma (1,800), and no tumor (1,800). Historical mirrors cite 7,023 images; our cleaned repository standardizes "
        "exactly 1,800 per class. Partitions: 4,760 training (1,190/class), 840 validation (210/class), and 1,600 held-out test scans (400/class).\n"
        "2) Pediatric Chest Radiograph Dataset (Kermany et al.) [1]: 5,856 anterior-posterior radiographs from pediatric patients aged one to five years "
        "(1,583 normal, 4,273 pneumonia). The official 624-image test set (234 normal, 390 pneumonia) was reserved. From the remaining 5,232 images, an 85/15 "
        "stratified split yielded 4,447 training (1,146 normal, 3,301 pneumonia) and 785 validation radiographs (203 normal, 582 pneumonia).\n"
        "3) HyperKvasir Endoscopic Dataset [15]: 4,738 video endoscopy frames labeled for mucosal health (3,165 normal mucosa, 1,573 ulcerated). Partitions: "
        "3,317 training (2,215 normal, 1,102 ulcer), 710 validation (475 normal, 235 ulcer), and 711 held-out test frames (475 normal, 236 ulcer)."
    )

    # TABLE II: DATASETS
    p_t2 = doc.add_paragraph()
    p_t2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_t2.paragraph_format.space_before = Pt(6)
    p_t2.paragraph_format.space_after = Pt(2)
    p_t2.paragraph_format.keep_with_next = True
    r_t2 = p_t2.add_run("TABLE II\nBENCHMARK DATASET SPECIFICATIONS AND PARTITIONS")
    r_t2.font.name = 'Times New Roman'
    r_t2.font.size = Pt(8)
    r_t2.font.bold = True

    t2 = doc.add_table(rows=5, cols=5)
    t2.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(t2)

    t2_headers = ["Dataset / Task", "Classes", "Training Set", "Validation Set", "Held-Out Test Set"]
    t2_data = [
        ["Brain Tumor MRI [2]", "4 (Glioma/Mening/NoTumor/Pituit)", "4,760", "840", "1,600 (400/class)"],
        ["Pediatric CXR [1]", "2 (Normal / Pneumonia)", "4,447", "785", "624 (Official Test)"],
        ["HyperKvasir Endoscopy [15]", "2 (Normal / Mucosal Ulcer)", "3,317", "710", "711 (Held-out)"],
        ["Aggregate Benchmark", "3 Clinical Tasks", "12,524", "2,335", "2,935 Total Test"],
    ]

    for col_idx, h in enumerate(t2_headers):
        cell = t2.cell(0, col_idx)
        cell.text = h
        set_cell_margins(cell, 80, 80, 80, 80)
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
            set_cell_margins(cell, 60, 60, 80, 80)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if col_idx < 2 else WD_ALIGN_PARAGRAPH.RIGHT
            p.runs[0].font.name = 'Times New Roman'
            p.runs[0].font.size = Pt(7.5)
            if row_idx == 3:
                p.runs[0].font.bold = True

    add_body(
        "Protocol Integrity & Patient Split Disclosure: Checkpoints and decision cutoffs τ were selected solely using validation data. "
        "Test partitions were evaluated strictly once upon freezing weights. We openly state that because public benchmark archives [1], [2] "
        "do not provide de-identified patient master IDs, partitions were created using stratified image-level separation. While images were verified "
        "free of duplicates across splits, cross-slice patient correlation cannot be definitively ruled out in 3D MRI series, representing a documented "
        "constraint of public benchmark research.",
        indent=True
    )

    # -------------------------------------------------------------------------
    # VII. RESULTS AND DISCUSSION
    # -------------------------------------------------------------------------
    add_h1("VII.  RESULTS AND DISCUSSION")

    add_h2("A. Primary Multi-Modal Diagnostic Performance")
    add_body(
        "Table III reports test evaluation metrics across all three clinical modalities using 4-pass TTA. Non-parametric bootstrap resampling "
        "(B=1,000 iterations) was conducted to compute 95% confidence intervals."
    )

    # TABLE III: MAIN RESULTS
    p_t3 = doc.add_paragraph()
    p_t3.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_t3.paragraph_format.space_before = Pt(6)
    p_t3.paragraph_format.space_after = Pt(2)
    p_t3.paragraph_format.keep_with_next = True
    r_t3 = p_t3.add_run("TABLE III\nPRIMARY MULTI-MODAL TEST PERFORMANCE WITH 95% BOOTSTRAP CONFIDENCE INTERVALS")
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
    add_img("paper_figures/fig_training_curves_multipanel.png", width=3.45)
    add_caption(
        "Fig. 2. Convergence curves across 20 training epochs for (a) TumorNet, (b) PneumoNet, and (c) GastroNet. "
        "Engagement of Stochastic Weight Averaging (SWA) at epoch 13 stabilizes loss trajectories and promotes broad-optima convergence."
    )

    add_img("paper_figures/fig_confusion_matrices_multipanel.png", width=3.45)
    add_caption(
        "Fig. 3. Confusion matrices on held-out test partitions: (a) TumorNet (1,600 brain MRI scans, exactly 400 per class), "
        "(b) PneumoNet (624 pediatric chest radiographs at τ=0.780), and (c) GastroNet (711 endoscopic mucosal frames at τ=0.580)."
    )

    add_h2("B. TumorNet Per-Class Breakdown & Glioma Error Analysis")
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
    r_t4 = p_t4.add_run("TABLE IV\nTUMORNET PER-CLASS TEST BREAKDOWN (N=1,600, 400 SCANS/CLASS)")
    r_t4.font.name = 'Times New Roman'
    r_t4.font.size = Pt(8)
    r_t4.font.bold = True

    t4 = doc.add_table(rows=6, cols=6)
    t4.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(t4)

    t4_headers = ["Pathological Class", "Support", "Precision (%)", "Sensitivity [95% CI]", "F1 (%)", "Specificity (%)"]
    t4_data = [
        ["Glioma", "400", "99.10", "83.00 [79.0, 86.4]", "90.34", "99.75"],
        ["Meningioma", "400", "89.86", "97.50 [95.5, 98.6]", "93.53", "96.33"],
        ["No Tumor (Normal)", "400", "93.24", "100.00 [99.0, 100.0]", "96.50", "97.58"],
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
    add_img("paper_figures/fig_roc_curves_multipanel.png", width=3.45)
    add_caption(
        "Fig. 4. Receiver Operating Characteristic (ROC) curves on benchmark test partitions: (a) TumorNet multi-class one-vs-rest (macro AUC 0.9874), "
        "(b) PneumoNet binary ROC (AUC 0.9586) depicting operating points at τ=0.780 and τ=0.500, and (c) GastroNet binary ROC (AUC 0.9904)."
    )

    add_img("paper_figures/fig_gradcam_multipanel.png", width=3.45)
    add_caption(
        "Fig. 5. Multi-modal visual explainability generated via Grad-CAM hooked to refine[2]. Representative test inputs: (a) Glioma, (b) Meningioma, "
        "(c) Pituitary Adenoma, (d) Normal MRI, (e) CXR Pneumonia, and (f) Endoscopic Mucosal Ulcer. Top row: original input scan; Bottom row: Grad-CAM "
        "jet-colormap overlay (7×7 upsampled to 224×224) showing anatomically congruent focus."
    )

    add_h2("D. Baseline Comparison Against Standard Vision Backbones")
    add_body(
        "To rigorously benchmark DualPathNet, five standard vision backbones were trained under identical data splits, augmentation schedules, "
        "and optimization protocols (Table V). DualPathNet (14.91M parameters, 2.45 GFLOPs) achieves superior test performance across all three modalities "
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

    # Save to MediScan_AI_IEEE_Conference_Paper_Final.docx
    final_docx = "d:/CapStone-2/MediScan_AI_IEEE_Conference_Paper_Final.docx"
    doc.save(final_docx)
    print(f"Saved: {final_docx}")

    # Also update MediScan_AI_Conference_Paper_v2.docx
    v2_docx = "d:/CapStone-2/MediScan_AI_Conference_Paper_v2.docx"
    try:
        doc.save(v2_docx)
        print(f"Updated: {v2_docx}")
    except Exception as e:
        print(f"Could not overwrite {v2_docx} ({e}).")

if __name__ == "__main__":
    build_paper()
