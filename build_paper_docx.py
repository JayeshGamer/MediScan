import os
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(f'<w:tcMar {nsdecls("w")}><w:top w:w="{top}" w:type="dxa"/><w:bottom w:w="{bottom}" w:type="dxa"/><w:left w:w="{left}" w:type="dxa"/><w:right w:w="{right}" w:type="dxa"/></w:tcMar>')
    tcPr.append(tcMar)

def create_document():
    doc = docx.Document()

    # Set page margins (0.75 in)
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(0.75)
        section.bottom_margin = Inches(0.75)
        section.left_margin = Inches(0.75)
        section.right_margin = Inches(0.75)

    # Styles
    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Times New Roman'
    normal_style.font.size = Pt(9.5)
    normal_style.font.color.rgb = RGBColor(0x22, 0x22, 0x22)

    # Document Title
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_before = Pt(0)
    p_title.paragraph_format.space_after = Pt(6)
    r_title = p_title.add_run("A Multiscale Dual-Path Residual Network with Integrated Explainability for Cross-Modality Medical Screening")
    r_title.font.name = 'Times New Roman'
    r_title.font.size = Pt(18)
    r_title.font.bold = True
    r_title.font.color.rgb = RGBColor(0x0f, 0x17, 0x2a)

    # Authors
    p_authors = doc.add_paragraph()
    p_authors.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_authors.paragraph_format.space_after = Pt(2)
    r_authors = p_authors.add_run("Daksh Gurjar, Aditya Yashovardhan, and Jayesh Mandloi")
    r_authors.font.name = 'Times New Roman'
    r_authors.font.size = Pt(10.5)
    r_authors.font.bold = True

    # Affiliation
    p_affil = doc.add_paragraph()
    p_affil.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_affil.paragraph_format.space_after = Pt(12)
    r_affil = p_affil.add_run("Department of Computer Engineering, SVKM's NMIMS, Mukesh Patel School of Technology Management and Engineering (MPSTME)\nShirpur, Maharashtra 425405, India\nEmails: {daksh.gurjar, aditya.yashovardhan, jayesh.mandloi}@nmims.edu")
    r_affil.font.name = 'Times New Roman'
    r_affil.font.size = Pt(8.5)
    r_affil.font.italic = True
    r_affil.font.color.rgb = RGBColor(0x47, 0x55, 0x69)

    # Abstract Box
    tbl_abs = doc.add_table(rows=1, cols=1)
    tbl_abs.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell_abs = tbl_abs.cell(0, 0)
    set_cell_background(cell_abs, "F8FAFC")
    set_cell_margins(cell_abs, top=140, bottom=140, left=180, right=180)
    
    p_abs = cell_abs.paragraphs[0]
    p_abs.paragraph_format.space_after = Pt(4)
    p_abs.paragraph_format.line_spacing = 1.15
    r_abs_lbl = p_abs.add_run("Abstract—")
    r_abs_lbl.font.bold = True
    r_abs_lbl.font.size = Pt(9)
    r_abs_lbl.font.name = 'Times New Roman'
    
    abstract_text = (
        "Frontline primary healthcare facilities in resource-constrained regions face severe interpretive backlogs "
        "across diverse diagnostic modalities. Computer-aided diagnosis systems typically address isolated pathologies "
        "on single academic datasets, leaving clinics without unified triage tools. We present MediScan AI, a cross-modality "
        "diagnostic and tele-triage prototype powered by DualPathNet, a 14.91-million-parameter dual-path residual "
        "convolutional backbone combining parallel 3x3 fine-texture and 5x5 broad-context branches with CBAM attention "
        "and joint global average and max pooling. Evaluated on public benchmark test sets across three clinical domains "
        "without pre-trained transfer learning, the system achieves 94.94% overall accuracy [95% CI: 93.8, 95.9] and 0.9874 "
        "macro ROC-AUC on 1,600 brain MRI scans; 89.58% accuracy [86.9, 91.7] with 91.88% specificity and 88.21% sensitivity "
        "at a specificity-protecting threshold (tau = 0.780), and 96.15% sensitivity at tau = 0.500 on 624 pediatric chest radiographs; "
        "and 96.06% accuracy [94.4, 97.3] with 94.49% sensitivity and 0.9904 ROC-AUC on 711 endoscopic images (tau = 0.580). "
        "Integrated gradient-weighted class activation mapping (Grad-CAM) from the bottleneck refinement layer provides coarse "
        "7x7 spatial interpretability to support clinical review. CPU benchmark testing demonstrates sub-second inference "
        "(single-pass latency: 101.9–108.9 ms; full 4-pass test-time augmentation plus Grad-CAM: 722.0–755.7 ms), confirming "
        "feasibility for district clinic workstations."
    )
    r_abs_txt = p_abs.add_run(abstract_text)
    r_abs_txt.font.size = Pt(9)
    r_abs_txt.font.name = 'Times New Roman'

    p_idx = cell_abs.add_paragraph()
    p_idx.paragraph_format.space_before = Pt(2)
    p_idx.paragraph_format.space_after = Pt(0)
    r_idx_lbl = p_idx.add_run("Index Terms—")
    r_idx_lbl.font.bold = True
    r_idx_lbl.font.size = Pt(8.5)
    r_idx_lbl.font.name = 'Times New Roman'
    r_idx_txt = p_idx.add_run("Computer-aided diagnosis, multiscale convolutional neural networks, explainable artificial intelligence (XAI), Grad-CAM, pediatric pneumonia, brain tumor, gastric ulcer, medical triage, decision threshold.")
    r_idx_txt.font.size = Pt(8.5)
    r_idx_txt.font.name = 'Times New Roman'

    p_spacer = doc.add_paragraph()
    p_spacer.paragraph_format.space_after = Pt(8)

    def add_sec_heading(title):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(12)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.keep_with_next = True
        r = p.add_run(title)
        r.font.name = 'Times New Roman'
        r.font.size = Pt(10)
        r.font.bold = True
        r.font.color.rgb = RGBColor(0x0f, 0x17, 0x2a)
        return p

    def add_subsec_heading(title):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(8)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.keep_with_next = True
        r = p.add_run(title)
        r.font.name = 'Times New Roman'
        r.font.size = Pt(9.5)
        r.font.bold = True
        r.font.italic = True
        r.font.color.rgb = RGBColor(0x1e, 0x29, 0x3b)
        return p

    def add_body_p(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.line_spacing = 1.15
        r = p.add_run(text)
        r.font.name = 'Times New Roman'
        r.font.size = Pt(9.5)
        return p

    # --- SECTION I ---
    add_sec_heading("I. INTRODUCTION")
    add_body_p(
        "Medical imaging—encompassing magnetic resonance imaging (MRI), chest radiography (CXR), and white-light "
        "endoscopy—forms the cornerstone of acute clinical triage, oncology staging, and emergency surgical intervention [1]–[3]. "
        "Early and precise identification of focal pathology carries decisive clinical weight: pediatric pneumonia demands prompt "
        "antimicrobial administration to avert acute respiratory failure [1], [4]; intracranial neoplasms such as aggressive gliomas "
        "mandate urgent neurosurgical biopsy and stereotactic planning [2], [5]; and bleeding gastric mucosal ulcerations require "
        "timely endoscopic therapeutic hemostasis [3], [6]. In many developing regions and district health ecosystems, however, "
        "the diagnostic workload outstrips the available specialist capacity [1], [2]. In rural and semi-urban India, the doctor-to-population "
        "ratio frequently lags behind the World Health Organization (WHO) benchmark of 1:1,000, and sub-specialist availability in "
        "neuroradiology, pediatric thoracic radiology, and gastroenterology is substantially lower. Consequently, primary health "
        "centers (PHCs) and community health posts are staffed primarily by medical officers who must manage acute triage decisions "
        "without specialist peer support."
    )
    add_body_p(
        "Although deep convolutional neural networks (CNNs) have shown substantial diagnostic promise across computer-aided diagnosis "
        "benchmarks [1]–[7], several systemic barriers impede their transition into frontline primary care:"
    )
    add_body_p(
        "• G1 — Single-Disease Fragmentation: The vast majority of published medical deep-learning systems are tailored to a single pathology "
        "on an isolated dataset [1], [3], [5]. A community health facility addressing neurological, respiratory, and digestive complaints "
        "must therefore deploy disjointed, non-interoperable tools."
    )
    add_body_p(
        "• G2 — Model Opacity: Standard classification backbones emit ungrounded scalar probabilities without spatial verification. Clinicians "
        "cannot determine whether a high prediction score reflects genuine pathology or irrelevant edge artifacts [7], [8]."
    )
    add_body_p(
        "• G3 — Domain and Distribution Shift: Models trained on curated tertiary-hospital benchmarks can degrade markedly when confronted "
        "with hardware noise, patient motion, and acquisition variations typical of secondary care centers."
    )
    add_body_p(
        "• G4 — Language and Accessibility Barriers: Existing AI clinical decision-support user interfaces operate almost exclusively in English, "
        "creating friction for bilingual community health extension workers."
    )
    add_body_p(
        "• G5 — Longitudinal Auditability: Transient on-screen inferences rarely integrate into persistent electronic health record (EHR) audit trails."
    )
    add_body_p(
        "• G6 — Unstructured Documentation: Unstandardized outputs cannot be seamlessly printed or archived for secondary referral."
    )
    add_body_p(
        "• G7 — Unexamined Operating Thresholds: Defaulting to an arbitrary 0.50 classification cutoff disregards the asymmetric clinical costs "
        "of false negatives versus false positives in clinical screening [9]."
    )
    add_body_p(
        "• G8 — Computational Overhead: State-of-the-art vision models frequently demand high-end workstation GPUs that are absent in resource-constrained rural clinics."
    )
    add_body_p(
        "To address these operational challenges, we present MediScan AI, an end-to-end medical decision-support and tele-triage prototype. "
        "Our core contributions are:"
    )
    add_body_p(
        "1) A Unified Multiscale Architecture (DualPathNet): A compact 14.91-million-parameter backbone designed from scratch without external pre-training. "
        "It integrates parallel 3x3 fine-texture and 5x5 broad-morphology residual convolutional streams with Convolutional Block Attention Modules (CBAM) "
        "and dual global average and maximum pooling to preserve subtle lesion margins alongside macro-architectural context."
    )
    add_body_p(
        "2) Cross-Modality Benchmark Evaluation: Rigorous empirical evaluation across three distinct clinical imaging modalities on held-out public "
        "benchmark test sets (1,600 brain MRI scans, 624 pediatric chest radiographs, and 711 endoscopy frames), complete with confusion-matrix-derived "
        "metrics, Wilson 95% confidence intervals, and per-class error breakdowns."
    )
    add_body_p(
        "3) Integrated Visual Explainability and Operating-Point Calibration: Transparent spatial attribution via coarse 7x7 Grad-CAM heatmaps "
        "extracted directly from the bottleneck refinement layer, coupled with empirical decision threshold sweeps (tau) that illustrate the explicit "
        "trade-offs between screening sensitivity and specificity."
    )
    add_body_p(
        "4) Edge-Ready Service Pipeline: A containerized FastAPI microservice coupled to a bilingual Next.js interface that delivers measured sub-second "
        "CPU inference latency (101.9–108.9 ms single-pass; 722.0–755.7 ms for 4-pass test-time augmentation plus Grad-CAM), enabling deployment on "
        "standard clinic hardware."
    )
    add_body_p(
        "Scope Statement: This paper focuses on addressing gaps G1 (via a unified backbone across three screening engines), G2 (spatial Grad-CAM attribution), "
        "G4 (bilingual UI integration), G7 (validation operating-point selection), and G8 (measured edge CPU latency). Gaps G3, G5, and G6 are documented "
        "transparently as ongoing engineering and multi-center clinical validation goals."
    )

    # --- SECTION II ---
    add_sec_heading("II. RELATED WORK")
    add_subsec_heading("A. Brain Tumor Classification in MRI")
    add_body_p(
        "Magnetic resonance imaging represents the primary modality for intracranial tumor detection and subtype stratification [2]. "
        "Noreen et al. [5] demonstrated that concatenating intermediate convolutional features from deep layers enhances discrimination among "
        "gliomas, meningiomas, and pituitary tumors. Younis et al. [10] evaluated pre-trained ResNet-50 backbones, while Asif et al. [11] and "
        "Ahmad and Choudhury [12] compared multiple transfer-learning feature extractors. Recent frameworks by Rizwan et al. [13] and Jia and "
        "Chen [14] investigated customized convolutional designs. However, these pipelines typically operate as isolated single-disease engines "
        "and rely heavily on very deep ImageNet-pretrained backbones exceeding 25–50M parameters."
    )
    add_subsec_heading("B. Pediatric Pneumonia Detection in Chest Radiography")
    add_body_p(
        "Pediatric chest radiograph interpretation is challenged by respiratory motion artifacts and subtle parenchymal consolidations [1], [15]. "
        "Baseline CNN pipelines were established by Sharma and Guleria [4], and deep EfficientNet models were applied by Ali et al. [16]. "
        "Attention mechanisms have gained prominence: Potharaju et al. [17] coupled CBAM with squeeze-and-excitation blocks for radiographic opacity "
        "classification, and Shilpa et al. [18] explored multi-scale feature hierarchies. Interpretability was investigated by Ren et al. [7], who "
        "applied saliency maps to pneumonia identification. Nonetheless, published CXR pipelines often adopt default 0.50 decision cutoffs without "
        "examining how operating thresholds shift the false-negative versus false-positive balance in pediatric populations."
    )
    add_subsec_heading("C. Gastric Lesion Recognition in Endoscopy")
    add_body_p(
        "Endoscopic assessment of upper gastrointestinal pathologies requires differentiating benign inflammatory peptic ulcers from early malignant "
        "neoplasms [3], [6]. Sun et al. [3] introduced specialized CNNs for ulcer boundary identification, while Hatami et al. [6] applied convolutional "
        "architectures to precancerous lesions. The release of the curated HyperKvasir dataset by Borgli et al. [19] catalyzed expanded investigations: "
        "Olimboyev et al. [20] developed Gastro-AI for multi-class digestive disease classification, and Tan et al. [21] demonstrated real-time "
        "differentiation of malignant versus benign gastric ulcers. Challenges persist regarding mucosal reflections, specular glare, and variable endoscopic lighting."
    )
    add_subsec_heading("D. Multiscale Architectures and Attention Mechanisms")
    add_body_p(
        "Residual learning, pioneered by He et al. [22], resolves gradient degradation in deep architectures. Multiscale receptive fields, such as "
        "parallel convolutional branches, capture features across differing spatial frequencies. The Convolutional Block Attention Module (CBAM), "
        "formulated by Woo et al. [23], applies sequential channel-wise and spatial-wise attention. Visual attribution via Gradient-weighted Class "
        "Activation Mapping (Grad-CAM), introduced by Selvaraju et al. [8], generates coarse localization maps by computing gradients with respect to "
        "feature maps. While attention and multiscale branches have been deployed independently, evaluating their unified combination across multiple "
        "diagnostic imaging modalities from scratch remains underexplored."
    )
    add_subsec_heading("E. Research Gap and System Focus")
    add_body_p(
        "Across the reviewed literature, existing works predominantly evaluate single pathologies on single datasets, rarely validate operating "
        "thresholds, and infrequently benchmark actual CPU inference latency on commodity hardware. MediScan AI addresses this gap by assessing "
        "a shared compact multiscale backbone across brain MRI, pediatric CXR, and endoscopy, accompanied by bottleneck Grad-CAM interpretability and "
        "measured edge efficiency."
    )

    # --- SECTION III ---
    add_sec_heading("III. MEDISCAN AI ARCHITECTURE & SYSTEM PIPELINE")
    add_body_p(
        "MediScan AI is structured as a decoupled microservice pipeline designed for resource-constrained clinics. "
        "The overall workflow encompasses image ingestion, modality-specific pre-processing, DualPathNet inference, "
        "calibrated operating-threshold application, Grad-CAM visualization, and bilingual clinical presentation."
    )
    add_subsec_heading("A. Pre-Processing Pipeline")
    add_body_p(
        "1) Spatial Standardization: Input images of arbitrary dimensions and color formats are standardized to 3x224x224 RGB tensors. "
        "Grayscale radiological scans (MRI and CXR) are replicated across three identical channels to conform to the input tensor shape."
    )
    add_body_p(
        "2) Contrast-Limited Adaptive Histogram Equalization (CLAHE): For radiological modalities (MRI and CXR), CLAHE [24] is applied on an 8x8 "
        "grid with a clip limit of 2.0. This local contrast enhancement sharpens subtle parenchymal consolidations and intracranial tumor margins "
        "while preventing excessive high-frequency noise amplification."
    )
    add_body_p(
        "3) Endoscopic Color Normalization: Endoscopic images undergo adaptive channel-wise standardization and illumination compensation to "
        "mitigate mucosal reflections and lighting disparities without distorting diagnostic mucosal color signatures."
    )
    add_subsec_heading("B. DualPathNet Architecture")
    add_body_p(
        "DualPathNet comprises 14.91M parameters built entirely from scratch. Its structural foundation includes two parallel residual feature "
        "streams, an attention bridge, bottleneck refinement, dual global pooling, and a regularized classification head:"
    )
    add_body_p(
        "1) Dual Receptive-Field Streams (Path A and Path B): Input x is processed concurrently through two 4-stage residual convolutional paths. "
        "Path A employs 3x3 kernels (padding=1) across stages with 32, 64, 128, and 256 channels, capturing fine-grained micro-textures, mucosal margins, "
        "and sharp structural edges. Path B utilizes 5x5 kernels (padding=2) across identical channel depths, extracting broad anatomical context, "
        "macro-morphological geometry, and diffuse parenchymal airspace density changes. Each stage consists of two-conv ResidualConvBlocks with "
        "Batch Normalization, SiLU (Swish) activations [25], and identity shortcuts (or 1x1 convolutions when channel counts expand). Spatial Dropout2d "
        "(p=0.08) is applied after stages 1–3 to prevent co-adaptation."
    )
    add_body_p(
        "2) Channel Fusion and CBAM Attention: The 256-channel feature maps from Path A and Path B are concatenated into a 512-channel composite tensor, "
        "which is projected through a 1x1 convolution with BatchNorm and SiLU. The resulting feature map passes through a CBAM block [23], consisting of "
        "Channel Attention (reduction ratio r=16, pooling average and max features through a shared two-layer MLP) followed by Spatial Attention "
        "(a 7x7 convolution across channel-wise average and max projections). This forces the network to emphasize clinically informative feature channels "
        "and focal pathology regions."
    )
    add_body_p(
        "3) Residual Refinement Bottleneck: The attention-modulated features are processed through two 512-channel 3x3 residual blocks separated by a "
        "2x2 max-pooling layer (model.refine[2]), producing the final spatial feature representation F in R^(512 x 7 x 7)."
    )
    add_body_p(
        "4) Dual Pooling and Classification Head: To preserve both background context and peak focal activations, F is pooled concurrently via Global "
        "Average Pooling (GAP) and Global Max Pooling (GMP). The resulting 512-dimensional vectors are concatenated into a 1,024-dimensional representation: "
        "z = [GAP(F) || GMP(F)]. The classification head is a multi-layer perceptron: Linear(1024, 512) -> BatchNorm1d -> SiLU -> Dropout(0.35) -> "
        "Linear(512, 256) -> BatchNorm1d -> SiLU -> Dropout(0.20) -> Linear(256, C)."
    )
    add_body_p(
        "5) Coarse Bottleneck Grad-CAM: To provide interpretable spatial attribution, forward activations A and backward gradients are hooked directly "
        "at the final convolutional block of the refinement bottleneck (model.refine[2]). Channel importance weights alpha_k^c are computed via global "
        "average pooling of gradients: alpha_k^c = (1/Z) sum_i sum_j (partial s_c / partial A_{i,j}^k). The coarse 7x7 activation map is computed via "
        "L_GradCAM^c = ReLU(sum_k alpha_k^c A^k) and bilinearly upsampled to 224x224. Because the hook captures the 7x7 refinement bottleneck, the displayed "
        "heatmap is coarse rather than pixel-level, providing regional focus rather than microscopic delineation."
    )
    add_body_p(
        "6) Operating Decision Threshold (tau): For binary screening models (PneumoNet and GastroNet), the positive-class probability p_+ is evaluated "
        "against an empirically determined decision cutoff tau. A scan is flagged as pathological if and only if p_+ >= tau. Operating thresholds were "
        "selected exclusively on validation data: tau = 0.780 for PneumoNet and tau = 0.580 for GastroNet."
    )
    add_body_p(
        "7) Clinical Risk-Tier Assignment: To translate model probabilities into actionable clinical triage levels, MediScan AI maps classifications "
        "into three standardized tiers based on clinical urgency: (i) TumorNet: Glioma and Pituitary adenoma are assigned High Risk (urgent surgical and "
        "neurosurgical oncology referral within 24h); Meningioma is assigned Moderate Risk (elective neurosurgical consultation and volumetric tracking); "
        "No Tumor is assigned Low Risk (routine baseline). (ii) PneumoNet: Scans exceeding tau = 0.780 are assigned High Risk (urgent pediatric clinical review, "
        "SpO2 tracking, empiric antibiotics within 12–24h); scans below tau are Low Risk. (iii) GastroNet: Ulcer detections exceeding tau = 0.580 are assigned "
        "Moderate Risk (outpatient specialist gastroenterology referral and endoscopy staging); normal scans are Low Risk."
    )

    # Table I: Model Architecture Specifications
    p_t1 = doc.add_paragraph()
    p_t1.paragraph_format.space_before = Pt(8)
    p_t1.paragraph_format.space_after = Pt(2)
    r_t1 = p_t1.add_run("TABLE I: MEDISCAN AI ARCHITECTURAL PARAMETERS, CHECKPOINT FOOTPRINT, AND MEMORY USAGE")
    r_t1.font.bold = True
    r_t1.font.size = Pt(8.5)

    tbl_1 = doc.add_table(rows=4, cols=8)
    tbl_1.alignment = WD_TABLE_ALIGNMENT.CENTER
    t1_headers = ["Model", "Modality", "Classes", "Backbone Architecture", "Attention", "Parameters", "FP32 Size", "INT8 Size"]
    t1_data = [
        ["TumorNet", "Brain MRI", "4", "DualPathNet (3x3 + 5x5)", "CBAM", "14,912,870", "59.77 MB", "57.70 MB"],
        ["PneumoNet", "Pediatric CXR", "2", "DualPathNet (3x3 + 5x5)", "CBAM", "14,912,356", "59.77 MB", "57.70 MB"],
        ["GastroNet", "Endoscopy", "2", "DualPathNet (3x3 + 5x5)", "CBAM", "14,912,356", "59.77 MB", "57.70 MB"]
    ]
    for c_idx, h in enumerate(t1_headers):
        cell = tbl_1.cell(0, c_idx)
        set_cell_background(cell, "0F172A")
        set_cell_margins(cell, 80, 80, 100, 100)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(h)
        r.font.bold = True
        r.font.size = Pt(8)
        r.font.color.rgb = RGBColor(0xff, 0xff, 0xff)

    for r_idx, row_vals in enumerate(t1_data):
        shd = "F8FAFC" if r_idx % 2 == 1 else "FFFFFF"
        for c_idx, val in enumerate(row_vals):
            cell = tbl_1.cell(r_idx + 1, c_idx)
            set_cell_background(cell, shd)
            set_cell_margins(cell, 60, 60, 80, 80)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if c_idx not in [0, 3] else WD_ALIGN_PARAGRAPH.LEFT
            r = p.add_run(val)
            r.font.size = Pt(8)

    # --- SECTION IV ---
    add_sec_heading("IV. EXPERIMENTAL DESIGN & METHODOLOGY")
    add_subsec_heading("A. Public Benchmark Datasets")
    add_body_p(
        "All models were developed and benchmarked on established public, de-identified datasets to ensure scientific reproducibility:"
    )
    add_body_p(
        "1) Brain Tumor MRI Dataset (TumorNet): The neuro-oncology model utilizes the aggregated Brain Tumor MRI collection curated by Nickparvar [26] "
        "(incorporating Figshare, SARTAJ, and Br35H public series [27]). The dataset contains 7,200 total axial scans across four classes: glioma (1,800), "
        "meningioma (1,800), no tumor (1,800), and pituitary adenoma (1,800). The benchmark comprises an official pre-partitioned Training set (5,600 images) "
        "and a balanced held-out Testing set (1,600 images; exactly 400 per class). We partitioned the 5,600 training scans into 4,760 training images "
        "(85%, 1,190 per class) and 840 validation images (15%, 210 per class) using stratified random sampling (seed=42). Patient-level metadata was not "
        "provided in the public release; this limitation is addressed in Section VI."
    )
    add_body_p(
        "2) Pediatric Chest Radiograph Dataset (PneumoNet): The thoracic screening model employs the widely utilized Kermany et al. pediatric chest "
        "radiography dataset [28], acquired from children aged 1 to 5 years at Guangzhou Women and Children's Medical Center. The dataset comprises 5,856 "
        "total radiographs in two classes: Normal (1,583) and Pneumonia (4,273). The official held-out test split consists of 624 radiographs (234 Normal, "
        "390 Pneumonia). Because the official validation split contains only 16 images, we pooled the remaining 5,232 images (1,349 Normal, 3,883 Pneumonia) "
        "and performed a stratified 85/15 split: 4,447 training images (1,146 Normal, 3,301 Pneumonia) and 785 validation images (203 Normal, 582 Pneumonia). "
        "The official 624-image test set remained strictly held out and untouched until final evaluation."
    )
    add_body_p(
        "3) HyperKvasir Endoscopy Dataset (GastroNet): The gastrointestinal screening model uses the HyperKvasir dataset [19], gathered at Bærum Hospital, "
        "Norway. We extracted 4,738 total endoscopic images representing normal anatomical landmarks (3,165 images across cecum, ileum, z-line, and normal mucosa) "
        "and mucosal ulcerations (1,573 images). We executed a stratified 70/15/15 random split: 3,317 training images (2,215 Normal, 1,102 Ulcer), 710 validation "
        "images (475 Normal, 235 Ulcer), and 711 held-out test images (475 Normal, 236 Ulcer). Patient-level tracking was unavailable in the public image release."
    )

    # Table II: Hyperparameters
    p_t2 = doc.add_paragraph()
    p_t2.paragraph_format.space_before = Pt(8)
    p_t2.paragraph_format.space_after = Pt(2)
    r_t2 = p_t2.add_run("TABLE II: UNIFIED TRAINING HYPERPARAMETERS AND OPTIMIZATION PROTOCOL ACROSS ALL THREE MODELS")
    r_t2.font.bold = True
    r_t2.font.size = Pt(8.5)

    tbl_2 = doc.add_table(rows=14, cols=4)
    tbl_2.alignment = WD_TABLE_ALIGNMENT.CENTER
    t2_headers = ["Hyperparameter / Setting", "TumorNet (Brain MRI)", "PneumoNet (Pediatric CXR)", "GastroNet (Endoscopy)"]
    t2_data = [
        ["Base Architecture", "DualPathNet (4 classes)", "DualPathNet (2 classes)", "DualPathNet (2 classes)"],
        ["Optimizer & Weight Decay", "AdamW (weight_decay=1e-4)", "AdamW (weight_decay=1e-4)", "AdamW (weight_decay=1e-4)"],
        ["Initial Learning Rate (LR)", "4e-4 (Warmup Cosine Anneal)", "4e-4 (Warmup Cosine Anneal)", "4e-4 (Warmup Cosine Anneal)"],
        ["Warmup & Total Epochs", "2 warmup, 20 total epochs", "2 warmup, 20 total epochs", "2 warmup, 20 total epochs"],
        ["Minimum Learning Rate (eta_min)", "1e-6", "1e-6", "1e-6"],
        ["Batch Size", "32", "32", "32"],
        ["Loss Function", "MultiClassFocalLoss", "ClassWeightedFocalLoss", "ClassWeightedFocalLoss"],
        ["Loss Parameters", "alpha=[1.35, 1.05, 0.90, 0.90], gamma=1.5", "alpha=[1.65, 1.0], gamma=2.0", "alpha=[1.0, 1.35], gamma=2.0"],
        ["Label Smoothing", "0.05", "0.04", "0.03"],
        ["Class-Balanced Sampling", "WeightedRandomSampler", "WeightedRandomSampler", "WeightedRandomSampler"],
        ["Stochastic Weight Averaging", "SWA starting at Epoch 13", "SWA starting at Epoch 13", "SWA starting at Epoch 13"],
        ["Test-Time Augmentation (TTA)", "4-pass (orig, flip, zoom, flip-zoom)", "4-pass (orig, flip, zoom, flip-zoom)", "4-pass (orig, flip, zoom, flip-zoom)"],
        ["Hardware & Compute Environment", "NVIDIA RTX 4050 GPU (AMP), PyTorch 2.6", "NVIDIA RTX 4050 GPU (AMP), PyTorch 2.6", "NVIDIA RTX 4050 GPU (AMP), PyTorch 2.6"]
    ]
    for c_idx, h in enumerate(t2_headers):
        cell = tbl_2.cell(0, c_idx)
        set_cell_background(cell, "0F172A")
        set_cell_margins(cell, 80, 80, 100, 100)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(h)
        r.font.bold = True
        r.font.size = Pt(8)
        r.font.color.rgb = RGBColor(0xff, 0xff, 0xff)

    for r_idx, row_vals in enumerate(t2_data):
        shd = "F8FAFC" if r_idx % 2 == 1 else "FFFFFF"
        for c_idx, val in enumerate(row_vals):
            cell = tbl_2.cell(r_idx + 1, c_idx)
            set_cell_background(cell, shd)
            set_cell_margins(cell, 60, 60, 80, 80)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if c_idx == 0 else WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(val)
            r.font.size = Pt(7.5)

    add_subsec_heading("B. Training Protocol and Regularization")
    add_body_p(
        "Models were trained using AdamW [29] with a base learning rate of 4e-4 and weight decay of 1e-4. A linear learning rate warmup over "
        "2 epochs was followed by cosine annealing decay down to a minimum learning rate of 1e-6 [30]. To address class imbalance without "
        "sacrificing rare classes, WeightedRandomSampler was employed during training. Mixed precision (PyTorch AMP) was utilized throughout. "
        "Stochastic Weight Averaging (SWA) [31] was engaged after epoch 13 (65% of training) to smooth the optimization landscape."
    )
    add_body_p(
        "Rigorous Held-Out Evaluation Principle: All model checkpoints, early stopping decisions (patience=8), and operating decision thresholds "
        "(tau) were selected strictly using the validation sets. The test sets were evaluated exactly once after training completion, guaranteeing "
        "that test metrics remain free of data leakage or threshold tuning bias."
    )

    # Embed Figure 1: Training Curves
    if os.path.exists("paper_figures/fig_training_curves_multipanel.png"):
        doc.add_paragraph().paragraph_format.space_before = Pt(8)
        p_fig1 = doc.add_paragraph()
        p_fig1.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_fig1.paragraph_format.space_after = Pt(2)
        p_fig1.add_run().add_picture("paper_figures/fig_training_curves_multipanel.png", width=Inches(6.8))
        
        p_c1 = doc.add_paragraph()
        p_c1.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_c1.paragraph_format.space_after = Pt(10)
        r_c1 = p_c1.add_run("Fig. 1. Multi-panel training and validation dynamics across DualPathNet models: (a) TumorNet, (b) PneumoNet, and (c) GastroNet. Top row shows training and validation loss; bottom row shows training accuracy alongside validation accuracy, F1-score, and ROC-AUC across training epochs.")
        r_c1.font.size = Pt(8)
        r_c1.font.italic = True

    # --- SECTION V ---
    add_sec_heading("V. RESULTS & QUANTITATIVE EVALUATION")
    add_subsec_heading("A. Benchmark Test Performance")
    add_body_p(
        "Table III summarizes training, validation, and test results across all three imaging domains. Headline results reflect held-out "
        "test set performance, reported with Wilson 95% confidence intervals derived directly from confusion matrices."
    )

    # Table III: Model Overview Table
    p_t3 = doc.add_paragraph()
    p_t3.paragraph_format.space_before = Pt(8)
    p_t3.paragraph_format.space_after = Pt(2)
    r_t3 = p_t3.add_run("TABLE III: SUMMARY OF CLINICAL DATASETS, VALIDATION BENCHMARKS, AND TEST SET PERFORMANCE")
    r_t3.font.bold = True
    r_t3.font.size = Pt(8.5)

    tbl_3 = doc.add_table(rows=4, cols=8)
    tbl_3.alignment = WD_TABLE_ALIGNMENT.CENTER
    t3_headers = ["Model", "Modality", "Dataset Split (Tr/Val/Test)", "Optimal tau", "Val Acc.", "Val AUC", "Test Acc. [95% CI]", "Test Macro AUC"]
    t3_data = [
        ["TumorNet", "Brain MRI", "4,760 / 840 / 1,600", "Argmax", "98.69%", "0.9982", "94.94% [93.8, 95.9]", "0.9874"],
        ["PneumoNet", "Pediatric CXR", "4,447 / 785 / 624", "0.780", "95.92%", "0.9984", "89.58% [86.9, 91.7]", "0.9586"],
        ["GastroNet", "Endoscopy", "3,317 / 710 / 711", "0.580", "96.77%", "0.9933", "96.06% [94.4, 97.3]", "0.9904"]
    ]
    for c_idx, h in enumerate(t3_headers):
        cell = tbl_3.cell(0, c_idx)
        set_cell_background(cell, "0F172A")
        set_cell_margins(cell, 80, 80, 100, 100)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(h)
        r.font.bold = True
        r.font.size = Pt(8)
        r.font.color.rgb = RGBColor(0xff, 0xff, 0xff)

    for r_idx, row_vals in enumerate(t3_data):
        shd = "F8FAFC" if r_idx % 2 == 1 else "FFFFFF"
        for c_idx, val in enumerate(row_vals):
            cell = tbl_3.cell(r_idx + 1, c_idx)
            set_cell_background(cell, shd)
            set_cell_margins(cell, 60, 60, 80, 80)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if c_idx == 0 else WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(val)
            r.font.size = Pt(8)

    # Table IV: Comprehensive Test Metrics with Wilson CIs
    p_t4 = doc.add_paragraph()
    p_t4.paragraph_format.space_before = Pt(8)
    p_t4.paragraph_format.space_after = Pt(2)
    r_t4 = p_t4.add_run("TABLE IV: COMPREHENSIVE CLINICAL TEST SET PERFORMANCE WITH WILSON 95% CONFIDENCE INTERVALS")
    r_t4.font.bold = True
    r_t4.font.size = Pt(8.5)

    tbl_4 = doc.add_table(rows=5, cols=8)
    tbl_4.alignment = WD_TABLE_ALIGNMENT.CENTER
    t4_headers = ["Model (Operating Pt)", "Test N", "Accuracy [95% CI]", "Sensitivity [95% CI]", "Specificity [95% CI]", "Precision [95% CI]", "F1-Score", "ROC-AUC"]
    t4_data = [
        ["TumorNet (Argmax, Macro)", "1,600", "94.94% [93.8, 95.9]", "94.94% [93.8, 95.9]", "98.31% [97.5, 98.8]", "95.24% [94.1, 96.2]", "94.84%", "0.9874"],
        ["PneumoNet (tau=0.78, Pos.)", "624", "89.58% [86.9, 91.7]", "88.21% [84.6, 91.0]", "91.88% [87.7, 94.7]", "94.77% [92.0, 96.6]", "91.37%", "0.9586"],
        ["PneumoNet (tau=0.50, Pos.)", "624", "88.94% [86.2, 91.2]", "96.15% [93.8, 97.7]", "76.92% [71.1, 81.9]", "87.41% [83.9, 90.2]", "91.58%", "0.9586"],
        ["GastroNet (tau=0.58, Pos.)", "711", "96.06% [94.4, 97.3]", "94.49% [90.8, 96.8]", "96.84% [94.9, 98.1]", "93.70% [89.9, 96.1]", "94.09%", "0.9904"]
    ]
    for c_idx, h in enumerate(t4_headers):
        cell = tbl_4.cell(0, c_idx)
        set_cell_background(cell, "0F172A")
        set_cell_margins(cell, 80, 80, 100, 100)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(h)
        r.font.bold = True
        r.font.size = Pt(8)
        r.font.color.rgb = RGBColor(0xff, 0xff, 0xff)

    for r_idx, row_vals in enumerate(t4_data):
        shd = "F8FAFC" if r_idx % 2 == 1 else "FFFFFF"
        for c_idx, val in enumerate(row_vals):
            cell = tbl_4.cell(r_idx + 1, c_idx)
            set_cell_background(cell, shd)
            set_cell_margins(cell, 60, 60, 80, 80)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if c_idx == 0 else WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(val)
            r.font.size = Pt(8)

    # Embed Figure 2: Confusion Matrices
    if os.path.exists("paper_figures/fig_confusion_matrices_multipanel.png"):
        doc.add_paragraph().paragraph_format.space_before = Pt(8)
        p_fig2 = doc.add_paragraph()
        p_fig2.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_fig2.paragraph_format.space_after = Pt(2)
        p_fig2.add_run().add_picture("paper_figures/fig_confusion_matrices_multipanel.png", width=Inches(6.8))
        
        p_c2 = doc.add_paragraph()
        p_c2.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_c2.paragraph_format.space_after = Pt(10)
        r_c2 = p_c2.add_run("Fig. 2. Confusion matrices evaluated on held-out public benchmark test sets: (a) TumorNet across 1,600 MRI scans (400 per class), (b) PneumoNet across 624 pediatric chest radiographs at tau = 0.780, and (c) GastroNet across 711 endoscopic images at tau = 0.580.")
        r_c2.font.size = Pt(8)
        r_c2.font.italic = True

    # Embed Figure 3: ROC Curves
    if os.path.exists("paper_figures/fig_roc_curves_multipanel.png"):
        p_fig3 = doc.add_paragraph()
        p_fig3.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_fig3.paragraph_format.space_after = Pt(2)
        p_fig3.add_run().add_picture("paper_figures/fig_roc_curves_multipanel.png", width=Inches(6.8))
        
        p_c3 = doc.add_paragraph()
        p_c3.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_c3.paragraph_format.space_after = Pt(10)
        r_c3 = p_c3.add_run("Fig. 3. Receiver Operating Characteristic (ROC) curves on test sets: (a) TumorNet one-vs-rest curves across four intracranial classes (macro AUC = 0.987), (b) PneumoNet pneumonia ROC curve (AUC = 0.959) highlighting operating points at tau = 0.780 and tau = 0.500, and (c) GastroNet mucosal ulceration ROC curve (AUC = 0.990).")
        r_c3.font.size = Pt(8)
        r_c3.font.italic = True

    add_subsec_heading("B. Detailed Per-Class Analysis for TumorNet")
    add_body_p(
        "Table V presents the recomputed, verified per-class breakdown for TumorNet across the 1,600 test scans (exactly 400 per class). "
        "The model demonstrates exceptional precision for glioma (99.10%, 332/335) and pituitary adenoma (98.76%, 397/402), alongside perfect "
        "sensitivity for identifying tumor absence (100.00%, 400/400 normal scans correctly classified with 0 false negatives). Meningioma "
        "sensitivity reaches 97.50% (390/400). In contrast, glioma sensitivity is lower at 83.00% (332/400), with 42 scans misclassified as "
        "meningioma (10.5%) and 25 misclassified as no tumor (6.25%). This specific error pattern is examined in Section VI."
    )

    # Table V: TumorNet Per-Class Breakdown
    p_t5 = doc.add_paragraph()
    p_t5.paragraph_format.space_before = Pt(8)
    p_t5.paragraph_format.space_after = Pt(2)
    r_t5 = p_t5.add_run("TABLE V: RECOMPUTED TUMORNET PER-CLASS TEST PERFORMANCE ON 1,600 BRAIN MRI SCANS (400 SCANS PER CLASS)")
    r_t5.font.bold = True
    r_t5.font.size = Pt(8.5)

    tbl_5 = doc.add_table(rows=6, cols=7)
    tbl_5.alignment = WD_TABLE_ALIGNMENT.CENTER
    t5_headers = ["Pathology Class", "Support", "Sensitivity (Recall) [95% CI]", "Precision [95% CI]", "Specificity [95% CI]", "F1-Score", "TP / Target"]
    t5_data = [
        ["Glioma", "400", "83.00% [79.0, 86.4]", "99.10% [97.4, 99.7]", "99.75% [99.3, 99.9]", "90.34%", "332 / 400"],
        ["Meningioma", "400", "97.50% [95.5, 98.6]", "89.86% [86.7, 92.4]", "96.33% [95.1, 97.3]", "93.53%", "390 / 400"],
        ["No Tumor (Normal)", "400", "100.00% [99.0, 100.0]", "93.24% [90.5, 95.3]", "97.58% [96.6, 98.3]", "96.50%", "400 / 400"],
        ["Pituitary Adenoma", "400", "99.25% [97.8, 99.7]", "98.76% [97.1, 99.5]", "99.58% [99.0, 99.8]", "99.00%", "397 / 400"],
        ["Macro Average / Total", "1,600", "94.94% [93.8, 95.9]", "95.24% [94.1, 96.2]", "98.31% [97.5, 98.8]", "94.84%", "1,519 / 1,600"]
    ]
    for c_idx, h in enumerate(t5_headers):
        cell = tbl_5.cell(0, c_idx)
        set_cell_background(cell, "0F172A")
        set_cell_margins(cell, 80, 80, 100, 100)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(h)
        r.font.bold = True
        r.font.size = Pt(8)
        r.font.color.rgb = RGBColor(0xff, 0xff, 0xff)

    for r_idx, row_vals in enumerate(t5_data):
        shd = "F1F5F9" if r_idx == 4 else ("F8FAFC" if r_idx % 2 == 1 else "FFFFFF")
        for c_idx, val in enumerate(row_vals):
            cell = tbl_5.cell(r_idx + 1, c_idx)
            set_cell_background(cell, shd)
            set_cell_margins(cell, 60, 60, 80, 80)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if c_idx == 0 else WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(val)
            r.font.bold = (r_idx == 4)
            r.font.size = Pt(8)

    add_subsec_heading("C. Operating-Point Threshold Sweeps")
    add_body_p(
        "Table VI illustrates the sensitivity versus specificity trade-off across varying decision cutoffs (tau) for PneumoNet and GastroNet. "
        "At tau = 0.500, PneumoNet achieves 96.15% sensitivity on pediatric radiographs (375 of 390 pneumonia cases detected; only 15 missed), "
        "at the expense of lower specificity (76.92%, 54 false alarms). At the validation-selected operating cutoff of tau = 0.780, specificity rises "
        "to 91.88% (false alarms fall to 19), while sensitivity adjusts to 88.21% (344 detected, 46 missed). Similarly, GastroNet achieves 95.34% sensitivity "
        "at tau = 0.500 and 94.49% sensitivity with 96.84% specificity at tau = 0.580. Thus, tau = 0.780 operates as a specificity-protecting threshold "
        "designed to prevent specialist referral fatigue, whereas primary triage centers prioritizing case capture can lower tau to 0.500."
    )

    # Table VI: Threshold Sweeps
    p_t6 = doc.add_paragraph()
    p_t6.paragraph_format.space_before = Pt(8)
    p_t6.paragraph_format.space_after = Pt(2)
    r_t6 = p_t6.add_run("TABLE VI: EMPIRICAL OPERATING THRESHOLD (tau) SWEEPS ON HELD-OUT TEST COHORTS")
    r_t6.font.bold = True
    r_t6.font.size = Pt(8.5)

    tbl_6 = doc.add_table(rows=10, cols=8)
    tbl_6.alignment = WD_TABLE_ALIGNMENT.CENTER
    t6_headers = ["Model", "Threshold (tau)", "Accuracy", "Sensitivity", "Specificity", "Precision", "Positive F1", "TP / FP / TN / FN"]
    t6_data = [
        ["PneumoNet (CXR)", "0.500", "88.94%", "96.15%", "76.92%", "87.41%", "91.58%", "375 / 54 / 180 / 15"],
        ["PneumoNet (CXR)", "0.600", "89.90%", "94.36%", "82.48%", "89.98%", "92.12%", "368 / 41 / 193 / 22"],
        ["PneumoNet (CXR)", "0.700", "90.06%", "91.54%", "87.61%", "92.49%", "92.01%", "357 / 29 / 205 / 33"],
        ["PneumoNet (CXR)", "0.780 (Val Best)", "89.58%", "88.21%", "91.88%", "94.77%", "91.37%", "344 / 19 / 215 / 46"],
        ["PneumoNet (CXR)", "0.850", "86.70%", "82.05%", "94.44%", "96.10%", "88.52%", "320 / 13 / 221 / 70"],
        ["GastroNet (Endo)", "0.500", "94.94%", "95.34%", "94.74%", "90.00%", "92.59%", "225 / 25 / 450 / 11"],
        ["GastroNet (Endo)", "0.580 (Val Best)", "96.06%", "94.49%", "96.84%", "93.70%", "94.09%", "223 / 15 / 460 / 13"],
        ["GastroNet (Endo)", "0.650", "96.62%", "93.22%", "98.32%", "96.49%", "94.83%", "220 / 8 / 467 / 16"],
        ["GastroNet (Endo)", "0.750", "94.23%", "83.47%", "99.58%", "98.99%", "90.57%", "197 / 2 / 473 / 39"]
    ]
    for c_idx, h in enumerate(t6_headers):
        cell = tbl_6.cell(0, c_idx)
        set_cell_background(cell, "0F172A")
        set_cell_margins(cell, 80, 80, 100, 100)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(h)
        r.font.bold = True
        r.font.size = Pt(8)
        r.font.color.rgb = RGBColor(0xff, 0xff, 0xff)

    for r_idx, row_vals in enumerate(t6_data):
        shd = "F8FAFC" if r_idx % 2 == 1 else "FFFFFF"
        for c_idx, val in enumerate(row_vals):
            cell = tbl_6.cell(r_idx + 1, c_idx)
            set_cell_background(cell, shd)
            set_cell_margins(cell, 60, 60, 80, 80)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if c_idx in [0, 1] else WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(val)
            r.font.size = Pt(7.5)

    add_subsec_heading("D. Measured CPU Inference Latency")
    add_body_p(
        "To validate feasibility for low-resource district clinics lacking dedicated GPUs, inference latency was benchmarked on a commodity "
        "desktop CPU (Intel Core i7-13700HX pinned to 4 worker threads, batch size = 1, N=25 iterations following warmup). As reported in Table VII, "
        "a single standard forward pass requires only 101.9–108.9 ms across all three models. Single forward pass coupled with Grad-CAM generation "
        "requires 302.0–306.9 ms. Even under the complete production pipeline—combining 4-pass test-time augmentation (original, horizontal flip, "
        "zoom-crop, and flipped zoom-crop) with backpropagated bottleneck Grad-CAM—inference requires only 722.0–755.7 ms (p95 < 800 ms). "
        "This confirms empirical sub-second response times without specialized hardware."
    )

    # Table VII: Measured Latency
    p_t7 = doc.add_paragraph()
    p_t7.paragraph_format.space_before = Pt(8)
    p_t7.paragraph_format.space_after = Pt(2)
    r_t7 = p_t7.add_run("TABLE VII: MEASURED CPU INFERENCE LATENCY (INTEL CORE I7-13700HX, 4 THREADS, BATCH SIZE = 1)")
    r_t7.font.bold = True
    r_t7.font.size = Pt(8.5)

    tbl_7 = doc.add_table(rows=4, cols=5)
    tbl_7.alignment = WD_TABLE_ALIGNMENT.CENTER
    t7_headers = ["Model", "Single Forward (No TTA, No CAM)", "4-Pass TTA (No CAM)", "Single Forward + Grad-CAM", "4-Pass TTA + Grad-CAM (Full)"]
    t7_data = [
        ["TumorNet (Brain MRI)", "101.9 ms (p95: 107.0 ms)", "413.4 ms (p95: 432.5 ms)", "303.0 ms (p95: 326.1 ms)", "722.0 ms (p95: 752.7 ms)"],
        ["PneumoNet (Pediatric CXR)", "105.0 ms (p95: 107.3 ms)", "428.3 ms (p95: 468.7 ms)", "302.0 ms (p95: 311.7 ms)", "731.0 ms (p95: 761.2 ms)"],
        ["GastroNet (Endoscopy)", "108.9 ms (p95: 116.5 ms)", "435.6 ms (p95: 456.5 ms)", "306.9 ms (p95: 329.0 ms)", "755.7 ms (p95: 797.6 ms)"]
    ]
    for c_idx, h in enumerate(t7_headers):
        cell = tbl_7.cell(0, c_idx)
        set_cell_background(cell, "0F172A")
        set_cell_margins(cell, 80, 80, 100, 100)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(h)
        r.font.bold = True
        r.font.size = Pt(8)
        r.font.color.rgb = RGBColor(0xff, 0xff, 0xff)

    for r_idx, row_vals in enumerate(t7_data):
        shd = "F8FAFC" if r_idx % 2 == 1 else "FFFFFF"
        for c_idx, val in enumerate(row_vals):
            cell = tbl_7.cell(r_idx + 1, c_idx)
            set_cell_background(cell, shd)
            set_cell_margins(cell, 60, 60, 80, 80)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if c_idx == 0 else WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(val)
            r.font.size = Pt(7.5)

    add_subsec_heading("E. Baseline Backbone Comparisons and Architectural Ablation")
    add_body_p(
        "Table VIII benchmarks DualPathNet against standard baseline vision backbones trained under identical protocol constraints. "
        "DualPathNet achieves competitive test accuracy across all three modalities (94.94% on brain MRI, 89.58% on pediatric CXR, 96.06% on endoscopy) "
        "with 14.91M parameters, avoiding the memory overhead of deeper 25M+ parameter models while natively integrating bottleneck attention. "
        "Table IX details the component-wise ablation of DualPathNet: isolating single 3x3 or 5x5 branches, removing CBAM attention, utilizing Global "
        "Average Pooling exclusively without Global Max Pooling, or omitting residual skip connections. The complete dual-path configuration with "
        "dual pooling and CBAM attention delivers superior stability across both fine-textured ulcer margins and macro-structural consolidations."
    )

    # Table VIII: Baselines Table
    p_t8 = doc.add_paragraph()
    p_t8.paragraph_format.space_before = Pt(8)
    p_t8.paragraph_format.space_after = Pt(2)
    r_t8 = p_t8.add_run("TABLE VIII: COMPARISON AGAINST STANDARD BASELINE BACKBONES UNDER EQUIVALENT COMPUTATIONAL PROTOCOLS")
    r_t8.font.bold = True
    r_t8.font.size = Pt(8.5)

    tbl_8 = doc.add_table(rows=7, cols=7)
    tbl_8.alignment = WD_TABLE_ALIGNMENT.CENTER
    t8_headers = ["Model Architecture", "Parameters", "GFLOPs", "CPU Latency", "Brain MRI Test Acc.", "Pneumonia Test Acc.", "Endoscopy Test Acc."]
    t8_data = [
        ["ResNet-18 [22]", "11.69M", "1.82", "72.4 ms", "92.15%", "87.34%", "93.81%"],
        ["ResNet-50 [22]", "25.56M", "4.12", "148.6 ms", "93.40%", "88.62%", "94.75%"],
        ["EfficientNet-B0 [16]", "5.29M", "0.39", "64.2 ms", "91.80%", "86.85%", "93.10%"],
        ["DenseNet-121 [7]", "7.98M", "2.88", "112.5 ms", "92.95%", "88.10%", "94.20%"],
        ["MobileNetV3-Large [18]", "5.48M", "0.22", "48.1 ms", "90.75%", "85.90%", "92.40%"],
        ["DualPathNet (Ours, Scratch)", "14.91M", "3.24", "101.9 ms", "94.94%", "89.58%", "96.06%"]
    ]
    for c_idx, h in enumerate(t8_headers):
        cell = tbl_8.cell(0, c_idx)
        set_cell_background(cell, "0F172A")
        set_cell_margins(cell, 80, 80, 100, 100)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(h)
        r.font.bold = True
        r.font.size = Pt(8)
        r.font.color.rgb = RGBColor(0xff, 0xff, 0xff)

    for r_idx, row_vals in enumerate(t8_data):
        shd = "F1F5F9" if r_idx == 5 else ("F8FAFC" if r_idx % 2 == 1 else "FFFFFF")
        for c_idx, val in enumerate(row_vals):
            cell = tbl_8.cell(r_idx + 1, c_idx)
            set_cell_background(cell, shd)
            set_cell_margins(cell, 60, 60, 80, 80)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if c_idx == 0 else WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(val)
            r.font.bold = (r_idx == 5)
            r.font.size = Pt(7.5)

    # Table IX: Ablation Table
    p_t9 = doc.add_paragraph()
    p_t9.paragraph_format.space_before = Pt(8)
    p_t9.paragraph_format.space_after = Pt(2)
    r_t9 = p_t9.add_run("TABLE IX: COMPONENT-WISE ARCHITECTURAL ABLATION STUDY OF DUALPATHNET")
    r_t9.font.bold = True
    r_t9.font.size = Pt(8.5)

    tbl_9 = doc.add_table(rows=7, cols=5)
    tbl_9.alignment = WD_TABLE_ALIGNMENT.CENTER
    t9_headers = ["DualPathNet Architectural Variant", "Parameters", "Brain MRI Test Acc.", "Pediatric CXR Test Acc.", "Endoscopy Test Acc."]
    t9_data = [
        ["Full DualPathNet (3x3 + 5x5 + CBAM + GAP/GMP)", "14.91M", "94.94%", "89.58%", "96.06%"],
        ["Single-Path A Only (3x3 Fine Kernels Only)", "11.61M", "92.62%", "86.86%", "93.53%"],
        ["Single-Path B Only (5x5 Broad Kernels Only)", "13.69M", "91.88%", "87.50%", "92.83%"],
        ["Without CBAM Attention Module", "14.88M", "93.19%", "87.98%", "94.23%"],
        ["GAP-Only Pooling (No Global Max Pooling)", "14.65M", "93.81%", "88.46%", "94.79%"],
        ["Without Residual Shortcut Connections (Plain CNN)", "14.91M", "88.44%", "83.97%", "89.17%"]
    ]
    for c_idx, h in enumerate(t9_headers):
        cell = tbl_9.cell(0, c_idx)
        set_cell_background(cell, "0F172A")
        set_cell_margins(cell, 80, 80, 100, 100)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(h)
        r.font.bold = True
        r.font.size = Pt(8)
        r.font.color.rgb = RGBColor(0xff, 0xff, 0xff)

    for r_idx, row_vals in enumerate(t9_data):
        shd = "F1F5F9" if r_idx == 0 else ("F8FAFC" if r_idx % 2 == 1 else "FFFFFF")
        for c_idx, val in enumerate(row_vals):
            cell = tbl_9.cell(r_idx + 1, c_idx)
            set_cell_background(cell, shd)
            set_cell_margins(cell, 60, 60, 80, 80)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if c_idx == 0 else WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(val)
            r.font.bold = (r_idx == 0)
            r.font.size = Pt(7.5)

    # Embed Figure 4: Grad-CAM Examples
    if os.path.exists("paper_figures/fig_gradcam_multipanel.png"):
        doc.add_paragraph().paragraph_format.space_before = Pt(8)
        p_fig4 = doc.add_paragraph()
        p_fig4.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_fig4.paragraph_format.space_after = Pt(2)
        p_fig4.add_run().add_picture("paper_figures/fig_gradcam_multipanel.png", width=Inches(6.8))
        
        p_c4 = doc.add_paragraph()
        p_c4.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_c4.paragraph_format.space_after = Pt(10)
        r_c4 = p_c4.add_run("Fig. 4. Visual explainability across all three clinical modalities using coarse bottleneck Grad-CAM (7x7 feature maps upsampled to 224x224): (a) TumorNet localization of an infiltrative glial neoplasm, (b) PneumoNet localization of dense right-sided pediatric parenchymal consolidation, and (c) GastroNet localization of focal mucosal ulceration.")
        r_c4.font.size = Pt(8)
        r_c4.font.italic = True

    # --- SECTION VI ---
    add_sec_heading("VI. DISCUSSION, ERROR ANALYSIS, & LIMITATIONS")
    add_subsec_heading("A. Diagnostic Findings and Error Analysis")
    add_body_p(
        "Quantitative evaluation yields three key insights. First, validation metrics exceed test metrics across all three domains, most notably in "
        "PneumoNet where validation accuracy (95.92%, AUC 0.9984) drops to 89.58% accuracy (AUC 0.9586) on the official test set. This drop reflects "
        "an inherent distribution shift between the training cohort and the official test set (collected across varying time windows and clinical severities). "
        "Second, per-class error analysis reveals that 42 of 400 glioma cases (10.5%) were misclassified as meningioma. Histologically and radiographically, "
        "low-grade or dural-abutting gliomas can exhibit circumscribed, homogeneous signal intensities on axial slices that closely resemble extra-axial "
        "meningiomas without clear sulcal effacement. An additional 25 glioma cases were classified as normal (6.25%), underscoring that TumorNet must "
        "function as an assistive screening tool rather than a standalone rule-out mechanism. Third, the safety-threshold analysis clarifies that selecting "
        "tau = 0.780 protects specificity (limiting false positive referrals to 19), whereas frontline triage prioritizing sensitivity should choose "
        "tau = 0.500 (capturing 96.15% of pneumonia cases with only 15 false negatives)."
    )
    add_subsec_heading("B. Study Limitations")
    add_body_p(
        "To uphold scientific rigor, the following limitations must be acknowledged:"
    )
    add_body_p(
        "1) Retrospective Public Benchmarks: All experiments rely on retrospective, publicly accessible datasets. The system has not undergone prospective "
        "multi-center clinical validation across district hospitals or community primary care facilities."
    )
    add_body_p(
        "2) Single-Center Pediatric CXR Data: The chest radiograph dataset originates from a single pediatric hospital center in Guangzhou, China. "
        "Generalizability to differing pediatric age brackets, nutritional statuses, or concurrent tuberculosis burdens in rural LMIC settings remains unproven."
    )
    add_body_p(
        "3) Potential Cross-Split Patient Leakage: Because the public Brain Tumor MRI and HyperKvasir datasets lack patient-level unique identifiers, "
        "stratified splits were performed at the image level. While test partitions were strictly held out, the theoretical possibility of patient-level overlap "
        "across slices cannot be entirely excluded."
    )
    add_body_p(
        "4) Coarse Spatial Resolution of Grad-CAM: Saliency heatmaps originate from the 7x7 refinement bottleneck (model.refine[2]) and are bilinearly "
        "interpolated to 224x224. They provide regional visual attention rather than sharp lesion boundary segmentation, and localization faithfulness has "
        "not been evaluated quantitatively against expert radiologist bounding-box annotations."
    )
    add_body_p(
        "5) Absence of Clinical Reader Trials: We did not conduct a prospective reader study with practicing radiologists or gastroenterologists to quantify "
        "whether Grad-CAM overlays improve clinician diagnostic accuracy, reduce reading time, or mitigate automation bias."
    )
    add_body_p(
        "6) Unvalidated Translation Usability: While the user interface includes a synchronized Hindi (hi-IN) localization, its usability, clinical vocabulary "
        "accuracy, and acceptability among accredited social health activists (ASHA) and rural auxiliary nurse midwives have not been formally evaluated in field trials."
    )

    # --- SECTION VII ---
    add_sec_heading("VII. CONCLUSION & FUTURE WORK")
    add_body_p(
        "This work introduced MediScan AI, an integrated multiscale deep-learning prototype for cross-modality medical screening. Powered by DualPathNet "
        "(14.91M parameters), the system couples parallel 3x3 and 5x5 residual convolutional streams with CBAM attention and dual pooling, demonstrating "
        "competitive performance from scratch without pre-trained transfer learning: 94.94% accuracy and 0.9874 macro AUC on 1,600 brain MRI scans; 89.58% accuracy "
        "with 91.88% specificity (tau = 0.780) and 96.15% sensitivity (tau = 0.500) on 624 pediatric chest radiographs; and 96.06% accuracy with 94.49% sensitivity "
        "on 711 endoscopic images. Coarse bottleneck Grad-CAM heatmaps deliver spatial visual attribution to support clinician review, while CPU latency benchmarking "
        "confirms sub-second throughput (101.9–108.9 ms single-pass; 722.0–755.7 ms for 4-pass TTA plus Grad-CAM), verifying feasibility on commodity clinic hardware."
    )
    add_body_p(
        "Future work will prioritize: (i) multi-center prospective validation across district healthcare centers in Maharashtra, India; (ii) formal clinical reader "
        "trials assessing diagnostic accuracy with and without Grad-CAM assistance; (iii) post-training dynamic INT8 quantization to further reduce memory footprint; "
        "and (iv) implementing longitudinal audit trails (G5) and structured FHIR-compliant electronic health record exports (G6)."
    )

    # Statements
    add_sec_heading("DECLARATIONS & COMPLIANCE")
    add_body_p(
        "Ethics Statement: This study utilized exclusively publicly available, fully de-identified imaging datasets. All original data collection protocols "
        "operated under institutional review board approvals and informed consent as documented in the primary dataset publications [19], [26], [28]."
    )
    add_body_p(
        "Data and Code Availability: Model architectures, evaluation scripts, trained PyTorch weights, and reproducible Jupyter notebooks are open-sourced "
        "at: https://github.com/v-mandloi/MediScan-AI. Dataset partitions and splits conform strictly to the respective public licenses."
    )
    add_body_p(
        "Conflict of Interest: The authors declare that they have no financial or commercial conflicts of interest."
    )
    add_body_p(
        "Funding: This research received no specific grant from funding agencies in the public, commercial, or not-for-profit sectors."
    )

    # --- REFERENCES ---
    add_sec_heading("REFERENCES")
    refs_list = [
        "[1] W. Khan, N. Zaki, and L. Ali, \"Intelligent pneumonia identification from chest X-rays: A systematic literature review,\" IEEE Access, vol. 9, pp. 51747–51771, 2021. DOI: 10.1109/ACCESS.2021.3069937.",
        "[2] S. Solanki, U. P. Singh, S. S. Chouhan, and S. Jain, \"Brain tumor detection and classification using intelligence techniques: An overview,\" IEEE Access, vol. 11, pp. 12870–12886, 2023. DOI: 10.1109/ACCESS.2023.3242666.",
        "[3] J.-Y. Sun, S.-W. Lee, M.-C. Kang, S.-W. Kim, S.-Y. Kim, and S.-J. Ko, \"A novel gastric ulcer differentiation system using convolutional neural networks,\" in Proc. IEEE 31st Int. Symp. Comput.-Based Med. Syst. (CBMS), 2018, pp. 351–356. DOI: 10.1109/CBMS.2018.00068.",
        "[4] S. Sharma and K. Guleria, \"A deep learning based model for the detection of pneumonia from chest X-ray images using VGG-16 and neural networks,\" Procedia Comput. Sci., vol. 218, pp. 357–366, 2023. DOI: 10.1016/j.procs.2023.01.018.",
        "[5] N. Noreen, S. Palaniappan, A. Qayyum, I. Ahmad, M. Imran, and M. Shoaib, \"A deep learning model based on concatenation approach for the diagnosis of brain tumor,\" IEEE Access, vol. 8, pp. 55135–55144, 2020. DOI: 10.1109/ACCESS.2020.2981324.",
        "[6] S. Hatami, M. H. Olyaei, and R. Shamsaee, \"Detection and classification of gastric precancerous diseases using deep learning,\" in Proc. 6th Iranian Conf. Signal Process. Intell. Syst. (ICSPIS), 2020, pp. 1–5. DOI: 10.1109/ICSPIS51611.2020.9349589.",
        "[7] H. Ren, C. Sun, J. Yang, and S. Dong, \"Interpretable pneumonia detection by combining deep learning and explainable models with multisource data,\" IEEE Access, vol. 9, pp. 95523–95533, 2021. DOI: 10.1109/ACCESS.2021.3093221.",
        "[8] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, \"Grad-CAM: Visual explanations from deep networks via gradient-based localization,\" in Proc. IEEE Int. Conf. Comput. Vis. (ICCV), 2017, pp. 618–626. DOI: 10.1109/ICCV.2017.74.",
        "[9] F. Provost and T. Fawcett, \"Robust classification for imprecise environments,\" Mach. Learn., vol. 42, no. 3, pp. 203–231, 2001. DOI: 10.1023/A:1007601015854.",
        "[10] A. Younis et al., \"Abnormal brain tumors classification using ResNet50 and its comprehensive evaluation,\" IEEE Access, vol. 12, pp. 80470–80480, 2024. DOI: 10.1109/ACCESS.2024.3408800.",
        "[11] S. Asif, W. Yi, Q. U. Ain, J. Hou, T. Yi, and J. Si, \"Improving effectiveness of different deep transfer learning-based models for detecting brain tumors from MR images,\" IEEE Access, vol. 10, pp. 34716–34730, 2022. DOI: 10.1109/ACCESS.2022.3162381.",
        "[12] S. Ahmad and P. K. Choudhury, \"On the performance of deep transfer learning networks for brain tumor detection using MR images,\" IEEE Access, vol. 10, pp. 59541–59556, 2022. DOI: 10.1109/ACCESS.2022.3179836.",
        "[13] M. Rizwan, A. Shabbir, A. R. Javed, M. Shabbir, T. Baker, and D. Al-Jumeily, \"Brain tumor and glioma grade classification using Gaussian convolutional neural network,\" IEEE Access, vol. 10, pp. 29731–29740, 2022. DOI: 10.1109/ACCESS.2022.3153028.",
        "[14] Z. Jia and D. Chen, \"Brain tumor identification and classification of MRI images using deep learning techniques,\" IEEE Access, vol. 8, pp. 123783–123797, 2020. DOI: 10.1109/ACCESS.2020.3006764.",
        "[15] D. Rickard, M. A. Kabir, and N. Homaira, \"Machine learning-based approaches for distinguishing viral and bacterial pneumonia in paediatrics: A scoping review,\" Comput. Methods Programs Biomed., vol. 258, p. 108512, 2025. DOI: 10.1016/j.cmpb.2024.108512.",
        "[16] M. Ali et al., \"Pneumonia detection using chest radiographs with novel EfficientNetV2L model,\" IEEE Access, vol. 12, pp. 24456–24467, 2024. DOI: 10.1109/ACCESS.2024.3364402.",
        "[17] S. Potharaju, S. N. Tambe, K. Dasari, N. Srikanth, R. Venkatarao, and S. Tambe, \"Enhanced X-ray image classification for pneumonia detection using deep learning based CBAM and SE mechanisms,\" Intell.-Based Med., vol. 11, p. 100142, 2025. DOI: 10.1016/j.ibmed.2024.100142.",
        "[18] N. Shilpa, W. A. Banu, and P. B. Metre, \"Revolutionizing pneumonia diagnosis: AI-driven deep learning framework for automated detection from chest X-rays,\" IEEE Access, vol. 12, pp. 171601–171616, 2024. DOI: 10.1109/ACCESS.2024.3498870.",
        "[19] P. H. Borgli, V. Thambawita, P. Smedsrud, S. Hicks, H. Jha, S. L. Eskeland, et al., \"HyperKvasir, a comprehensive multi-class image and video dataset for gastrointestinal endoscopy,\" Sci. Data, vol. 7, no. 1, p. 283, 2020. DOI: 10.1038/s41597-020-00622-y.",
        "[20] H. Olimboyev et al., \"Development of the Gastro-AI model for diagnosing gastrointestinal diseases based on the HyperKvasir,\" in Proc. IEEE Conf. Artif. Intell. Med., 2024, pp. 1–6. DOI: 10.1109/AIMED61304.2024.10541280.",
        "[21] Y. Tan et al., \"Deep learning-based AI system for real-time differentiation of benign and malignant gastric ulcers,\" BMC Gastroenterol., vol. 26, p. 378, 2026. DOI: 10.1186/s12876-026-03780-1.",
        "[22] K. He, X. Zhang, S. Ren, and J. Sun, \"Deep residual learning for image recognition,\" in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2016, pp. 770–778. DOI: 10.1109/CVPR.2016.90.",
        "[23] S. Woo, J. Park, J.-Y. Lee, and I. S. Kweon, \"CBAM: Convolutional block attention module,\" in Proc. Eur. Conf. Comput. Vis. (ECCV), 2018, pp. 3–19. DOI: 10.1007/978-3-030-01234-2_1.",
        "[24] K. Zuiderveld, \"Contrast limited adaptive histogram equalization,\" in Graphics Gems IV, P. S. Heckbert, Ed. San Diego, CA, USA: Academic Press, 1994, pp. 474–485. DOI: 10.1016/B978-0-12-336156-1.50061-6.",
        "[25] S. Elfwing, E. Uchibe, and K. Doya, \"Sigmoid-weighted linear units for neural network function approximation in reinforcement learning,\" Neural Netw., vol. 107, pp. 3–11, 2018. DOI: 10.1016/j.neunet.2018.07.019.",
        "[26] M. Nickparvar, \"Brain Tumor MRI Dataset,\" Kaggle, 2021. DOI: 10.34740/KAGGLE/DSV/2645886.",
        "[27] J. Cheng et al., \"Enhanced performance of brain tumor classification via tumor region augmentation,\" PLOS ONE, vol. 10, no. 12, p. e0144481, 2015. DOI: 10.1371/journal.pone.0144481.",
        "[28] D. S. Kermany, M. Goldbaum, W. Cai, C. C. Valentim, H. Liang, S. L. Baxter, et al., \"Identifying medical diagnoses and treatable diseases by image-based deep learning,\" Cell, vol. 172, no. 5, pp. 1122–1131, 2018. DOI: 10.1016/j.cell.2018.02.010.",
        "[29] I. Loshchilov and F. Hutter, \"Decoupled weight decay regularization,\" in Proc. Int. Conf. Learn. Represent. (ICLR), 2019, pp. 1–10.",
        "[30] I. Loshchilov and F. Hutter, \"SGDR: Stochastic gradient descent with warm restarts,\" in Proc. Int. Conf. Learn. Represent. (ICLR), 2017, pp. 1–16.",
        "[31] P. Izmailov, D. Podoprikhin, T. Garipov, D. Vetrov, and A. G. Wilson, \"Averaging weights leads to wider optima and better generalization,\" in Proc. Conf. Uncertainty Artif. Intell. (UAI), 2018, pp. 876–885.",
        "[32] T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollár, \"Focal loss for dense object detection,\" in Proc. IEEE Int. Conf. Comput. Vis. (ICCV), 2017, pp. 2980–2988. DOI: 10.1109/ICCV.2017.324.",
        "[33] C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens, and Z. Wojna, \"Rethinking the Inception architecture for computer vision,\" in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2016, pp. 2818–2826. DOI: 10.1109/CVPR.2016.308."
    ]

    for ref in refs_list:
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.left_indent = Inches(0.25)
        p.paragraph_format.first_line_indent = Inches(-0.25)
        r = p.add_run(ref)
        r.font.name = 'Times New Roman'
        r.font.size = Pt(8)

    # Author Biographies
    add_sec_heading("AUTHOR BIOGRAPHIES")
    bios = [
        ("Daksh Gurjar", "is an undergraduate researcher in the Department of Computer Engineering at SVKM's NMIMS MPSTME, Shirpur, India. His research interests include computer-aided diagnosis, multiscale deep learning architectures, explainable artificial intelligence (XAI), and scalable medical tele-triage systems."),
        ("Aditya Yashovardhan", "is an undergraduate researcher in the Department of Computer Engineering at SVKM's NMIMS MPSTME, Shirpur, India. His research focuses on applied machine learning in healthcare, clinical imaging pipelines, and full-stack biomedical software engineering."),
        ("Jayesh Mandloi", "is an undergraduate researcher in the Department of Computer Engineering at SVKM's NMIMS MPSTME, Shirpur, India. His research interests include computer vision, attention mechanisms, and medical decision-support systems for resource-constrained clinical environments.")
    ]
    for name, bio in bios:
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(4)
        r_n = p.add_run(name + " ")
        r_n.font.name = 'Times New Roman'
        r_n.font.bold = True
        r_n.font.size = Pt(8.5)
        r_b = p.add_run(bio)
        r_b.font.name = 'Times New Roman'
        r_b.font.size = Pt(8.5)

    output_path = "MediScan_AI_Conference_Paper_v2.docx"
    doc.save(output_path)
    print(f"Successfully generated revised IEEE conference paper -> {output_path}")

if __name__ == "__main__":
    create_document()
