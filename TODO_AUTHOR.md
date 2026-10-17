# TODO for Authors: Camera-Ready & Submission Checklist

This checklist outlines the remaining author-specific actions required before final submission of the revised paper to an IEEE conference or journal (e.g., IEEE JBHI, TBME, EMBC, BIBM).

---

### 1. Author Metadata and Affiliations
- [ ] **Author Names and Degrees**: In `conference_paper_v2.tex` (lines 20--26) and `MediScan_AI_Conference_Paper_v2.docx`, replace the placeholder `"Author Name(s) Withheld for Double-Blind Review"` with the actual author names, academic affiliations, institutional addresses, and official email addresses.
- [ ] **Funding and Grant Acknowledgments**: If research funding, university grants, or compute sponsorships supported this project, add an unnumbered footnote or acknowledgment statement (e.g., `"This research was supported by Grant No. XXX..."`).
- [ ] **Corresponding Author Designation**: Explicitly mark the corresponding author with an asterisk (`*`) and include their direct email.

---

### 2. Target Venue Formatting and Page Constraints
- [ ] **Select Target Venue**:
  - *IEEE Conference (e.g., EMBC, BIBM, ISBI)*: Typical page limit is 4 to 6 pages (standard IEEE two-column).
  - *IEEE Journal / Transactions (e.g., JBHI, TBME)*: Typically 8 to 12 pages in IEEE transactions template.
- [ ] **Page Limit Check**:
  - The revised manuscript contains 5 multi-panel figures and 9 comprehensive tables. If submitting to a 4-page conference, move Tables VI, VII, and IX to supplementary materials or combine them. If submitting to a 6-to-8 page venue, the current layout is optimal.

---

### 3. Open Source Repository & Checkpoints
- [ ] **GitHub / Zenodo URL**:
  - In Section VIII (Data and Code Availability), update the anonymous placeholder URL (`https://github.com/anonymous-author/mediscan-ai`) with your permanent repository link or Zenodo DOI for public release.
- [ ] **Include Checkpoints**: Ensure model weights (`checkpoints/TumorNet_best.pth`, `PneumoNet_best.pth`, `GastroNet_best.pth`) and evaluation scripts are archived or uploaded as release artifacts.

---

### 4. Ethics, Institutional Review Board (IRB), and Data Governance
- [ ] **IRB Exemption Statement**: Because all three datasets (Brain Tumor MRI, Kermany CXR, HyperKvasir) are public, fully de-identified open-access benchmarks, include a formal ethics disclaimer:
  > *"Ethical approval and patient consent were not required as this study analyzed secondary, fully anonymized public datasets in accordance with institutional guidelines."*
- [ ] **Patient Master ID Clarification**: If patient metadata or hospital medical record identifiers become available from the dataset creators, confirm whether the 1,600 MRI slices or 4,738 endoscopy frames derive from unique patient encounters to strengthen the patient-level split claim.

---

### 5. Conversion to PDF
- [ ] **Word Document (`MediScan_AI_Conference_Paper_v2.docx`)**:
  - Open `MediScan_AI_Conference_Paper_v2.docx` in Microsoft Word or LibreOffice.
  - Review formatting, line breaks, and image alignments.
  - Export/Save As PDF: `MediScan_AI_Conference_Paper_v2.pdf`.
- [ ] **LaTeX Compilation (`conference_paper_v2.tex`)**:
  - Upload `conference_paper_v2.tex` along with the `paper_figures/` folder to Overleaf or compile locally using `pdflatex conference_paper_v2.tex` and `bibtex conference_paper_v2`.
  - The source is 100% clean, standard IEEEtran compliant, with no external package conflicts.
