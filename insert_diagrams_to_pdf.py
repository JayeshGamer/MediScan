"""
Insert diagrams into the literature review PDF.

This script reads the original literature_review.pdf, identifies the correct
insertion points for each diagram based on section content, and creates a new
PDF (literature_review_with_diagrams.pdf) with diagram pages inserted at the
appropriate locations.

Diagram placement:
  - Diagram 2 (DL Pipeline): After Section I Introduction (page 1)
  - Diagram 6 (Grad-CAM samples): After Section I Introduction (page 1, after pipeline)
  - Diagram 1 (Architecture Taxonomy): After TABLE IV in Section VI (page 5)
  - Diagram 3 (Cross-Domain Comparison): After Section VI-D (page 5)
  - Diagram 4 (Publication Timeline): After Section VI (page 5)
  - Diagram 5 (Research Gaps): After Section VII (page 6)
"""

import fitz  # PyMuPDF
from PIL import Image
import os
import io

PROJ_DIR = os.path.dirname(os.path.abspath(__file__))
DIAGRAMS_DIR = os.path.join(PROJ_DIR, "diagrams")
RESULTS_DIR = os.path.join(PROJ_DIR, "results")
INPUT_PDF = os.path.join(PROJ_DIR, "literature_review.pdf")
OUTPUT_PDF = os.path.join(PROJ_DIR, "literature_review_with_diagrams.pdf")

# IEEE two-column format page dimensions (Letter: 8.5 x 11 inches = 612 x 792 points)
PAGE_WIDTH = 612
PAGE_HEIGHT = 792
MARGIN = 54  # ~0.75 inch margins


def create_diagram_page(doc, image_path, caption, fig_number):
    """Create a new page with a diagram image and caption, formatted for IEEE style."""
    page = doc.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
    
    # Calculate usable area
    usable_width = PAGE_WIDTH - 2 * MARGIN
    usable_height = PAGE_HEIGHT - 2 * MARGIN
    
    # Reserve space for caption at bottom
    caption_height = 60
    img_area_height = usable_height - caption_height - 20  # 20pt gap
    
    # Open image to get aspect ratio
    img = Image.open(image_path)
    img_w, img_h = img.size
    aspect = img_w / img_h
    
    # Fit image within available area
    if aspect > (usable_width / img_area_height):
        # Width-constrained
        display_w = usable_width
        display_h = display_w / aspect
    else:
        # Height-constrained
        display_h = img_area_height
        display_w = display_h * aspect
    
    # Center horizontally
    x_offset = MARGIN + (usable_width - display_w) / 2
    y_offset = MARGIN + (img_area_height - display_h) / 2
    
    # Insert image
    img_rect = fitz.Rect(x_offset, y_offset, x_offset + display_w, y_offset + display_h)
    page.insert_image(img_rect, filename=image_path)
    
    # Add caption below image
    caption_y = MARGIN + img_area_height + 15
    caption_rect = fitz.Rect(MARGIN, caption_y, PAGE_WIDTH - MARGIN, caption_y + caption_height)
    
    caption_text = f"Fig. {fig_number}. {caption}"
    page.insert_textbox(
        caption_rect,
        caption_text,
        fontsize=9,
        fontname="times-roman",
        align=fitz.TEXT_ALIGN_CENTER,
    )
    
    return page


def create_gradcam_composite_page(doc, fig_number):
    """Create a page with 3 Grad-CAM images side by side."""
    page = doc.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
    
    gradcam_images = [
        (os.path.join(RESULTS_DIR, "GastroNet_gradcam_Ulcer_sample1.png"), "(a) Gastric Ulcer"),
        (os.path.join(RESULTS_DIR, "PneumoNet_gradcam_PNEUMONIA_sample1.png"), "(b) Pneumonia"),
        (os.path.join(RESULTS_DIR, "TumorNet_gradcam_glioma_sample1.png"), "(c) Brain Tumor"),
    ]
    
    usable_width = PAGE_WIDTH - 2 * MARGIN
    col_width = (usable_width - 20) / 3  # 10pt gap between images
    img_top = MARGIN + 10
    img_height = 280  # Fixed height for consistency
    
    for i, (img_path, sublabel) in enumerate(gradcam_images):
        if not os.path.exists(img_path):
            print(f"  WARNING: {img_path} not found, skipping.")
            continue
        
        x = MARGIN + i * (col_width + 10)
        img_rect = fitz.Rect(x, img_top, x + col_width, img_top + img_height)
        page.insert_image(img_rect, filename=img_path)
        
        # Sub-label below each image
        label_rect = fitz.Rect(x, img_top + img_height + 2, x + col_width, img_top + img_height + 18)
        page.insert_textbox(
            label_rect,
            sublabel,
            fontsize=8,
            fontname="times-roman",
            align=fitz.TEXT_ALIGN_CENTER,
        )
    
    # Main caption
    caption_y = img_top + img_height + 25
    caption_rect = fitz.Rect(MARGIN, caption_y, PAGE_WIDTH - MARGIN, caption_y + 50)
    caption_text = (
        f"Fig. {fig_number}. Representative medical images with Grad-CAM attention "
        f"visualizations across the three study domains, demonstrating deep learning "
        f"model interpretability for gastric ulcer detection, pneumonia diagnosis, "
        f"and brain tumor classification."
    )
    page.insert_textbox(
        caption_rect,
        caption_text,
        fontsize=9,
        fontname="times-roman",
        align=fitz.TEXT_ALIGN_CENTER,
    )
    
    return page


def find_page_containing(doc, search_text):
    """Find the page index containing specific text."""
    for i, page in enumerate(doc):
        text = page.get_text()
        if search_text in text:
            return i
    return -1


def main():
    print(f"Opening: {INPUT_PDF}")
    doc = fitz.open(INPUT_PDF)
    original_pages = len(doc)
    print(f"Original PDF has {original_pages} pages.")
    
    # Find insertion points by searching for section markers
    intro_page = find_page_containing(doc, "I. INTRODUCTION")
    methodology_page = find_page_containing(doc, "II. METHODOLOGY")
    gastric_page = find_page_containing(doc, "III. GASTRIC")
    pneumonia_page = find_page_containing(doc, "IV. PNEUMONIA")
    brain_page = find_page_containing(doc, "V. BRAIN TUMOR")
    crossdomain_page = find_page_containing(doc, "VI. CROSS-DOMAIN")
    table4_page = find_page_containing(doc, "TABLE IV")
    gaps_page = find_page_containing(doc, "VII. RESEARCH GAPS")
    conclusion_page = find_page_containing(doc, "VIII. CONCLUSION")
    
    print(f"  Section I (Introduction): page {intro_page + 1}")
    print(f"  Section II (Methodology): page {methodology_page + 1}")
    print(f"  Section III (Gastric): page {gastric_page + 1}")
    print(f"  Section IV (Pneumonia): page {pneumonia_page + 1}")
    print(f"  Section V (Brain Tumor): page {brain_page + 1}")
    print(f"  Section VI (Cross-Domain): page {crossdomain_page + 1}")
    print(f"  TABLE IV: page {table4_page + 1}")
    print(f"  Section VII (Research Gaps): page {gaps_page + 1}")
    print(f"  Section VIII (Conclusion): page {conclusion_page + 1}")
    
    # We'll build a new document by interleaving original pages with diagram pages
    output_doc = fitz.open()
    
    # Define insertions: (after_original_page_index, diagram_creation_function)
    # We insert after the page where the relevant section ENDS (next section starts)
    
    # Diagram 2 (Pipeline) → after Introduction, before Methodology
    # Diagram 6 (Grad-CAM) → after Pipeline
    # Diagram 1 (Taxonomy) → after TABLE IV page 
    # Diagram 3 (Comparison) → after Cross-Domain section
    # Diagram 4 (Timeline) → after Comparison
    # Diagram 5 (Gaps) → after Research Gaps, before Conclusion
    
    fig_counter = 1
    
    # Build insertion map: {after_page_index: [(image_path, caption, is_composite)]}
    insertions = {}
    
    # After Introduction (page before methodology)
    insert_after_intro = methodology_page - 1 if methodology_page > 0 else intro_page
    insertions.setdefault(insert_after_intro, [])
    insertions[insert_after_intro].append({
        "type": "single",
        "image": os.path.join(DIAGRAMS_DIR, "diagram2_pipeline.jpg"),
        "caption": "Generalized deep learning pipeline for medical image classification across the three study domains. The shared pipeline illustrates that despite different imaging modalities (endoscopy, chest X-ray, MRI), all three domains converge on similar feature extraction backbones, task heads, and evaluation frameworks."
    })
    insertions[insert_after_intro].append({
        "type": "composite_gradcam",
    })
    
    # After TABLE IV / Cross-Domain section
    insert_after_crossdomain = gaps_page - 1 if gaps_page > crossdomain_page else crossdomain_page
    # If TABLE IV and cross-domain are on same page, insert after that page
    if table4_page >= 0:
        insert_after_crossdomain = max(table4_page, insert_after_crossdomain)
    
    insertions.setdefault(insert_after_crossdomain, [])
    insertions[insert_after_crossdomain].append({
        "type": "single",
        "image": os.path.join(DIAGRAMS_DIR, "diagram1_taxonomy.jpg"),
        "caption": "Taxonomy of deep learning architectures identified across the three medical imaging domains. CNN classifiers (ResNet, VGG, DenseNet, EfficientNet, Inception) are used universally, U-Net variants cluster in brain tumor segmentation, YOLO-based detection spans gastric and brain tumor domains, and Vision Transformers have so far appeared primarily in pneumonia detection."
    })
    insertions[insert_after_crossdomain].append({
        "type": "single",
        "image": os.path.join(DIAGRAMS_DIR, "diagram3_comparison.jpg"),
        "caption": "Cross-domain comparison of deep learning approaches across gastric ulcer detection, pneumonia diagnosis, and brain tumor MRI classification. Despite different clinical contexts and imaging modalities, the three domains share fundamental DL foundations while differing in maturity, dominant architectures, and unresolved challenges."
    })
    insertions[insert_after_crossdomain].append({
        "type": "single",
        "image": os.path.join(DIAGRAMS_DIR, "diagram4_timeline.jpg"),
        "caption": "Publication timeline of key reviewed studies by domain (2018–2026). The timeline illustrates the acceleration of research post-2020, particularly in pneumonia detection driven by the COVID-19 pandemic, the steady growth in brain tumor MRI research, and the more recent emergence of sophisticated gastric ulcer AI systems."
    })
    
    # After Research Gaps section
    insert_after_gaps = conclusion_page - 1 if conclusion_page > gaps_page else gaps_page
    insertions.setdefault(insert_after_gaps, [])
    insertions[insert_after_gaps].append({
        "type": "single",
        "image": os.path.join(DIAGRAMS_DIR, "diagram5_gaps.jpg"),
        "caption": "Identified research gaps organized by domain-specific challenges and cross-domain systemic barriers, with recommended future directions. While each domain has unique technical gaps, the overarching barriers to clinical translation—limited validation, dataset homogeneity, and absence of prospective trials—are universal."
    })
    
    # Now build the output document
    print("\nBuilding output PDF with diagrams...")
    
    for page_idx in range(original_pages):
        # Copy original page
        output_doc.insert_pdf(doc, from_page=page_idx, to_page=page_idx)
        
        # Check if we need to insert diagrams after this page
        if page_idx in insertions:
            for item in insertions[page_idx]:
                if item["type"] == "composite_gradcam":
                    print(f"  Inserting Fig. {fig_counter}: Grad-CAM composite (after page {page_idx + 1})")
                    create_gradcam_composite_page(output_doc, fig_counter)
                    fig_counter += 1
                elif item["type"] == "single":
                    img_path = item["image"]
                    if os.path.exists(img_path):
                        img_name = os.path.basename(img_path)
                        print(f"  Inserting Fig. {fig_counter}: {img_name} (after page {page_idx + 1})")
                        create_diagram_page(output_doc, img_path, item["caption"], fig_counter)
                        fig_counter += 1
                    else:
                        print(f"  WARNING: {img_path} not found, skipping.")
    
    # Save output
    output_doc.save(OUTPUT_PDF)
    output_doc.close()
    doc.close()
    
    final_pages = fitz.open(OUTPUT_PDF).page_count
    print(f"\nDone! Output saved to: {OUTPUT_PDF}")
    print(f"Original: {original_pages} pages -> With diagrams: {final_pages} pages")
    print(f"Total figures inserted: {fig_counter - 1}")


if __name__ == "__main__":
    main()
