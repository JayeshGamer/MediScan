import docx

doc = docx.Document('Claude_MediScan_revised.docx')
ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
w_fill = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}fill'

with open('yellow_paragraphs_detail.txt', 'w', encoding='utf-8') as out:
    for p_idx in [4, 8, 37, 41, 43, 47, 56, 67, 69, 71, 78, 79, 80, 82, 84, 86, 95, 109, 117, 128, 132, 143, 144, 145, 211]:
        p = doc.paragraphs[p_idx]
        out.write(f'=== P{p_idx} (Runs: {len(p.runs)}) ===\n')
        for r_idx, r in enumerate(p.runs):
            rPr = r._r.find('w:rPr', ns)
            is_yellow = False
            if rPr is not None:
                shd = rPr.find('w:shd', ns)
                if shd is not None and shd.get(w_fill) == 'FFFF00':
                    is_yellow = True
            out.write(f'  R{r_idx} [Yellow={is_yellow}]: {repr(r.text)}\n')
        out.write('\n')

print("Wrote yellow_paragraphs_detail.txt successfully.")
