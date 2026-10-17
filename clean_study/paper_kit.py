
"""Helper kit for building an IEEE-conference-style two-column DOCX with python-docx."""
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT
from docx.enum.section import WD_SECTION, WD_ORIENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import copy

BODY_FONT = "Times New Roman"
MATH_FONT = "Cambria Math"

def _el(tag, **attrs):
    e = OxmlElement(tag)
    for k, v in attrs.items():
        e.set(qn("w:" + k), str(v))
    return e

def setup_page(doc):
    s = doc.sections[0]
    s.page_width, s.page_height = Inches(8.5), Inches(11)
    s.top_margin, s.bottom_margin = Inches(0.75), Inches(1.0)
    s.left_margin, s.right_margin = Inches(0.625), Inches(0.625)
    s.header_distance, s.footer_distance = Inches(0.5), Inches(0.5)
    set_cols(s, 1)
    return s

def set_cols(section, n, space_in=0.25):
    sectPr = section._sectPr
    for c in sectPr.findall(qn("w:cols")):
        sectPr.remove(c)
    cols = _el("w:cols", num=n, space=int(space_in * 1440), equalWidth=1)
    sectPr.append(cols)

def new_section(doc, n):
    """Continuous section break; inherits page geometry, sets column count."""
    sec = doc.add_section(WD_SECTION.CONTINUOUS)
    prev = doc.sections[-2]
    sec.page_width, sec.page_height = prev.page_width, prev.page_height
    sec.top_margin, sec.bottom_margin = prev.top_margin, prev.bottom_margin
    sec.left_margin, sec.right_margin = prev.left_margin, prev.right_margin
    set_cols(sec, n)
    return sec

# ------------------------------------------------------------------ paragraphs
def _fmt(p, indent=0.2, before=0, after=0, align="justify", spacing=1.0):
    pf = p.paragraph_format
    pf.first_line_indent = Inches(indent)
    pf.space_before, pf.space_after = Pt(before), Pt(after)
    pf.line_spacing = spacing
    p.alignment = {"justify": WD_ALIGN_PARAGRAPH.JUSTIFY, "center": WD_ALIGN_PARAGRAPH.CENTER,
                   "left": WD_ALIGN_PARAGRAPH.LEFT, "right": WD_ALIGN_PARAGRAPH.RIGHT}[align]

def rich(p, text, size=10, bold=False, italic=False, font=BODY_FONT, color=None):
    """Add text with **bold**, *italic*, _sub..., ^sup... mini-markup."""
    i, buf = 0, ""
    def flush(b, bo, it, sub, sup):
        if not b: return
        r = p.add_run(b)
        r.font.size, r.font.name = Pt(size), font
        r.font.bold, r.font.italic = bo, it
        r.font.subscript, r.font.superscript = sub, sup
        if color: r.font.color.rgb = color
    state = dict(bold=bold, italic=italic, sub=False, sup=False); buf = ""
    while i < len(text):
        c = text[i]
        if c == "*" and not state["sub"] and not state["sup"]:
            flush(buf, state["bold"], state["italic"], False, False); buf = ""
            if text[i:i+2] == "**":
                state["bold"] = not state["bold"]; i += 2; continue
            state["italic"] = not state["italic"]; i += 1; continue
        if c in "_^" and i + 1 < len(text) and text[i+1] == "{":
            flush(buf, state["bold"], state["italic"], False, False); buf = ""
            j = text.index("}", i)
            r = p.add_run(text[i+2:j])
            r.font.size, r.font.name = Pt(size), font
            r.font.bold, r.font.italic = state["bold"], state["italic"]
            r.font.subscript = (c == "_"); r.font.superscript = (c == "^")
            if color: r.font.color.rgb = color
            i = j + 1; continue
        buf += c; i += 1
    flush(buf, state["bold"], state["italic"], False, False)
    return p

def body(doc, text, first=False, size=10):
    p = doc.add_paragraph()
    _fmt(p, indent=0.0 if first else 0.2, after=0)
    rich(p, text, size=size)
    return p

def bullet(doc, text, size=9.5):
    p = doc.add_paragraph()
    _fmt(p, indent=0.2, after=2, align="justify")
    p.paragraph_format.left_indent = Inches(0.22)
    p.paragraph_format.first_line_indent = Inches(-0.14)
    rich(p, "\u2022  " + text, size=size)
    return p

def h1(doc, text, num=None, before=10, after=4):
    p = doc.add_paragraph()
    _fmt(p, indent=0, before=before, after=after, align="center")
    p.paragraph_format.keep_with_next = True
    label = (num + ". " if num else "")
    r = p.add_run(label + text.upper()); r.font.size, r.font.name = Pt(10), BODY_FONT
    r.font.bold = False; r.font.small_caps = True
    return p

def h2(doc, text, before=7, after=3):
    p = doc.add_paragraph()
    _fmt(p, indent=0.2, before=before, after=after, align="left")
    p.paragraph_format.keep_with_next = True
    rich(p, text, size=10, italic=True)
    return p

def h3(doc, text, before=5, after=2):
    p = doc.add_paragraph()
    _fmt(p, indent=0.2, before=before, after=after, align="left")
    p.paragraph_format.keep_with_next = True
    rich(p, text, size=10, italic=True)
    return p

# ------------------------------------------------------------------ math
def equation(doc, spec, n, size=10):
    p = doc.add_paragraph()
    _fmt(p, indent=0, before=6, after=6, align="left")
    p.paragraph_format.tab_stops.add_tab_stop(Inches(3.5), WD_TAB_ALIGNMENT.RIGHT)
    p.paragraph_format.left_indent = Inches(0.3)
    rich(p, "\t" + spec + "\t(" + str(n) + ")", size=size, italic=True, font=MATH_FONT)
    for r in p.runs:
        r.font.italic = r.font.italic
    return p

# ------------------------------------------------------------------ figures
def figure(doc, path, caption, width=3.45, num=None):
    p = doc.add_paragraph()
    _fmt(p, indent=0, before=8, after=2, align="center")
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(path, width=Inches(width))
    c = doc.add_paragraph()
    _fmt(c, indent=0, before=0, after=8, align="center")
    lab = ("Fig. %s. " % num) if num else ""
    rich(c, lab + caption, size=8)
    return c

# ------------------------------------------------------------------ tables
def _cell_text(cell, text, size=7.5, bold=False, align="left"):
    cell.text = ""
    p = cell.paragraphs[0]
    _fmt(p, indent=0, before=1.5, after=1.5, align=align, spacing=1.0)
    rich(p, text, size=size, bold=bold)

def _shade(cell, hexcolor):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = _el("w:shd", val="clear", color="auto", fill=hexcolor)
    tcPr.append(shd)

def _borders(table, sz=4):
    tbl = table._tbl
    tblPr = tbl.tblPr
    for old in tblPr.findall(qn("w:tblBorders")):
        tblPr.remove(old)
    b = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        e = OxmlElement("w:" + edge)
        e.set(qn("w:val"), "single" if edge in ("top", "bottom") else "none")
        e.set(qn("w:sz"), str(sz)); e.set(qn("w:color"), "000000")
        b.append(e)
    tblPr.append(b)

def _no_borders(table):
    tblPr = table._tbl.tblPr
    for old in tblPr.findall(qn("w:tblBorders")):
        tblPr.remove(old)
    b = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        e = OxmlElement("w:" + edge); e.set(qn("w:val"), "none"); e.set(qn("w:sz"), "0")
        b.append(e)
    tblPr.append(b)

def table(doc, num, title, header, rows, widths, size=7.2, note=None, borders=True, align=None, width_in=3.42):
    cap = doc.add_paragraph()
    _fmt(cap, indent=0, before=8, after=2, align="center")
    cap.paragraph_format.keep_with_next = True
    r = cap.add_run("TABLE " + num); r.font.size, r.font.name = Pt(7.5), BODY_FONT; r.font.small_caps = True
    r2 = cap.add_run("\n" + title); r2.font.size, r2.font.name = Pt(7.5), BODY_FONT
    t = doc.add_table(rows=0, cols=len(header))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    tblPr = t._tbl.tblPr
    lay = OxmlElement("w:tblLayout"); lay.set(qn("w:type"), "fixed"); tblPr.append(lay)
    if borders: _borders(t)
    else: _no_borders(t)
    total = sum(widths)
    row = t.add_row()
    for i, h in enumerate(header):
        _cell_text(row.cells[i], h, size=size, bold=False, align="center")
    for r_ in rows:
        row = t.add_row()
        for i, v in enumerate(r_):
            a = (align[i] if align else ("left" if i == 0 else "center"))
            _cell_text(row.cells[i], str(v), size=size, align=a)
    for row in t.rows:
        for i, c in enumerate(row.cells):
            c.width = Inches(widths[i] * width_in / total)
    # rule under header
    return t

def note(doc, text, size=7):
    p = doc.add_paragraph()
    _fmt(p, indent=0, before=3, after=8, align="justify")
    rich(p, text, size=size)
    return p

def spacer(doc, pts=4):
    p = doc.add_paragraph(); _fmt(p, indent=0, before=0, after=0)
    p.add_run().font.size = Pt(pts)
    return p
