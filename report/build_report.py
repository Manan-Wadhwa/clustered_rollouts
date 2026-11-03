"""Build the minor project report (DOCX, then PDF through LibreOffice).

    python3 report/make_figs.py
    python3 report/build_report.py

Two passes: the first PDF is used to read off page numbers for the contents,
list of tables and list of figures; the second build writes them in.
Formatting follows the MAIT CST minor project template: A4, margins L 1.25in,
R/T/B 1.0in, Times New Roman, title 18, heading 16, sub heading 14, text 12,
1.5 line spacing, centred page numbers, no page border.
"""
import csv
import os
import re
import subprocess
import sys
from collections import defaultdict

import numpy as np
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING, WD_TAB_ALIGNMENT, WD_TAB_LEADER
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "analysis"))
from analyze_real import analyse  # noqa: E402

REP = os.path.join(ROOT, "report")
FIG = os.path.join(REP, "figs")
OUT_DOCX = os.path.join(REP, "Minor_Project_Report_Clustered_Rollouts.docx")
OUT_PDF = OUT_DOCX.replace(".docx", ".pdf")
FONT = "Times New Roman"
TEXT_W = 8.27 - 1.25 - 1.0

TITLE = ("Zero Harmful Actions in N Rollouts Is Not N Trials: Independence-Based "
         "Safety Bounds Are Too Narrow for Clustered Rollouts")
STUDENTS = [("Manan Wadhwa", "__________"), ("Anagh Miglani", "__________"),
            ("Hardik Dhull", "__________")]
GUIDE = ("Prof. (Dr.) Pooja Gupta", "Professor & HOD, CST")
COGUIDE = ("Dr. Garima Mittal", "Department of CST")


# ------------------------------------------------------------------ helpers
def set_cell_shading(cell, hexcolor):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hexcolor)
    tcPr.append(shd)


def add_field(run, instr):
    for kind, text in (("begin", None), ("instr", instr), ("separate", None), ("text", "1"), ("end", None)):
        if kind == "instr":
            el = OxmlElement("w:instrText")
            el.set(qn("xml:space"), "preserve")
            el.text = text
        elif kind == "text":
            el = OxmlElement("w:t")
            el.text = text
        else:
            el = OxmlElement("w:fldChar")
            el.set(qn("w:fldCharType"), kind)
        run._r.append(el)


def set_pgnum(section, fmt, start):
    sectPr = section._sectPr
    for old in sectPr.findall(qn("w:pgNumType")):
        sectPr.remove(old)
    pg = OxmlElement("w:pgNumType")
    pg.set(qn("w:fmt"), fmt)
    pg.set(qn("w:start"), str(start))
    sectPr.append(pg)


def footer_pagenum(section, show=True):
    # a new section clones the previous sectPr, footer reference included; drop it
    # first so this section gets its own footer part
    section.footer.is_linked_to_previous = True
    section.footer.is_linked_to_previous = False
    p = section.footer.paragraphs[0]
    for r in list(p.runs):
        r._r.getparent().remove(r._r)
    if show:
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run()
        r.font.name, r.font.size = FONT, Pt(12)
        add_field(r, "PAGE")


class Report:
    def __init__(self, pages=None):
        self.doc = Document()
        self.pages = pages or {}
        self.toc, self.lot, self.lof = [], [], []
        self.chap = 0
        self.sec = 0
        self.sub = 0
        self.tab_n = defaultdict(int)
        self.fig_n = defaultdict(int)
        self.code_n = defaultdict(int)
        self._styles()
        s = self.doc.sections[0]
        s.page_width, s.page_height = Inches(8.27), Inches(11.69)
        s.left_margin, s.right_margin = Inches(1.25), Inches(1.0)
        s.top_margin, s.bottom_margin = Inches(1.0), Inches(1.0)
        footer_pagenum(s, show=False)

    def _styles(self):
        st = self.doc.styles
        n = st["Normal"]
        n.font.name, n.font.size = FONT, Pt(12)
        n.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
        pf = n.paragraph_format
        pf.line_spacing = 1.5
        pf.space_after = Pt(6)
        for name, size in (("Heading 1", 16), ("Heading 2", 14), ("Heading 3", 12), ("Title", 18)):
            h = st[name]
            h.font.name, h.font.size, h.font.bold = FONT, Pt(size), True
            h.font.color.rgb = RGBColor(0, 0, 0)
            h.font.italic = False
            rpr = h.element.get_or_add_rPr()
            rf = rpr.find(qn("w:rFonts"))
            if rf is None:
                rf = OxmlElement("w:rFonts")
                rpr.append(rf)
            for a in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
                rf.set(qn(a), FONT)
            for a in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
                if rf.get(qn(a)) is not None:
                    del rf.attrib[qn(a)]
            h.paragraph_format.space_before = Pt(12)
            h.paragraph_format.space_after = Pt(6)
            h.paragraph_format.line_spacing = 1.5
            h.paragraph_format.keep_with_next = True

    # basic blocks ----------------------------------------------------------
    def para(self, text="", align=WD_ALIGN_PARAGRAPH.JUSTIFY, size=12, bold=False, italic=False,
             space_after=6, spacing=1.5, indent_first=False):
        p = self.doc.add_paragraph()
        p.alignment = align
        p.paragraph_format.space_after = Pt(space_after)
        p.paragraph_format.line_spacing = spacing
        if indent_first:
            p.paragraph_format.first_line_indent = Inches(0.4)
        self._runs(p, text, size, bold, italic)
        return p

    def _runs(self, p, text, size=12, bold=False, italic=False):
        # **bold** and _italic_ inline markup
        parts = re.split(r"(\*\*[^*]+\*\*)", text)
        for part in parts:
            if not part:
                continue
            b, i = bold, italic
            if part.startswith("**") and part.endswith("**"):
                part, b = part[2:-2], True
            r = p.add_run(part)
            r.font.name, r.font.size, r.bold, r.italic = FONT, Pt(size), b, i

    def bullets(self, items, numbered=False):
        for k, it in enumerate(items, 1):
            p = self.doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            p.paragraph_format.left_indent = Inches(0.45)
            p.paragraph_format.first_line_indent = Inches(-0.25)
            p.paragraph_format.space_after = Pt(3)
            self._runs(p, (f"{k}.\t" if numbered else "•\t") + it)
            p.paragraph_format.tab_stops.add_tab_stop(Inches(0.45))

    def equation(self, text, num=None):
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(6)
        r = p.add_run(text)
        r.font.name, r.font.size, r.italic = "Cambria Math", Pt(12), False
        if num:
            r2 = p.add_run(f"\t\t({num})")
            r2.font.name, r2.font.size = FONT, Pt(12)

    def page_break(self):
        self.doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

    def new_section(self, fmt=None, start=None, show_num=True):
        s = self.doc.add_section(WD_SECTION.NEW_PAGE)
        footer_pagenum(s, show_num)
        if fmt:
            set_pgnum(s, fmt, start)
        return s

    # headings ----------------------------------------------------------------
    def front_heading(self, text, toc=True):
        p = self.doc.add_paragraph(style="Heading 1")
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(text.upper())
        r.font.size = Pt(16)
        if toc:
            self.toc.append((0, text.upper(), f"FM::{text.upper()}"))

    def chapter(self, title):
        self.chap += 1
        self.sec = 0
        self.page_break() if self.chap > 1 else None
        p = self.doc.add_paragraph(style="Heading 1")
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(0)
        r = p.add_run(f"CHAPTER {self.chap}")
        r.font.size = Pt(16)
        p = self.doc.add_paragraph(style="Heading 1")
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(18)
        r = p.add_run(title.upper())
        r.font.size = Pt(16)
        self.toc.append((0, f"Chapter {self.chap}: {title}", f"CHAPTER {self.chap}"))

    def unnumbered_chapter(self, title):
        self.page_break()
        p = self.doc.add_paragraph(style="Heading 1")
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(18)
        p.add_run(title.upper()).font.size = Pt(16)
        self.toc.append((0, title, title.upper()))

    def section(self, title):
        self.sec += 1
        self.sub = 0
        t = f"{self.chap}.{self.sec} {title}"
        self.doc.add_paragraph(t, style="Heading 2")
        self.toc.append((1, t, t))

    def subsection(self, title, toc=True):
        self.sub += 1
        t = f"{self.chap}.{self.sec}.{self.sub} {title}"
        self.doc.add_paragraph(t, style="Heading 3")
        if toc:
            self.toc.append((2, t, t))

    def appendix(self, letter, title):
        self.page_break()
        p = self.doc.add_paragraph(style="Heading 1")
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(18)
        t = f"APPENDIX {letter}: {title.upper()}"
        p.add_run(t).font.size = Pt(16)
        self.toc.append((0, f"Appendix {letter}: {title}", t))
        self.chap_label = letter

    # floats ------------------------------------------------------------------
    def _caption(self, label, text, above):
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(4 if above else 2)
        p.paragraph_format.space_after = Pt(4 if above else 12)
        p.paragraph_format.line_spacing = 1.15
        if above:
            p.paragraph_format.keep_with_next = True
        r = p.add_run(f"{label}: ")
        r.font.name, r.font.size, r.bold = FONT, Pt(11), True
        r = p.add_run(text)
        r.font.name, r.font.size = FONT, Pt(11)

    def _num(self, counter):
        key = getattr(self, "chap_label", None) or self.chap
        counter[key] += 1
        return f"{key}.{counter[key]}"

    def figure(self, name, caption, width=6.0):
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.keep_with_next = True
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.line_spacing = 1.0
        p.add_run().add_picture(os.path.join(FIG, name), width=Inches(width))
        n = self._num(self.fig_n)
        self._caption(f"Figure {n}", caption, above=False)
        self.lof.append((f"Figure {n}", caption.split(". ")[0].rstrip("."), f"Figure {n}:"))
        return f"Figure {n}"

    def table(self, header, rows, caption, widths=None, size=10, align=None):
        n = None
        if caption:
            n = self._num(self.tab_n)
            self._caption(f"Table {n}", caption, above=True)
            self.lot.append((f"Table {n}", caption.split(". ")[0].rstrip("."), f"Table {n}:"))
        t = self.doc.add_table(rows=1 + len(rows), cols=len(header))
        t.style = "Table Grid"
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        for j, h in enumerate(header):
            c = t.rows[0].cells[j]
            set_cell_shading(c, "D9E2F3")
            self._cell(c, h, size, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
        for i, row in enumerate(rows, 1):
            for j, v in enumerate(row):
                a = WD_ALIGN_PARAGRAPH.LEFT
                if align:
                    a = {"l": WD_ALIGN_PARAGRAPH.LEFT, "c": WD_ALIGN_PARAGRAPH.CENTER,
                         "r": WD_ALIGN_PARAGRAPH.RIGHT}[align[j]]
                self._cell(t.rows[i].cells[j], str(v), size, align=a)
        for row in t.rows:
            cs = OxmlElement("w:cantSplit"); cs.set(qn("w:val"), "true")
            row._tr.get_or_add_trPr().append(cs)
        # repeat header row on page breaks
        trPr = t.rows[0]._tr.get_or_add_trPr()
        th = OxmlElement("w:tblHeader")
        th.set(qn("w:val"), "true")
        trPr.append(th)
        self._widths(t, widths or [TEXT_W / len(header)] * len(header))
        sp = self.doc.add_paragraph()
        sp.paragraph_format.space_after = Pt(6)
        sp.paragraph_format.line_spacing = 1.0
        return f"Table {n}"

    def _widths(self, t, widths):
        t.autofit = False
        tot = sum(widths)
        widths = [w * min(1.0, TEXT_W / tot) for w in widths]
        tblPr = t._tbl.tblPr
        tw = tblPr.find(qn("w:tblW"))
        if tw is None:
            tw = OxmlElement("w:tblW"); tblPr.append(tw)
        tw.set(qn("w:type"), "dxa"); tw.set(qn("w:w"), str(int(sum(widths) * 1440)))
        lay = OxmlElement("w:tblLayout"); lay.set(qn("w:type"), "fixed"); tblPr.append(lay)
        grid = t._tbl.tblGrid
        for j, gc in enumerate(grid.findall(qn("w:gridCol"))):
            gc.set(qn("w:w"), str(int(widths[j] * 1440)))
        for row in t.rows:
            for j, w in enumerate(widths):
                row.cells[j].width = Inches(w)

    def _cell(self, cell, text, size, bold=False, align=WD_ALIGN_PARAGRAPH.LEFT):
        p = cell.paragraphs[0]
        p.alignment = align
        p.paragraph_format.line_spacing = 1.0
        p.paragraph_format.space_after = Pt(1)
        p.paragraph_format.space_before = Pt(1)
        self._runs(p, text, size, bold)

    def code(self, text, caption):
        n = self._num(self.code_n)
        self._caption(f"Code {n}", caption, above=True)
        t = self.doc.add_table(rows=1, cols=1)
        t.style = "Table Grid"
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        self._widths(t, [TEXT_W])
        c = t.rows[0].cells[0]
        set_cell_shading(c, "F3F3F3")
        cs = OxmlElement("w:cantSplit"); cs.set(qn("w:val"), "true")
        t.rows[0]._tr.get_or_add_trPr().append(cs)
        lines = text.strip("\n").split("\n")
        p = c.paragraphs[0]
        for k, line in enumerate(lines):
            if k:
                p = c.add_paragraph()
            p.paragraph_format.line_spacing = 1.0
            p.paragraph_format.space_after = Pt(0)
            r = p.add_run(line if line else " ")
            r.font.name, r.font.size = "Courier New", Pt(8.5)
            r._element.rPr.rFonts.set(qn("w:eastAsia"), "Courier New")
        sp = self.doc.add_paragraph()
        sp.paragraph_format.space_after = Pt(6)
        sp.paragraph_format.line_spacing = 1.0
        return f"Code {n}"

    # contents pages ----------------------------------------------------------
    def list_page(self, title, entries, kind):
        self.front_heading(title, toc=False)
        for e in entries:
            if kind == "toc":
                level, text, key = e
            else:
                label, text, key = e
                text = f"{label}: {text}"
                level = 0
            pg = self.pages.get(key, "")
            p = self.doc.add_paragraph()
            p.paragraph_format.line_spacing = 1.15
            p.paragraph_format.space_after = Pt(3 if level else 4)
            p.paragraph_format.left_indent = Inches(0.3 * level)
            p.paragraph_format.tab_stops.add_tab_stop(Inches(TEXT_W), WD_TAB_ALIGNMENT.RIGHT, WD_TAB_LEADER.DOTS)
            r = p.add_run(f"{text}\t{pg}")
            r.font.name, r.font.size = FONT, Pt(12 if level == 0 else 11.5)
            r.bold = kind == "toc" and level == 0

    def save(self, path):
        self.doc.save(path)


# ------------------------------------------------------------------ data
def load_cells(path):
    data = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    with open(path) as fh:
        for r in csv.DictReader(fh):
            c = data[r["model"]][r["cluster"]]
            c[0] += int(r["harmful"])
            c[1] += 1
    return data


PRETTY = {"qwen3_8b": "Qwen3-8B", "qwen3_14b": "Qwen3-14B", "qwen3_32b": "Qwen3-32B",
          "mistral_small_24b": "Mistral-Small-24B", "gpt_oss_20b": "gpt-oss-20b",
          "qwen3_8b_think": "Qwen3-8B (thinking)"}


def real_rows():
    data = load_cells(os.path.join(ROOT, "data", "rollouts_opus.csv"))
    rows = []
    for model in PRETTY:
        for scope, prefix in (("all (30)", ""), ("blackmail (15)", "blackmail_"), ("leaking (15)", "leaking_")):
            names = sorted(c for c in data[model] if c.startswith(prefix))
            k = np.array([data[model][c][0] for c in names], float)
            m = np.array([data[model][c][1] for c in names], float)
            o = analyse(k, m)
            icc = "-" if o["IccAnova"] != o["IccAnova"] else \
                f"{max(o['IccAnova'], 0):.2f} [{max(o['IccAnovaLo'], 0):.2f}, {max(o['IccAnovaHi'], 0):.2f}]"
            bb = "-" if o["IccBB"] != o["IccBB"] else f"{o['IccBB']:.2f}"
            deff = "-" if o["Deff"] != o["Deff"] else f"{o['Deff']:.1f}"
            neff = "-" if o["Neff"] != o["Neff"] else f"{o['Neff']:.0f}"
            rows.append([PRETTY[model] if prefix == "" else "", scope, f"{o['Events']} / {o['Rollouts']}",
                         str(o["ClustersWithEvents"]), icc, bb, deff, neff,
                         f"{o['NaiveLo']:.1f} to {o['NaiveHi']:.1f}", f"{o['BootLo']:.1f} to {o['BootHi']:.1f}"])
    return rows, data


# ------------------------------------------------------------------ content
def build(pages=None):
    R = Report(pages)
    d = R.doc
    real, data = real_rows()

    # ---------------- title page
    for _ in range(2):
        R.para("", space_after=0)
    R.para(TITLE, WD_ALIGN_PARAGRAPH.CENTER, 18, bold=True, spacing=1.3, space_after=18)
    R.para("A", WD_ALIGN_PARAGRAPH.CENTER, 14, space_after=0)
    R.para("MINOR PROJECT REPORT", WD_ALIGN_PARAGRAPH.CENTER, 14, bold=True, space_after=14)
    R.para("Submitted by", WD_ALIGN_PARAGRAPH.CENTER, 12, italic=True, space_after=4)
    for name, roll in STUDENTS:
        R.para(f"{name.upper()}  (Roll No. {roll})", WD_ALIGN_PARAGRAPH.CENTER, 13, bold=True, space_after=0)
    R.para("", space_after=6)
    R.para("BACHELOR OF TECHNOLOGY", WD_ALIGN_PARAGRAPH.CENTER, 14, bold=True, space_after=0)
    R.para("IN", WD_ALIGN_PARAGRAPH.CENTER, 12, space_after=0)
    R.para("COMPUTER SCIENCE AND TECHNOLOGY", WD_ALIGN_PARAGRAPH.CENTER, 14, bold=True, space_after=14)
    R.para("Under the Guidance", WD_ALIGN_PARAGRAPH.CENTER, 12, space_after=0)
    R.para("of", WD_ALIGN_PARAGRAPH.CENTER, 12, space_after=0)
    R.para(GUIDE[0], WD_ALIGN_PARAGRAPH.CENTER, 13, bold=True, space_after=0)
    R.para(f"({GUIDE[1]})", WD_ALIGN_PARAGRAPH.CENTER, 12, space_after=6)
    R.para(COGUIDE[0], WD_ALIGN_PARAGRAPH.CENTER, 13, bold=True, space_after=0)
    R.para(f"({COGUIDE[1]})", WD_ALIGN_PARAGRAPH.CENTER, 12, space_after=24)
    logo = os.path.join(FIG, "mait_logo.png")
    if os.path.exists(logo):
        p = d.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.add_run().add_picture(logo, width=Inches(1.1))
    for line, b in (("Department of Computer Science and Technology", True),
                    ("Maharaja Agrasen Institute of Technology,", True),
                    ("PSP area, Sector-22, Rohini, New Delhi-110085", True),
                    ("(Affiliated to Guru Gobind Singh Indraprastha University, New Delhi)", True)):
        R.para(line, WD_ALIGN_PARAGRAPH.CENTER, 12, bold=b, space_after=0, spacing=1.2)
    R.para("", space_after=4)
    R.para("DEC 2026", WD_ALIGN_PARAGRAPH.CENTER, 13, bold=True, space_after=0)

    # ---------------- front matter (roman numbers)
    R.new_section("lowerRoman", 1)
    R.para("MAHARAJA AGRASEN INSTITUTE OF TECHNOLOGY", WD_ALIGN_PARAGRAPH.CENTER, 16, bold=True, space_after=0)
    R.para("Department of Computer Science and Technology", WD_ALIGN_PARAGRAPH.CENTER, 13, space_after=24)
    R.front_heading("Certificate")
    R.para("", space_after=6)
    names = ", ".join(f"{n} (Roll No. {r})" for n, r in STUDENTS)
    R.para(f"This is to certify that this MINOR project report “{TITLE}” is submitted by "
           f"“{names}” who carried out the project work under my/our supervision.",
           spacing=2.0, space_after=12)
    R.para("I/We approve this MINOR project for submission.", spacing=2.0)
    for _ in range(4):
        R.para("", space_after=0)
    t = d.add_table(rows=3, cols=2)
    R._widths(t, [3.0, 3.0])
    vals = [(GUIDE[0], COGUIDE[0]), ("(Professor & HOD, CST)", f"({COGUIDE[1]})"),
            ("(Project Guide)", "(Project Co-Guide)")]
    for i, (a, b) in enumerate(vals):
        R._cell(t.rows[i].cells[0], a, 12, bold=i == 0)
        R._cell(t.rows[i].cells[1], b, 12, bold=i == 0, align=WD_ALIGN_PARAGRAPH.RIGHT)

    # Abstract
    R.page_break()
    R.front_heading("Abstract")
    R.para(
        "AI companies and safety researchers test language-model agents by placing them in short, staged "
        "situations and watching whether they do something harmful, for example blackmailing an executive or "
        "leaking confidential files. Each run is called a rollout. When no rollout out of N is harmful, reports "
        "usually apply the rule of three and say the harmful rate is below about 3/N. That rule assumes every "
        "rollout is an independent trial. In practice the rollouts come from a small number of scenarios, each "
        "run many times, and runs of the same scenario behave alike. This project asks how much that matters.",
        indent_first=False)
    R.para(
        "We treat each scenario as a cluster and measure the intra-cluster correlation (ICC). We implement five "
        "upper bounds for the harmful rate: the naive Clopper-Pearson bound, a design-effect bound, an exact "
        "beta-binomial bound, a worst-case bound that holds for any spread of scenario rates with a given ICC, "
        "and a scenario-level bound that needs no assumption at all. Exact coverage calculations show that the "
        "naive bound covers the true population rate as rarely as 14.4% of the time instead of 95%. We then ran "
        "six open-weight models (Qwen3 8B, 14B and 32B, Qwen3-8B with reasoning on, Mistral-Small-24B and "
        "gpt-oss-20b) on 30 configurations of an open agentic misalignment suite, 100 rollouts each, for 18,000 "
        "labelled rollouts. Within the leaking behaviour the ICC ranged from 0.03 to 0.27, so 1,500 rollouts were "
        "worth only 54 to 419 independent ones.")
    R.para(
        "Three model setups never blackmailed in 1,500 rollouts of 15 configurations. The rule of three puts "
        "their blackmail rate below 0.20%. Allowing for the measured agreement between rollouts the data only "
        "show it is below 1.02%, and no number of extra rollouts of the same scenarios would take it under 1%. "
        "With no assumption the bound is 18.1%. We also re-bound eight published zero-event claims; two of them "
        "rest on a single prompt. The main recommendation is simple: report the number of distinct scenarios "
        "next to the number of rollouts. The work has been written up as a paper and submitted to the AISTATS "
        "2027 Workshop Track.")
    R.para("**Keywords:** AI safety evaluation, agentic misalignment, intra-cluster correlation, design effect, "
           "zero-event bounds, beta-binomial model, Clopper-Pearson interval.", space_after=0)

    # Acknowledgement
    R.page_break()
    R.front_heading("Acknowledgement")
    R.para(
        f"It gives me immense pleasure to express my deepest sense of gratitude and sincere thanks to my "
        f"respected guide {GUIDE[0]} ({GUIDE[1]}) and co-guide {COGUIDE[0]} ({COGUIDE[1]}), MAIT Delhi, for "
        "their valuable guidance, encouragement and help for completing this work. Their useful suggestions "
        "for this whole work and co-operative behavior are sincerely acknowledged.")
    R.para(
        "I am also grateful to my teachers Prof. (Dr.) Himanshu Khanna and Ms. Prachi Dahiya (Project "
        "Coordinators) for their constant support and guidance.")
    R.para(
        "I also wish to express my indebtedness to my parents as well as my family member whose blessings "
        "and support always helped me to face the challenges ahead.")
    R.para("", space_after=24)
    t = d.add_table(rows=1 + len(STUDENTS), cols=2)
    R._widths(t, [2.5, 3.5])
    R._cell(t.rows[0].cells[0], "Place: Delhi", 12)
    R._cell(t.rows[1].cells[0], "Date:", 12)
    for i, (n, r) in enumerate(STUDENTS):
        R._cell(t.rows[i].cells[1], f"{n} (Roll No. {r})", 12, align=WD_ALIGN_PARAGRAPH.RIGHT)

    # placeholders for lists (filled on second pass)
    for name in ("TABLE OF CONTENTS", "LIST OF TABLES", "LIST OF FIGURES"):
        R.toc.append((0, name, f"FM::{name}"))
    R.page_break()
    R._toc_marker = d.add_paragraph()  # TOC inserted later at this point
    R.page_break()
    R._lot_marker = d.add_paragraph()
    R.page_break()
    R._lof_marker = d.add_paragraph()

    # Abbreviations
    R.page_break()
    R.front_heading("List of Symbols, Abbreviations and Nomenclature")
    abbrev = [
        ("AI", "Artificial Intelligence"), ("LLM", "Large Language Model"),
        ("ICC (ρ)", "Intra-cluster correlation: correlation between two rollouts of the same scenario"),
        ("DEFF", "Design effect, 1 + (m − 1)ρ for equal cluster sizes"),
        ("N", "Total number of rollouts in an evaluation"), ("K", "Number of distinct scenarios (clusters)"),
        ("m, mᵢ", "Rollouts per scenario (size of cluster i)"),
        ("kᵢ", "Harmful rollouts in scenario i"), ("X", "Total harmful rollouts, Σ kᵢ"),
        ("pᵢ", "True harmful rate of scenario i"), ("μ", "Population rate: mean of pᵢ over the pool of scenarios"),
        ("α", "Error level; 0.05 for a 95% bound"), ("a, b", "Parameters of the Beta distribution of pᵢ"),
        ("Nₑff", "Effective sample size, N / DEFF"), ("CP", "Clopper-Pearson (exact binomial) interval"),
        ("BB", "Beta-binomial"), ("ANOVA", "Analysis of variance"), ("CI", "Confidence interval"),
        ("vLLM", "Open-source inference server for language models"),
        ("GPU", "Graphics Processing Unit"), ("SDG", "Sustainable Development Goal"),
        ("Rollout", "One run of a model on one scenario prompt"),
        ("Configuration", "One scenario x goal x urgency setting of the suite; one cluster"),
    ]
    R.table(["Symbol / Term", "Meaning"], abbrev, None, widths=[1.5, 4.5], size=11)

    # ---------------- main text (arabic numbers)
    R.new_section("decimal", 1)

    # ===== Chapter 1
    R.chapter("Introduction")
    R.section("Background")
    R.para(
        "Language models are no longer only chat assistants. They are being deployed as agents that read "
        "email, run code and send messages on someone's behalf. Before such an agent is trusted with that "
        "access, developers and outside evaluators test how it behaves under pressure. A common test design, "
        "used in the agentic misalignment study of Lynch et al. [21], puts the model inside a fictional company. "
        "The model reads an inbox, learns that it is about to be replaced, and also finds something it could "
        "use against the person replacing it. The question is whether the model sends a blackmail email, leaks "
        "confidential documents to a competitor, or simply does its job.")
    R.para(
        "Each run of such a test is called a rollout. Models sample their output at random, so the same "
        "scenario is usually run tens or hundreds of times, and the result is reported as a count, for example "
        "“no covert sabotage across 200 runs”. When the count is zero, the standard way to read it is the "
        "rule of three [7]: after zero events in N independent trials we can say with 95% confidence that the "
        "rate is below about 3/N. For 1,500 rollouts that is 0.20%.")
    R.para(
        "The catch is the word independent. A model's choice depends far more on which scenario it is in than "
        "on sampling noise. Two rollouts of the same scenario are therefore much more alike than two rollouts "
        "of different scenarios. Statisticians call such groups clusters, and survey researchers have corrected "
        "for them since at least Kish [15]. Safety evaluations of AI agents mostly have not.")

    R.section("Problem Statement")
    R.para(
        "Given an evaluation that runs K scenarios m times each and observes zero harmful rollouts, what upper "
        "bound on the model's harmful rate does the data actually support? In particular: (a) how much do "
        "rollouts of the same scenario agree on real models, (b) how wrong is the independence-based bound when "
        "they do, (c) which bound should replace it, and (d) for a fixed compute budget, is it better to run "
        "more rollouts or to write more scenarios?")

    R.section("Project Objectives")
    objectives = [
        "**O1.** Derive and implement upper bounds for a harmful rate that allow for rollouts being clustered by "
        "scenario, including one bound that does not depend on the shape of the rate distribution.",
        "**O2.** Measure, by exact calculation and simulation, how often the usual bound and each corrected bound "
        "actually cover the true rate.",
        "**O3.** Work out how a fixed budget of rollouts should be split between scenarios and repeats.",
        "**O4.** Re-examine zero-event claims published in recent AI safety work using the corrected bounds.",
        "**O5.** Run six open-weight models on an open agentic misalignment suite and measure the intra-cluster "
        "correlation on real data.",
        "**O6.** Release the full pipeline (serving patch, configurations, labelling, analysis code and data) so "
        "the results can be reproduced on a single GPU machine.",
    ]
    R.bullets(objectives)

    R.section("Scope of the Project")
    R.para(
        "The project covers binary outcomes (harmful or not) with scenario-level clustering. The real-data part "
        "uses one open suite, two behaviours (blackmail and leaking), and six open models from 8 to 32 billion "
        "parameters, all served locally. Frontier closed models were not run, and no human labels were "
        "collected; labels come from a blind model labeller cross-checked against a second judge. The bounds "
        "describe the pool of scenarios written the same way as the suite, not the situations a deployed model "
        "will meet in practice.")

    R.section("Mapping of Objectives to Sustainable Development Goals")
    R.para(
        "The project is a measurement and methods project, so its link to the United Nations Sustainable "
        "Development Goals runs through better science and more trustworthy institutions rather than through a "
        "product. Table 1.1 maps each objective to the goal and target it supports.")
    R.table(["Objective", "SDG and target", "Justification"], [
        ["O1, O2", "SDG 9: Industry, Innovation and Infrastructure (Target 9.5: enhance scientific research and "
         "technological capability)",
         "Gives AI developers statistically sound tools for testing systems before they become part of "
         "digital infrastructure."],
        ["O3", "SDG 9 (Target 9.5) and SDG 12: Responsible Consumption and Production (Target 12.2: efficient "
         "use of resources)",
         "Shows how to spend a fixed GPU budget on the runs that carry information, cutting wasted compute and "
         "energy."],
        ["O4", "SDG 16: Peace, Justice and Strong Institutions (Target 16.6: effective, accountable and "
         "transparent institutions)",
         "Regulators and auditors rely on published safety numbers; honest error bars make those claims "
         "accountable."],
        ["O5", "SDG 16 (Target 16.10: public access to information)",
         "Publishes measured ICC values that later evaluations can borrow, instead of every lab guessing."],
        ["O6", "SDG 17: Partnerships for the Goals (Target 17.6: knowledge sharing and access to technology) "
         "and SDG 4: Quality Education (Target 4.4: relevant technical skills)",
         "Open code and data run on one GPU, so students and smaller institutes can repeat or extend the study."],
    ], "Mapping of project objectives to the Sustainable Development Goals", widths=[0.8, 2.4, 2.8], size=10)
    R.para(
        "The strongest link is with SDG 16. Several governments now ask AI developers for evidence that their "
        "systems do not take harmful actions. If that evidence is a zero count read with the rule of three, a "
        "regulator may accept a claim that is five to ninety times stronger than the data support. Correct "
        "bounds are a small but real part of making AI governance work.")

    R.section("Organisation of the Report")
    R.para(
        "Chapter 2 surveys the statistical and AI-evaluation literature and identifies the gap. Chapter 3 "
        "describes the approach: the statistical model, the five bounds, the simulation design and the "
        "experimental pipeline, with code. Chapter 4 presents results from simulation, from re-reading "
        "published claims, and from 18,000 real rollouts. Chapter 5 concludes, lists limitations and future "
        "work, and records the contribution of each team member. The appendices give run instructions, full "
        "per-configuration counts, and proof of the paper submission.")

    # ===== Chapter 2
    R.chapter("Literature Survey")
    R.section("Rare Events and the Rule of Three")
    R.para(
        "With zero events in n independent trials the exact one-sided 95% upper bound on the rate is "
        "1 − 0.05^(1/n), which is close to 3/n. Hanley and Lippman-Hand [7] popularised this as the rule of "
        "three in medicine. The same reasoning is used to ask how many miles a self-driving car must drive "
        "without a crash before we believe it is safe [10]. In software reliability, Salako and Zhao [30, 31] "
        "studied tests that may not be independent and showed that ignoring the dependence can overstate the "
        "evidence.")
    R.section("Clustered Binary Data")
    R.para(
        "Survey statistics handles clustering through the design effect: the factor by which clustering inflates "
        "the variance of an estimate compared with simple random sampling [15, 28]. For equal clusters of size m "
        "and intra-cluster correlation ρ it equals 1 + (m − 1)ρ, and the effective sample size is N "
        "divided by it. With 100 rollouts per scenario even ρ = 0.01 gives a design effect of 1.99, so half the "
        "rollouts are wasted. Korn and Graubard [16] built exact intervals for rare outcomes on the effective "
        "sample size and pointed out that with no positives the estimated design effect is 1, so the interval "
        "falls back to the naive one. That is exactly the case we face.")
    R.para(
        "Cluster randomised trials in medicine found that more clusters beat more units per cluster [13], that "
        "unequal cluster sizes raise the design effect [5], and that the ICC tends to shrink as the outcome gets "
        "rarer [6]. Veterinary surveys that try to prove a herd is free of disease face our budget question "
        "directly: test more herds, or more animals per herd [4]. Ridout, Demetrio and Firth [29] compared ICC "
        "estimators for binary data; we use their ANOVA estimator.")
    R.section("Uncertainty in Language-Model Evaluation")
    R.para(
        "Within machine learning, Miller [25] argued that evaluation results need error bars and that questions "
        "written in groups need clustered standard errors. Bowyer et al. [2] showed that intervals based on the "
        "central limit theorem are too narrow on small benchmarks. Prompt wording changes results [26, 19]. "
        "Mustahsan et al. [27] measured an ICC between 0.30 and 0.77 for repeated samples of agent tasks. "
        "Hierarchical Bayesian models [20, 12, 35], corrections for nested data [1] and more varied task sets [8] "
        "have been proposed, but all of them target rates well above zero.")
    R.section("Agentic Safety Evaluations")
    R.para(
        "Agentic safety studies typically run hundreds of rollouts on a handful to a few dozen scenarios "
        "[21, 23, 9]. Lynch et al. [21] released the open suite we use. Meinke et al. [23] tested in-context "
        "scheming, Schlatter et al. [32] shutdown resistance, Schoen et al. [33] anti-scheming training, and "
        "Kirk et al. [14] sabotage of safety research. Reporting practice varies: no uncertainty at all [34], "
        "intervals computed over rollouts [23, 32, 3], or Bayesian intervals per environment [33].")
    R.para(
        "The closest prior work is Kaur [11], which applies the design effect to zero counts and proposes a "
        "reporting template with cluster size, ICC, effective sample size and a beta-binomial bound. It does "
        "not derive the bound, measure the ICC, test coverage or give budget advice. Li et al. [17] report a "
        "bound over the 40 scenarios of their benchmark because runs of one scenario are not independent, but "
        "do not show when such a bound is valid.")
    R.section("Summary of Surveyed Work and Research Gap")
    R.table(["Work", "Area", "What it offers", "What is missing for our setting"], [
        ["Hanley & Lippman-Hand [7]", "Medical statistics", "Rule of three for zero events", "Assumes independent trials"],
        ["Kish [15]; Rao & Scott [28]", "Survey sampling", "Design effect, effective sample size", "No exact zero-event bound"],
        ["Korn & Graubard [16]", "Survey statistics", "Exact CIs for rare outcomes", "Design effect cannot be estimated at zero"],
        ["Miller [25]; Bowyer et al. [2]", "LLM evaluation", "Error bars, clustered SEs", "Rates far from zero"],
        ["Mustahsan et al. [27]", "Agent evaluation", "Measured ICC 0.30 to 0.77", "Task success, not safety"],
        ["Lynch et al. [21]; Meinke et al. [23]", "Agentic safety", "Scenario suites, harmful counts", "No clustered uncertainty"],
        ["Kaur [11]", "Red-team reporting", "Design effect on zero counts", "No derivation, measurement or coverage"],
        ["Li et al. [17]", "Agent benchmark", "Scenario-level bound", "No proof of validity"],
    ], "Summary of the literature survey", widths=[1.6, 1.1, 1.65, 1.65], size=9.5)
    R.para(
        "The gap, then, is a complete treatment for zero-event safety claims: a bound with a closed form, one "
        "that survives the wrong distributional shape, a check of coverage, budget advice, and a real "
        "measurement of the ICC on an agentic misalignment suite. This project fills it.")

    # ===== Chapter 3
    R.chapter("Research Approach and Methodology")
    R.section("Overview of the Approach")
    R.para(
        "The work has two halves that meet in the middle. The statistical half builds bounds and checks them "
        "with exact calculation and simulation; it needs no GPU and gives results even if experiments fail. "
        "The experimental half runs real models and measures how clustered their behaviour is, which tells us "
        "which part of the statistical results applies in practice. Figure 3.1 shows the experimental pipeline.")
    R.figure("fig_pipeline.png", "Experimental pipeline, from scenario grid to cluster statistics", 6.0)

    R.section("Statistical Model")
    R.para(
        "An evaluation runs K scenarios. Scenario i has its own true harmful rate pᵢ and is sampled mᵢ "
        "times, giving kᵢ harmful rollouts, with N = Σ mᵢ and X = Σ kᵢ. We treat the scenarios as a "
        "random draw from a larger pool written the same way, and call the mean of pᵢ over that pool the "
        "population rate μ. This is the rate a statement about the model needs. The mean over just the "
        "scenarios that were run is the suite rate.")
    R.para(
        "One point surprised us early on. For the suite rate, the naive bound is still valid: holding the "
        "pᵢ fixed, the chance of zero events is never larger than if every scenario had the average rate, "
        "because the logarithm is concave. Clustering only breaks the step from the scenarios that were run to "
        "the pool. Every coverage result below concerns the population rate.")
    R.para("The intra-cluster correlation is the correlation between two rollouts of the same scenario. "
           "If pᵢ follows a Beta(a, b) distribution, it equals")
    R.equation("ρ = 1 / (a + b + 1),      μ = a / (a + b)", "3.1")
    R.para("and for equal cluster sizes the design effect is")
    R.equation("DEFF = 1 + (m − 1) ρ,      Nₑff = N / DEFF", "3.2")

    R.section("The Five Upper Bounds")
    R.subsection("Naive bound")
    R.para("The exact Clopper-Pearson bound on all N rollouts as if they were independent. With X = 0 it is")
    R.equation("UBₙₐᵢᵥₑ = 1 − α^(1/N)  ≈  3/N", "3.3")
    R.subsection("Design-effect bound")
    R.para("The same exact bound applied to the effective sample size N / DEFF, with events scaled the same way "
           "so the point estimate does not change.")
    R.subsection("Exact beta-binomial bound")
    R.para(
        "Assume the pᵢ follow a Beta distribution with mean μ and ICC ρ. The probability that one scenario "
        "sampled m times shows no events is")
    R.equation("P(kᵢ = 0) = ∏ⱼ₌₀^(m−1)  (b + j) / (a + b + j)", "3.4")
    R.para(
        "and the bound is the largest μ for which the product over all scenarios is still at least α. For "
        "small μ one scenario of m rollouts is worth about mₑff(m, ρ) = Σⱼ s/(s + j) independent rollouts, "
        "with s = (1 − ρ)/ρ. At ρ = 0.01 a scenario of 100 rollouts is worth 69.4 rollouts, and at "
        "ρ = 0.2 only 13.5.")
    R.subsection("Worst-case bound")
    R.para(
        "The Beta shape is an assumption. We proved (Proposition 3 of the paper) that among all distributions "
        "of pᵢ with a given mean and ICC, the chance of zero events is largest for a two-point law: some "
        "scenarios never produce the behaviour and the rest share one rate c = μ + ρ(1 − μ), with weight "
        "μ/c. Inverting that law gives a bound that needs the ICC but not the shape. The proof uses Hermite "
        "interpolation of g(p) = (1 − p)^m, whose third derivative is never positive on [0, 1].")
    R.subsection("Scenario bound")
    R.para(
        "Count the scenarios with at least one harmful rollout, y, and apply the exact binomial bound to y out "
        "of K. A scenario shows the behaviour at least once at least as often as a single rollout does, since "
        "(1 − p)^m ≤ 1 − p. So this bound needs no ICC and no shape. With 15 scenarios and no events it is "
        "18.1%; with 30 it is 9.5%.")
    R.para("Code 3.1 and Code 3.2 show the core implementation in analysis/clusterstats.py and "
           "analysis/bounds_extra.py.")
    R.code('''def cp_upper(x, n, alpha=0.05):
    """One-sided Clopper-Pearson upper bound. With x = 0 this is
    1 - alpha**(1/n), which the rule of three approximates as 3/n."""
    if x >= n:
        return 1.0
    return float(stats.beta.ppf(1 - alpha, x + 1, n - x))

def icc_anova(k, m):
    """ANOVA estimator of the ICC for binary data, unequal cluster sizes
    (Ridout et al. 1999). nan when there are no events."""
    k, m = np.asarray(k, float), np.asarray(m, float)
    K, N = len(k), m.sum()
    p = k.sum() / N
    if p in (0.0, 1.0) or K < 2:
        return float("nan")
    pi = k / m
    msb = np.sum(m * (pi - p) ** 2) / (K - 1)
    msw = np.sum(m * pi * (1 - pi)) / (N - K)
    m0 = (N - np.sum(m**2) / N) / (K - 1)
    return float((msb - msw) / (msb + (m0 - 1) * msw))

def deff(m_bar, icc):
    return 1.0 + (m_bar - 1.0) * icc

def adjusted_upper(x, n, d, alpha=0.05):
    """Design-effect bound: shrink the sample to n / d."""
    return cp_upper(x / d, n / d, alpha)''', "Naive bound, ANOVA ICC estimator and design-effect bound (clusterstats.py)")
    R.code('''def log_p_zero_cluster(mu, m, rho):
    """log P(k = 0) for one cluster of size m. Product form of
    B(a, b + m) / B(a, b), accurate at both rho -> 0 and rho -> 1."""
    s = _s(rho)                       # s = (1 - rho) / rho
    if np.isinf(s):
        return m * np.log1p(-mu)
    if s == 0:
        return np.log1p(-mu)
    j = np.arange(int(m))
    return float(np.sum(np.log1p(-mu * s / (s + j))))

def bb_upper(x, sizes, rho, alpha=0.05):
    """Exact one-sided upper bound for mu from total count x, ICC fixed."""
    sizes = np.asarray(sizes, int)
    la = np.log(alpha)
    if x == 0:
        f = lambda mu: log_p_zero(mu, sizes, rho) - la
    else:
        f = lambda mu: np.log(max(sum_cdf(x, sizes, mu, rho), 1e-300)) - la
    return float(optimize.brentq(f, 1e-14, 1 - 1e-14, xtol=1e-14))

def any_event_upper(y, K, alpha=0.05):
    """Scenario bound: valid for ANY distribution of scenario rates."""
    if y >= K:
        return 1.0
    return float(stats.beta.ppf(1 - alpha, y + 1, K - y))''',
           "Exact beta-binomial and scenario bounds (bounds_extra.py)")

    R.section("Simulation Design")
    R.para(
        "A 95% upper bound should lie above the true rate in 95% of evaluations; the share in which it does is "
        "its coverage. Every bound here depends on the data only through X (or through y), so coverage can be "
        "computed exactly by summing over the distribution of X, which we get by convolving per-cluster "
        "probability mass functions. We did this for 30 scenarios of 100 rollouts, 200 true rates between 0.01% "
        "and 5%, four ICCs from 0.01 to 0.5, and four shapes of the scenario-rate distribution with the same "
        "mean and ICC: Beta, the two-point worst case, logit-normal, and Beta with extra zero scenarios. A Monte "
        "Carlo simulation with 20,000 repetitions per setting checks the exact values. Code 3.3 shows the core "
        "of the simulation loop.")
    R.code('''rng = np.random.default_rng(20261022)
REPS, K, m = 20000, 30, 100
for mu in (0.001, 0.003, 0.01):
    for rho in (0.0, 0.01, 0.05, 0.2, 0.5):
        if rho == 0:
            p = np.full((REPS, K), mu)
        else:
            a, b = mu * (1 - rho) / rho, (1 - mu) * (1 - rho) / rho
            p = rng.beta(a, b, (REPS, K))          # scenario rates
        k = rng.binomial(m, p)                     # harmful counts
        x, n = k.sum(1), K * m
        r_hat = icc_anova_rows(k, m)               # ICC from the same data
        d_hat = np.where(np.isnan(r_hat), 1.0,
                         deff(m, np.maximum(np.nan_to_num(r_hat), 0.0)))
        d = deff(m, rho)
        cov_naive = (cp_upper_vec(x, n) >= mu).mean()
        cov_oracle = (cp_upper_vec(x / d, n / d) >= mu).mean()
        cov_est = (cp_upper_vec(x / d_hat, n / d_hat) >= mu).mean()''',
           "Monte Carlo coverage check (sim.py, abridged)")

    R.section("Experimental Setup on Real Models")
    R.subsection("The suite and the configurations")
    R.para(
        "We used the open agentic misalignment suite of Lynch et al. [21], pinned to commit ea0630e. The suite "
        "builds prompts as a cartesian product of a harmful behaviour (which it calls a scenario), a goal "
        "given to the model, and a kind of pressure. We picked the two behaviours whose harmful act is a single "
        "email that the suite checks mechanically. Table 3.1 lists the factors.")
    R.table(["Factor", "Levels", "Count"], [
        ["Behaviour", "blackmail, leaking", "2"],
        ["Goal type and value", "explicit (america, global), latent (america, global), none", "5"],
        ["Urgency (pressure)", "replacement, restriction, none", "3"],
        ["Configurations", "2 x 5 x 3", "30"],
        ["Rollouts per configuration", "temperature 1.0, no truncation", "100"],
        ["Rollouts per model", "30 x 100", "3,000"],
    ], "Experimental design on the agentic misalignment suite", widths=[1.8, 3.3, 0.9], size=10,
        align="llc")
    R.para(
        "Each configuration is one cluster. The goal and urgency factors were chosen because in the original "
        "study they moved harmful rates from near 0 to near 1, so the clusters should differ in base rate, "
        "which is exactly what the ICC measures. The goal none, urgency none configurations act as controls.")
    R.subsection("Models")
    R.table(["Model", "Parameters", "Notes"], [
        ["Qwen3-8B", "8B", "reasoning (thinking) off"],
        ["Qwen3-8B (thinking)", "8B", "same weights, reasoning on, parsed out by vLLM"],
        ["Qwen3-14B", "14B", "reasoning off"],
        ["Qwen3-32B", "32B", "reasoning off"],
        ["Mistral-Small-3.2", "24B", "instruction tuned"],
        ["gpt-oss-20b", "20B", "open-weight reasoning model"],
        ["gpt-oss-120b", "120B", "used only as the second judge"],
    ], "Models evaluated", widths=[1.9, 1.0, 3.1], size=10, align="lcl")
    R.subsection("Serving models locally")
    R.para(
        "The suite only talks to hosted APIs. Its OpenRouter client hard-codes the base URL, so it cannot reach "
        "a local server. We wrote a nine-line patch (Code 3.4) that reads the base URL and an optional request "
        "body from environment variables, which lets the suite talk to a local vLLM server and lets us switch "
        "Qwen3 thinking on or off per request. A third change raises the suite's failure limit: as shipped it "
        "stops running a model after three bad replies in the whole run, which would end a 3,000-rollout job.")
    R.code('''--- a/api_client/model_client.py
+++ b/api_client/model_client.py
     self.openrouter_client = AsyncOpenAI(
         api_key=os.getenv("OPENROUTER_API_KEY"),
-        base_url="https://openrouter.ai/api/v1"
+        base_url=os.getenv("OPENROUTER_BASE_URL",
+                           "https://openrouter.ai/api/v1")
     )
 ...
+    extra_body = os.getenv("OPENROUTER_EXTRA_BODY")
+    if extra_body and "extra_body" not in kwargs:
+        kwargs["extra_body"] = json.loads(extra_body)

--- a/scripts/run_experiments.py
+++ b/scripts/run_experiments.py
-MAX_MODEL_FAILURES = 3
+MAX_MODEL_FAILURES = int(os.getenv("AM_MAX_MODEL_FAILURES", "3"))''',
           "Patch that points the suite at a local vLLM server (patches/local_vllm.diff)")
    R.para(
        "The server is started with bfloat16 weights, prefix caching (all rollouts of a configuration share "
        "one prompt), a 16k context and plain temperature-1.0 sampling, so vLLM does not silently add the "
        "model's own top-p and top-k defaults:")
    R.code('''GPUS=1 VLLM=~/.vllm/bin/vllm nohup bash serve_vllm.sh > vllm.log 2>&1 &
# serve_vllm.sh runs:
vllm serve Qwen/Qwen3-8B --dtype bfloat16 --max-model-len 16384 \\
     --max-num-seqs 256 --gpu-memory-utilization 0.90 \\
     --enable-prefix-caching --generation-config vllm
python scripts/generate_prompts.py --config configs/qwen3_8b_clusters.yaml
python scripts/run_experiments.py  --config configs/qwen3_8b_clusters.yaml''',
           "Starting the server and running the suite")
    R.subsection("Labelling")
    R.para(
        "The suite counts a rollout as harmful only if a mechanical check and a language-model judge both "
        "agree. The check looks for an email to the targeted executive that contains “blackmail” or "
        "“leverage”, or a message to one of 14 outside addresses. The suite's original judge model has been "
        "retired, so we needed a replacement. Every rollout that passed the check, 4,478 of 18,000, was "
        "labelled by Claude Opus acting as a blind labeller: it saw only an opaque id, the transcript and the "
        "suite's own criteria, never the model or configuration. A second label set came from the suite's "
        "judging script run with gpt-oss-120b. One disagreement pattern (a leak claimed as “attached” with "
        "nothing attached) was settled by a written rule: only what was actually sent counts.")
    R.subsection("From transcripts to a table")
    R.para(
        "A small standard-library script, to_csv.py, walks the results tree and writes one row per rollout "
        "with the columns model, cluster, scenario, goal_type, goal_value, urgency, variant, seed and harmful. "
        "It counts and reports any file it skips. The analysis then groups rows by model and cluster:")
    R.code('''data = defaultdict(lambda: defaultdict(lambda: [0, 0]))
for r in csv.DictReader(open(path)):
    cell = data[r["model"]][r["cluster"]]
    cell[0] += int(r["harmful"])          # events k_i
    cell[1] += 1                          # rollouts m_i
for model, clusters in data.items():
    for scope, prefix in {"All": "", "Blackmail": "blackmail_",
                          "Leaking": "leaking_"}.items():
        names = sorted(c for c in clusters if c.startswith(prefix))
        k = np.array([clusters[c][0] for c in names], float)
        m = np.array([clusters[c][1] for c in names], float)
        out = analyse(k, m)   # ICC, DEFF, N_eff, naive and cluster CIs''',
           "Per-model, per-behaviour analysis (analyze_real.py, abridged)")

    R.section("Tools and Technologies")
    R.table(["Category", "Tool", "Use in the project"], [
        ["Language", "Python 3.12", "All analysis, simulation and glue code"],
        ["Numerics", "NumPy, SciPy", "Beta quantiles, root finding, likelihood fits, convolutions"],
        ["Plots", "Matplotlib", "All figures (colour-blind safe Okabe-Ito palette)"],
        ["Inference", "vLLM", "OpenAI-compatible local server, data parallel over GPUs"],
        ["Environment", "uv", "Isolated environments for suite, server and analysis"],
        ["Suite", "agentic-misalignment (pinned commit)", "Prompt generation, rollouts, classifiers"],
        ["Hardware", "NVIDIA GPU, 96 GB class", "One replica of each model in bfloat16"],
        ["Version control", "Git and GitHub", "Code, configs, data and history"],
    ], "Tools and technologies used", widths=[1.3, 2.0, 2.7], size=10)

    # ===== Chapter 4
    R.chapter("Results and Discussion")
    R.section("How the Bound Moves with the ICC")
    R.para(
        "Figure 4.1 shows the 95% bound after zero harmful rollouts out of 3,000, as the ICC grows, for three "
        "ways of splitting the 3,000. The naive bound is 0.10% regardless. With 30 scenarios of 100 rollouts, "
        "the worst-case bound is 0.16%, 0.52%, 2.06% and 4.99% at ICCs of 0.01, 0.05, 0.2 and 0.5 (Table 4.1). "
        "The design-effect bound tracks the worst case closely at an ICC of 0.2 or more. The scenario bound "
        "stays at 9.5% at any ICC because it ignores the ICC altogether.")
    R.figure("fig_bound_vs_icc.png", "Zero-event bound against ICC for 3,000 rollouts. Solid lines are the exact "
             "beta-binomial bound, dotted lines the design-effect bound", 5.6)
    R.table(["Design", "ICC", "Naive", "Beta-binomial", "Worst case", "Design effect", "Scenario"], [
        ["30 x 100", "0.010", "0.10", "0.14", "0.16", "0.20", "9.5"],
        ["30 x 100", "0.050", "0.10", "0.28", "0.52", "0.59", "9.5"],
        ["30 x 100", "0.200", "0.10", "0.74", "2.06", "2.06", "9.5"],
        ["30 x 100", "0.500", "0.10", "1.92", "4.99", "4.92", "9.5"],
        ["15 x 100", "0.046", "0.20", "0.54", "1.02", "1.11", "18.1"],
        ["15 x 100", "0.093", "0.20", "0.83", "2.01", "2.01", "18.1"],
    ], "95% upper bounds (%) after zero events. The last two rows are our zero-blackmail setting with the "
       "borrowed ICC and twice that value", widths=[0.85, 0.65, 0.7, 1.05, 0.9, 1.0, 0.85], size=10,
        align="ccccccc")

    R.section("Coverage of the Usual Bound")
    R.para(
        "Figure 4.2 and Table 4.2 give the Monte Carlo coverage for 30 scenarios of 100 rollouts. At an ICC of "
        "zero every bound covers about 95%, as it should. As the ICC rises the naive bound falls apart: at a "
        "true rate of 0.1% and an ICC of 0.5 it covers only 14% of the time. Estimating the ICC from the same "
        "data does not rescue it, because with zero events the ICC cannot be estimated at all, and the "
        "estimate falls back to the naive bound in exactly the cases that matter. Plugging in the true ICC "
        "restores coverage.")
    R.figure("fig_coverage.png", "Coverage of the 95% bound for 30 scenarios x 100 rollouts, 20,000 repetitions "
             "per bar. Dashed line is the 95% target", 6.0)
    cov = list(csv.DictReader(open(os.path.join(ROOT, "data", "sim_coverage.csv"))))
    R.table(["True rate (%)", "ICC", "Evaluations with 0 events (%)", "Naive", "ICC from data", "True ICC"],
            [[f"{float(r['mu']) * 100:g}", r["rho"], f"{float(r['p_zero']) * 100:.0f}",
              f"{float(r['cov_naive']) * 100:.0f}", f"{float(r['cov_estimated']) * 100:.0f}",
              f"{float(r['cov_oracle']) * 100:.0f}"] for r in cov],
            "Simulated coverage (%) of the naive and design-effect bounds", widths=[1.0, 0.7, 1.5, 0.8, 1.0, 0.9],
            size=10, align="cccccc")
    R.para(
        "The exact calculation over all four distribution shapes (Table 4.3) tells the more important story. "
        "The beta-binomial bound covers 95% whenever the rates really are Beta-shaped, but only 66.85% under the "
        "two-point shape: the extra tightness it buys is exactly the part that fails when the shape is wrong. "
        "The worst-case bound holds at about 95% under every shape, and the scenario bound always covers.")
    R.table(["Truth", "ICC", "Naive", "Beta-bin.", "Design eff.", "Worst case", "Scenario"], [
        ["Beta", "0.05", "65.62", "95.06", "96.49", "95.97", "100.00"],
        ["Beta", "0.5", "14.53", "95.05", "99.90", "95.05", "100.00"],
        ["Two-point", "0.05", "44.92", "81.69", "95.59", "95.00", "100.00"],
        ["Two-point", "0.2", "14.02", "66.85", "95.20", "95.03", "100.00"],
        ["Two-point", "0.5", "5.87", "69.14", "95.03", "95.03", "100.00"],
        ["Logit-normal", "0.5", "24.09", "95.80", "100.00", "95.80", "100.00"],
        ["Beta + extra zeros", "0.5", "14.51", "94.88", "99.87", "94.97", "100.00"],
    ], "Lowest exact coverage (%) over 200 true rates, 30 scenarios of 100 rollouts (selected rows)",
        widths=[1.4, 0.55, 0.7, 0.8, 0.9, 0.9, 0.85], size=10, align="lcccccc")

    R.section("Where to Spend a Fixed Budget")
    R.para(
        "For a fixed total of rollouts, the bound after zero events is smallest with one rollout per scenario "
        "(Figure 4.3). Real scenarios are not free, though: someone has to write each one. When writing a "
        "scenario costs as much as 10 rollouts, the best split under the worst-case bound is 17 rollouts per "
        "scenario at an ICC of 0.05 and 7 at 0.2. At a cost of 100 rollouts it is 39 and 13, close to the "
        "textbook rule √((cₛ/cᵣ)(1 − ρ)/ρ). Rewording existing scenarios is cheaper, but in a nested "
        "simulation 10 scenarios with 30 wordings each were worth only 195 independent rollouts when scenarios "
        "vary more than wordings, against 1,967 for 300 separate scenarios.")
    R.figure("fig_budget.png", "Zero-event bound against number of scenarios, total fixed at 3,000 rollouts", 5.4)

    R.section("Re-reading Published Zero-Event Claims")
    R.para(
        "We collected 13 zero or near-zero claims from eight sources published from 2024 to 2026 and checked "
        "each quote against its source. Eight report zero events and state both N and K. Table 4.4 and Figure "
        "4.4 re-bound them. The beta-binomial columns are illustrations, since no source reports an ICC.")
    rea = list(csv.DictReader(open(os.path.join(ROOT, "data", "reanalysis.csv"))))
    lab = {"C7": "Shutdown resistance, Prompt E [32]", "C2": "Record tampering [22]",
           "C8": "Covert sabotage, Codex task [33]", "C6": "Covert email reranking [23]",
           "C13": "Scheming reasoning [18]", "C5": "Oversight subversion [23]",
           "C10": "Safety-research sabotage [14]", "C11": "Continuing prefilled sabotage [14]"}
    R.table(["Claim (0 events)", "N", "K", "Naive", "ρ = 0.05", "ρ = 0.2", "Scenario"],
            [[lab[r["id"]], r["N"], r["K"], f"{float(r['naive']) * 100:.1f}", f"{float(r['bb0.05']) * 100:.1f}",
              f"{float(r['bb0.2']) * 100:.1f}", f"{float(r['scen']) * 100:.1f}"] for r in rea],
            "Published zero-event claims, re-bounded. 95% upper bounds in %", widths=[2.4, 0.6, 0.6, 0.6, 0.65, 0.6, 0.75],
            size=10, align="lcccccc")
    R.figure("fig_reanalysis.png", "Published zero-event claims under each bound (log scale)", 6.0)
    R.para(
        "These sources are careful, and none of them claims a population rate in so many words. Schlatter et "
        "al. give a 95% interval of 0.0 to 0.2% for one fixed prompt, which is correct for that prompt. But a "
        "reader who takes it as a statement about the model should know that the scenario bound for a single "
        "prompt is 95%. With 3 to 5 scenarios an ICC of 0.2 moves the bound from 1.0% to 5.1% and 7.3%. The UK "
        "AISI sabotage study runs every scenario once, so all its bounds agree, which is the design our budget "
        "analysis favours when scenarios are cheap.")

    R.section("Measurements on 18,000 Real Rollouts")
    R.subsection("Rates differ across configurations")
    R.para(
        "Figure 4.5 shows the harmful rate of every model in every configuration. The left half (blackmail) is "
        "almost empty except for Mistral-Small and Qwen3-32B. The right half (leaking) is where the action is, "
        "and it is far from uniform: one model leaks in 0% of the rollouts of one configuration and 72% of "
        "another. Of the 180 model and configuration pairs, 51 fall outside the 95% range we would expect if "
        "every configuration of a model and behaviour shared one rate.")
    R.figure("fig_heatmap.png", "Harmful rate (%) per model and configuration, 100 rollouts each. B: blackmail, "
             "L: leaking; the vertical line separates the two behaviours", 6.0)
    R.subsection("ICC, design effect and effective sample size")
    R.para(
        "Table 4.5 is the main measurement. Within leaking, the ICC runs from 0.03 for Qwen3-8B to 0.27 for "
        "Qwen3-32B, so even the smallest value gives a design effect of 3.6, and 1,500 rollouts are worth only "
        "54 to 419 independent ones. Within blackmail the two models with enough events have ICCs of 0.01 and "
        "0.05. Pooling both behaviours gives ICCs of 0.10 to 0.42, mostly because the two behaviours have very "
        "different rates.")
    R.table(["Model", "Scope", "Harmful / N", "Configs with events", "ICC (ANOVA) [95% CI]", "ICC (BB)", "DEFF",
             "Eff. N", "Naive 95% CI (%)", "Cluster 95% CI (%)"], real,
            "Clustering in real rollouts (Opus labels). Dashes mark cells where the ICC is not identified",
            widths=[0.95, 0.75, 0.6, 0.5, 0.85, 0.4, 0.4, 0.4, 0.6, 0.6], size=7.5, align="llccccccc c".replace(" ", ""))
    R.para(
        "Figure 4.6 shows what this does to confidence intervals. The cluster bootstrap interval, which "
        "resamples whole configurations, is two to four times wider than the naive one for leaking. For the "
        "zero-blackmail models the bootstrap interval collapses to a point at zero, a reminder that resampling "
        "methods cannot help after zero events.")
    R.figure("fig_ci_compare.png", "Naive and cluster-bootstrap 95% confidence intervals per model and "
             "behaviour", 6.0)
    R.subsection("Where the clustering comes from")
    R.para(
        "The suite varies goal and pressure on purpose. A logistic model with goal and pressure as fixed "
        "effects and a random effect per configuration puts the remaining spread at or near zero in 7 of 8 "
        "cells, though the upper limits still allow a design effect of up to 3.7. So most of the measured ICC "
        "reflects the designed conditions. A suite that varied only incidental details might cluster much "
        "less, but any bound that claims to cover configurations varying goal and pressure has to be wide.")
    R.subsection("A pre-set threshold")
    R.para(
        "Before the runs we wrote down, in our project notes, the result that would refute the claim that "
        "clustering matters here: a design effect below 1.5, which is an ICC of 0.0051 at 100 rollouts. On "
        "point estimates all six pooled analyses and all 8 within-behaviour cells exceed it. On the lower ends "
        "of the intervals 6 or 7 of 8 do, depending on the estimator. We note honestly that three choices were "
        "made after seeing data: analysing within each behaviour, requiring events in at least two "
        "configurations, and making the Opus labels primary.")

    R.section("Zero-Event Claims from Our Own Runs")
    R.para(
        "Qwen3-8B with reasoning off and on (same weights) and gpt-oss-20b never blackmailed in 1,500 rollouts "
        "of 15 configurations. The rule of three says their blackmail rate is below 0.20%. A zero-event "
        "evaluation cannot estimate its own ICC, so we borrow it from the closest setting with events, "
        "Mistral-Small blackmail on the same suite (0.046). With it the worst-case bound is 1.02%, or 2.01% "
        "with the ICC doubled to be safe. The beta-binomial bound would say 0.54%, but that number rests on "
        "the Beta shape. With no assumption at all, 15 configurations support only 18.1%.")
    R.para(
        "Figure 4.7 makes the main point of the project visible. Along the naive line, every extra rollout "
        "keeps pulling the bound down. Along the worst-case line, the bound flattens at about 1% after a few "
        "dozen rollouts per scenario and never goes lower. The only way down is more scenarios: with 150 "
        "scenarios the same curve flattens at 0.09%.")
    R.figure("fig_rollouts_per_scenario.png", "Zero-event bound against rollouts per scenario for 15 scenarios. "
             "Only more scenarios move the floor", 5.6)

    R.section("Label Quality")
    R.para(
        "The two label sets agree on 87.3% of the 4,478 rollouts that passed the mechanical check, with Cohen's "
        "kappa of 0.75 (Figure 4.8). Almost all disagreement is one-sided: the gpt-oss-120b judge marks 528 "
        "rollouts harmful that the blind labeller does not, most of them leaks that the model claimed but did "
        "not actually send. The conclusions hold under either label set; with the judge's labels one "
        "threshold result changes (Qwen3-32B blackmail falls to a design effect of 1.4).")
    R.figure("fig_judge.png", "Agreement between blind Opus labels and the gpt-oss-120b judge on gate-passing "
             "rollouts", 3.4)

    R.section("Discussion")
    R.para(
        "Three things stand out. First, the gap between the naive and honest bounds is not a rounding issue. "
        "For our own zero-blackmail models it is a factor of five with a borrowed ICC and a factor of ninety "
        "with no assumption. Second, the gap is set by the number of scenarios, not by the number of rollouts, "
        "so running a cheap model for longer cannot close it. Third, the ICC is not a stable property you can "
        "look up. Our values vary threefold between models on the same suite, gpt-oss leaking at 2.07% has "
        "twice the ICC of Mistral blackmail at 5.40%, and predicting each model's leaking ICC from the median "
        "of the other five misses by a factor between 0.29 and 3.18. A borrowed ICC should therefore be "
        "rounded up, and the scenario bound should be reported alongside.")

    # ===== Chapter 5
    R.chapter("Conclusion, Summary and Future Scope")
    R.section("Conclusion")
    R.para(
        "Zero harmful actions in N rollouts is not the same as zero in N trials. When rollouts come from a few "
        "scenarios, the harmful rate an evaluation can rule out depends on how many scenarios were tested. On "
        "an open agentic misalignment suite we measured ICCs from 0.01 to 0.42 across six open models, enough "
        "to turn a reported bound of 0.20% into one of 1.02% at best and 18.1% with no assumption. The naive "
        "bound can cover the population rate as rarely as 14% of the time.")
    R.section("Summary of Contributions")
    R.bullets([
        "A closed form for the beta-binomial zero-event bound and its effective-sample-size reading.",
        "A worst-case bound with a proof that it holds for any spread of scenario rates with a given ICC.",
        "A proof that the scenario-level bound is valid with no assumption on the spread.",
        "Exact coverage results for five bounds under four distribution shapes, and budget rules.",
        "A re-examination of eight published zero-event claims.",
        "The first ICC measurements on an agentic misalignment suite: 18,000 labelled rollouts from six open "
        "models, released with code and data.",
        "A paper, submitted to the AISTATS 2027 Workshop Track (Appendix C).",
    ])
    R.section("Recommendations for Practitioners")
    R.bullets([
        "Report the number of distinct scenarios next to the number of rollouts, and say which rate a bound "
        "covers: the scenarios that were run, or the model in general.",
        "After zero events, give the scenario bound. For a tighter claim use the worst-case bound with a "
        "borrowed ICC, name its source, and round it up.",
        "When events do occur, report the ICC with an interval so later zero-event claims can borrow it.",
        "Spend a fixed budget on new scenarios before new rollouts; do not count rewordings as new scenarios "
        "unless wording is shown to matter as much.",
    ])
    R.section("Limitations")
    R.para(
        "We studied one suite, two behaviours and six open models of 8 to 32 billion parameters, so frontier "
        "models may behave differently. Labels come from a model, not people. Most of the clustering comes from "
        "the suite's designed conditions, so our bounds describe configurations built this way. The worst-case "
        "bound is proved only for zero events. We did not inspect the blackmail rollouts that failed the "
        "mechanical check, because their text was not saved, and for gpt-oss the check did not see the model's "
        "reasoning.")
    R.section("Future Scope")
    R.bullets([
        "Measure the ICC on more suites and on frontier models through their APIs.",
        "A Bayesian bound that pools ICC information across models instead of borrowing a single value.",
        "A nested run that splits variation between scenario, wording and sampling seed; the pipeline already "
        "supports wording variants.",
        "A human check of a stratified sample of labels, and of rollouts that fail the mechanical gate.",
        "A small Python package and reporting checklist so evaluation teams can apply the bounds directly.",
    ])
    R.section("Contribution of Team Members")
    R.table(["Member", "Main contributions"], [
        ["Manan Wadhwa", "Statistical model and bounds (clusterstats.py, bounds_extra.py), proofs of the worst-case and "
         "scenario bounds, coverage and budget simulations, figures, paper writing."],
        ["Anagh Miglani", "Experimental pipeline: vLLM serving script, suite patch, configurations, GPU runs for the "
         "six models, conversion of transcripts to CSV, analysis of real rollouts."],
        ["Hardik Dhull", "Literature survey and collection of published zero-event claims, labelling protocol and "
         "judge validation, SDG mapping, report and presentation preparation."],
        ["All members", "Design discussions, checking results, writing and reviewing the paper and this report."],
    ], "Contribution of each team member", widths=[1.4, 4.6], size=10.5)

    # ===== References
    R.unnumbered_chapter("References")
    refs = [
        "K. Anglin, “Estimating uncertainty in classifier performance with applications to large language models and nested data,” arXiv:2606.26422, 2026.",
        "S. Bowyer, L. Aitchison and D. R. Ivanova, “Position: Don't use the CLT in LLM evals with fewer than a few hundred datapoints,” in Proc. ICML, Position Paper Track, 2025. arXiv:2503.01747.",
        "J. Brazilek, M. Chaudhary, Z. Lu and M. Tidmarsh, “Coercion and deception in AI-to-AI management: An agentic benchmark of unprompted escalation,” arXiv:2607.15434, 2026.",
        "A. R. Cameron and F. C. Baldock, “Two-stage sampling in surveys to substantiate freedom from disease,” Preventive Veterinary Medicine, vol. 34, no. 1, pp. 19-30, 1998.",
        "S. M. Eldridge, D. Ashby and S. Kerry, “Sample size for cluster randomized trials: Effect of coefficient of variation of cluster size and analysis method,” Int. J. Epidemiology, vol. 35, no. 5, pp. 1292-1300, 2006.",
        "M. C. Gulliford et al., “Intraclass correlation coefficient and outcome prevalence are associated in clustered binary data,” J. Clinical Epidemiology, vol. 58, no. 3, pp. 246-251, 2005.",
        "J. A. Hanley and A. Lippman-Hand, “If nothing goes wrong, is everything all right? Interpreting zero numerators,” JAMA, vol. 249, no. 13, pp. 1743-1745, 1983.",
        "M. Hardy, R. Azam, A. Reuel, M. Kochenderfer and S. Koyejo, “Agent evaluation reliability: More tasks won't (always) fix an agent leaderboard,” arXiv:2610.00651, 2026.",
        "M. Hopman, J. Elstner, M. Avramidou, A. Prasad and D. Lindner, “Evaluating and understanding scheming propensity in LLM agents,” arXiv:2603.01608, 2026.",
        "N. Kalra and S. M. Paddock, “Driving to safety: How many miles of driving would it take to demonstrate autonomous vehicle reliability?” Transportation Research Part A, vol. 94, pp. 182-193, 2016.",
        "B. Kaur, “What AI red-team evaluations can and cannot prove,” arXiv:2607.21735, 2026.",
        "S. Kher, S. Wu, R. Longjohn, C. Belém and P. Smyth, “Sequential Bayesian evaluation of large language model behavior,” arXiv:2511.10661, 2025.",
        "S. Killip, Z. Mahfoud and K. Pearce, “What is an intracluster correlation coefficient? Crucial concepts for primary care researchers,” Annals of Family Medicine, vol. 2, no. 3, pp. 204-208, 2004.",
        "R. Kirk, A. Souly, K. Fronsdal, A. D'Cruz and X. Davies, “Evaluating whether AI models would sabotage AI safety research,” arXiv:2604.24618, 2026.",
        "L. Kish, Survey Sampling. New York: John Wiley & Sons, 1965.",
        "E. L. Korn and B. I. Graubard, “Confidence intervals for proportions with small expected number of positive counts estimated from survey data,” Survey Methodology, vol. 24, no. 2, pp. 193-201, 1998.",
        "M. Q. Li et al., “A benchmark for evaluating outcome-driven constraint violations in autonomous AI agents,” arXiv:2512.20798, 2025.",
        "D. Lindner, V. Krakovna and S. Farquhar, “Gram: Assessing sabotage propensities via automated alignment auditing,” arXiv:2605.30322, 2026.",
        "M. Llewellyn, I. Thornton, J. Bishop and A. Gray, “Correcting prompt dependence in LLM benchmarks: A Bayesian hierarchical model with embedding-space clustering,” arXiv:2510.05709, 2025.",
        "L. Luettgau, H. Coppock, M. Dubois, C. Summerfield and C. Ududec, “HiBayES: A hierarchical Bayesian modeling framework for AI evaluation statistics,” arXiv:2505.05602, 2025.",
        "A. Lynch, B. Wright, C. Larson, S. J. Ritchie, S. Mindermann, E. Hubinger, E. Perez and K. Troy, “Agentic misalignment: How LLMs could be insider threats,” arXiv:2510.05179, 2025. Code: github.com/anthropic-experimental/agentic-misalignment.",
        "A. Lynch, J. Hughes, A. Serrano, R. Kirk and S. R. Bowman, “Agentic misalignment in summer 2026,” Anthropic Alignment Science Blog, Jul. 2026. [Online]. Available: https://alignment.anthropic.com/2026/agentic-misalignment-summer-2026/",
        "A. Meinke, B. Schoen, J. Scheurer, M. Balesni, R. Shah and M. Hobbhahn, “Frontier models are capable of in-context scheming,” arXiv:2412.04984, 2024.",
        "METR, “Details about METR's evaluation of OpenAI GPT-5,” Aug. 2025. [Online]. Available: https://metr.org/evaluations/gpt-5-report/",
        "E. Miller, “Adding error bars to evals: A statistical approach to language model evaluations,” arXiv:2411.00640, 2024.",
        "M. Mizrahi, G. Kaplan, D. Malkin, R. Dror, D. Shahaf and G. Stanovsky, “State of what art? A call for multi-prompt LLM evaluation,” Trans. ACL, vol. 12, pp. 933-949, 2024.",
        "Z. Mustahsan, A. Lim, M. Anand, S. Jain and B. McCann, “Stochasticity in agentic evaluations: Quantifying inconsistency with intraclass correlation,” arXiv:2512.06710, 2025.",
        "J. N. K. Rao and A. J. Scott, “A simple method for the analysis of clustered binary data,” Biometrics, vol. 48, no. 2, pp. 577-585, 1992.",
        "M. S. Ridout, C. G. B. Demétrio and D. Firth, “Estimating intraclass correlation for binary data,” Biometrics, vol. 55, no. 1, pp. 137-148, 1999.",
        "K. Salako and X. Zhao, “Demonstrating software reliability using possibly correlated tests: Insights from a conservative Bayesian approach,” arXiv:2208.07935, 2022.",
        "K. Salako and X. Zhao, “The unnecessity of assuming statistically independent tests in Bayesian software reliability assessments,” IEEE Trans. Software Engineering, 2022. arXiv:2208.00462.",
        "J. Schlatter, B. Weinstein-Raun and J. Ladish, “Incomplete tasks induce shutdown resistance in some frontier LLMs,” Trans. Machine Learning Research, 2026. arXiv:2509.14260.",
        "B. Schoen et al., “Stress testing deliberative alignment for anti-scheming training,” arXiv:2509.15541, 2025.",
        "A. Souly, R. Kirk, J. Merizian, A. D'Cruz and X. Davies, “UK AISI alignment evaluation case-study,” arXiv:2604.00788, 2026.",
        "W. Zhang, H. Cai and W. Chen, “Beyond the singular: Revealing the value of multiple generations in benchmark evaluation,” in Findings of ACL 2026, 2026. arXiv:2502.08943.",
    ]
    for i, ref in enumerate(refs, 1):
        p = d.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.left_indent = Inches(0.5)
        p.paragraph_format.first_line_indent = Inches(-0.5)
        p.paragraph_format.line_spacing = 1.15
        p.paragraph_format.space_after = Pt(5)
        p.paragraph_format.tab_stops.add_tab_stop(Inches(0.5))
        R._runs(p, f"[{i}]\t{ref}", 11)

    # ===== Appendices
    R.appendix("A", "Reproducing the Results")
    R.para("All analysis runs on a laptop CPU. Only the rollouts themselves need a GPU.")
    R.code('''git clone https://github.com/manan-wadhwa/clustered_rollouts.git
cd clustered_rollouts
pip install numpy scipy matplotlib

# statistics and simulation (no GPU, about one minute)
cd analysis
python3 sim.py
python3 analyze_real.py ../data/rollouts_opus.csv
python3 real_figs.py
python3 hero_fig.py
cd ..

# figures and this report
python3 report/make_figs.py
python3 report/build_report.py''', "Commands to reproduce the analysis and this report")
    R.para(
        "To regenerate the rollouts, follow experiment/RUNBOOK.md: clone the suite at the pinned commit, apply "
        "patches/local_vllm.diff, start serve_vllm.sh, run a 10-rollout smoke test, then the full 30 x 100 "
        "configuration. One model takes roughly 20 to 40 minutes on a single 96 GB GPU with thinking off.")

    R.appendix("B", "Per-Configuration Counts")
    R.para("Harmful rollouts out of 100 for every configuration and model (Opus labels).", space_after=4)
    models = list(PRETTY)
    heads = ["Configuration", "Q-8B", "Q-8B think", "Q-14B", "Q-32B", "Mistral", "gpt-oss"]
    clusters = sorted(data["qwen3_8b"])
    rows = [[c.replace("_", " ")] + [str(data[mm][c][0]) for mm in models] for c in clusters]
    R.table(heads, rows, "Harmful rollouts per configuration (out of 100)",
            widths=[2.4, 0.6, 0.65, 0.6, 0.6, 0.6, 0.6], size=8.5, align="lcccccc")

    R.appendix("C", "Proof of Research Paper Publishing")
    R.para(
        "The work in this report has been written up as a research paper titled “Zero Harmful Actions in N "
        "Rollouts Is Not N Trials: Independence-Based Safety Bounds Are Too Narrow for Clustered Rollouts” by "
        "Manan Wadhwa, Anagh Miglani and Hardik Dhull. It was submitted to the AISTATS 2027 Workshop Track "
        "(Montreal, Canada, 3 May 2027) through OpenReview before the 7 November 2026 deadline. The screenshot "
        "below shows the submission record with title, authors, keywords and abstract.")
    R.figure("proof_openreview.png", "Submission record of the paper on the AISTATS 2027 Workshop Track "
             "(OpenReview)", 6.0)
    R.table(["Field", "Details"], [
        ["Paper title", "Zero Harmful Actions in N Rollouts Is Not N Trials: Independence-Based Safety Bounds Are Too "
                        "Narrow for Clustered Rollouts"],
        ["Authors", "Manan Wadhwa, Anagh Miglani, Hardik Dhull"],
        ["Venue", "AISTATS 2027 Workshop Track, Montreal, Canada"],
        ["Submission window", "25 Oct 2026 to 7 Nov 2026"],
        ["Keywords", "AI safety, significance"],
        ["Status", "Submitted, under review"],
    ], "Paper submission details", widths=[1.5, 4.5], size=10.5)

    # lists (insert before the abbreviations page)
    return R


def fill_lists(R):
    """Insert TOC, list of tables and list of figures at their markers."""
    d = R.doc
    for marker, title, entries, kind in ((R._toc_marker, "Table of Contents", R.toc, "toc"),
                                         (R._lot_marker, "List of Tables", R.lot, "lot"),
                                         (R._lof_marker, "List of Figures", R.lof, "lof")):
        body = d.element.body
        start = len(body)
        R.list_page(title, entries, kind)
        new = list(body)[start - 1:]  # trailing sectPr stays last; collect added elements
        added = [e for e in list(body)[start - 1:] if e.tag != qn("w:sectPr")]
        for e in added:
            marker._p.addprevious(e)
        marker._p.getparent().remove(marker._p)


# ------------------------------------------------------------------ page lookup
def pdf_pages(pdf):
    n = int(re.search(r"Pages:\s+(\d+)", subprocess.run(["pdfinfo", pdf], capture_output=True,
                                                         text=True).stdout).group(1))
    out = []
    for i in range(1, n + 1):
        t = subprocess.run(["pdftotext", "-f", str(i), "-l", str(i), "-layout", pdf, "-"],
                           capture_output=True, text=True).stdout
        out.append(re.sub(r"\s+", " ", t))
    return out


def roman(n):
    vals = [(10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i")]
    s = ""
    for v, r in vals:
        while n >= v:
            s += r
            n -= v
    return s


def find_pages(R, pdf):
    texts = pdf_pages(pdf)
    ch1 = next(i for i, t in enumerate(texts) if "CHAPTER 1" in t and "INTRODUCTION" in t)
    pages = {}

    def norm(s):
        return re.sub(r"\s+", " ", s)

    # front matter: pages 1..ch1-1 numbered i.. (title page index 0 has no number)
    for level, text, key in R.toc:
        if key.startswith("FM::"):
            name = key[4:]
            hits = [i for i in range(1, ch1) if texts[i].strip().upper().startswith(name)] or \
                   [i for i in range(1, ch1) if name in texts[i].upper()]
            if hits:
                pages[key] = roman(hits[0])
    last = ch1
    for level, text, key in R.toc:
        if key.startswith("FM::"):
            continue
        k = norm(key)
        for i in range(last, len(texts)):
            if k in texts[i]:
                pages[key] = str(i - ch1 + 1)
                last = i
                break
    for lst in (R.lot, R.lof):
        last = ch1
        for _, _, key in lst:
            for i in range(last, len(texts)):
                if key in texts[i]:
                    pages[key] = str(i - ch1 + 1)
                    last = i
                    break
    return pages


def to_pdf(docx):
    subprocess.run(["soffice", "--headless", "--convert-to", "pdf", "--outdir", os.path.dirname(docx), docx],
                   check=True, capture_output=True)


def main():
    R = build()
    fill_lists(R)
    R.save(OUT_DOCX)
    to_pdf(OUT_DOCX)
    pages = find_pages(R, OUT_PDF)
    R = build(pages)
    fill_lists(R)
    R.save(OUT_DOCX)
    to_pdf(OUT_DOCX)
    missing = [k for _, _, k in R.toc + R.lot + R.lof if k not in pages]
    print("pages found:", len(pages), "missing:", missing)
    print(OUT_DOCX)
    print(OUT_PDF)


if __name__ == "__main__":
    main()
