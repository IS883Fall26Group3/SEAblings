"""The results page as a PDF, for the "Download report (PDF)" button.

Same content as the page: what we read from the CV, then each job's score, skills, score breakdown
(core/score_notes.py) and advice, if the user asked for it. Uses fpdf2's built-in Helvetica, which
only has Latin-1 characters, so text is mapped to those first (the app is English-only)."""
import re
from datetime import date

from fpdf import FPDF, FontFace

from core import score_notes as notes
from core.schemas import CandidateProfile, MatchResult
from ui.components import DISCLAIMER

DEEP = (14, 94, 111)    # var(--deep) in ui/styles.css
GREY = (90, 100, 105)
ROW_H = 5.5

_LATIN1 = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-", "•": "-",
                         "●": "-", "▪": "-", "…": "...", "→": "->", "≥": ">=", "≤": "<=", "×": "x",
                         " ": " ", "​": ""})


def latin1(text: str) -> str:
    """Text Helvetica can draw: common punctuation mapped, anything else shown as '?'."""
    return str(text).translate(_LATIN1).encode("latin-1", "replace").decode("latin-1")


class Report(FPDF):
    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", size=8)
        self.set_text_color(*GREY)
        self.cell(0, 5, f"SEAblings AI CV-to-job matcher  |  page {self.page_no()}", align="C")

    def text(self, s: str, size: float = 10, bold: bool = False, color=(0, 0, 0), indent: float = 0):
        self.set_font("Helvetica", "B" if bold else "", size)
        self.set_text_color(*color)
        self.set_x(self.l_margin + indent)
        self.multi_cell(0, ROW_H if size <= 10 else size * 0.55, latin1(s), align="L",
                        new_x="LMARGIN", new_y="NEXT")

    def labeled(self, label: str, value: str, indent: float = 0):
        """'Label  value' on one line, the label in bold."""
        if not value:
            return
        self.set_x(self.l_margin + indent)
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(0, 0, 0)
        self.write(ROW_H, latin1(label) + "  ")
        self.set_font("Helvetica", "", 10)
        self.write(ROW_H, latin1(value))
        self.ln(ROW_H)

    def bullet(self, s: str, indent: float = 4, mark: str = "-"):
        self.text(f"{mark} {s}", indent=indent)

    def gap(self, h: float = 3):
        self.ln(h)


def _note(pdf: Report, name: str, value: float, note: notes.Note):
    pdf.text(f"{name}: {value:.0%}. {note.label}", bold=True)
    pdf.text(note.text)
    for b in note.bullets:
        pdf.bullet(b)
    for g in note.groups:
        if g.inline:
            pdf.labeled(g.title, ", ".join(g.items))
        else:
            pdf.text(g.title, bold=True)
            for item in g.items:
                pdf.bullet(item)
    if note.table:
        head = ("", "Your CV", "This job") + (("Match",) if len(note.table[0]) > 3 else ())
        pdf.set_font("Helvetica", size=9)
        pdf.set_text_color(0, 0, 0)
        with pdf.table(col_widths=(22, 38, 38, 16)[:len(head)], line_height=5,
                       headings_style=FontFace(emphasis="BOLD")) as t:
            for row in (head, *note.table):
                cells = t.row()
                for c in row:
                    cells.cell(latin1(c))
    if note.after:
        pdf.text(note.after)
    pdf.gap(2)


def _advice(pdf: Report, markdown: str):
    """The advice as the page shows it: headings in bold, bullets indented, ** markers dropped."""
    for line in markdown.splitlines():
        line = line.rstrip()
        if not line.strip():
            pdf.gap(1.5)
            continue
        plain = re.sub(r"\*\*|__|`", "", line)
        if m := re.match(r"\s*#+\s*(.*)", plain):
            pdf.text(m.group(1), bold=True)
        elif re.fullmatch(r"\s*\*\*[^*]+\*\*:?\s*", line):  # a line all in bold is a heading too
            pdf.text(plain.strip(), bold=True)
        elif m := re.match(r"(\s*)([-*+]|\d+[.)])\s+(.*)", plain):
            mark = m.group(2) if m.group(2)[0].isdigit() else "-"
            pdf.bullet(m.group(3), indent=4 + 2 * len(m.group(1)), mark=mark)
        else:
            pdf.text(plain.strip())


def build_pdf(results: list[MatchResult], profile: CandidateProfile, *, summary: str, models: str,
              embedder: str, explanations: dict[str, str] | None = None) -> bytes:
    explanations = explanations or {}
    pdf = Report(format="A4")
    pdf.set_auto_page_break(True, margin=16)
    pdf.set_margins(16, 16, 16)
    pdf.add_page()

    pdf.text("Your CV-to-job match report", size=18, bold=True, color=DEEP)
    pdf.text(f"{summary}. Analyzed with {models}; similarity by {embedder}. "
             f"Created {date.today():%B %d, %Y}.", size=9, color=GREY)
    pdf.gap(4)

    pdf.text("What we read from your CV", size=13, bold=True, color=DEEP)
    pdf.labeled("Experience", f"{profile.years_experience:g} years")
    pdf.labeled("Level", profile.seniority.title())
    pdf.text(profile.summary)
    pdf.labeled("Skills", ", ".join(profile.skills))
    pdf.gap(4)

    for i, r in enumerate(results, 1):
        job = r.job
        if pdf.get_y() > pdf.h - 70:  # start a job near the bottom on a new page
            pdf.add_page()
        pdf.set_draw_color(*DEEP)
        pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
        pdf.gap(2)
        pdf.text(f"{i}. {job.title}  -  {r.score:.0f}/100", size=13, bold=True, color=DEEP)
        meta = ", ".join(x for x in (job.company, job.location, job.employment_type.title()) if x)
        if meta or job.source:
            pdf.text(meta + (f" (via {job.source})" if job.source else ""), size=9, color=GREY)
        pdf.gap(1)
        pdf.labeled("You have", ", ".join(r.matched_skills))
        pdf.labeled("Partly", ", ".join(r.partial_skills))
        pdf.labeled("To build", ", ".join(r.missing_skills))
        pdf.gap(2)

        _note(pdf, "Skill coverage", r.skill_coverage, notes.skill_note(r, profile))
        _note(pdf, "Profile similarity", r.semantic_similarity, notes.similarity_note(r, profile))
        _note(pdf, "Experience fit", r.seniority_fit, notes.experience_note(r))

        if advice := explanations.get(job.id):
            pdf.text("Advice for this role", size=11, bold=True, color=DEEP)
            _advice(pdf, advice)
        if job.url:
            pdf.text(f"Job posting: {job.url}", size=9, color=GREY)
        pdf.gap(5)

    pdf.text("The score is based on your CV text only. AI advice can be wrong: check it before you "
             "change your CV. " + DISCLAIMER, size=8, color=GREY)
    return bytes(pdf.output())
