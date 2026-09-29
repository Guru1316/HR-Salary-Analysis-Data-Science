import re, sys, nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell
from nbconvert.preprocessors import ExecutePreprocessor
src = open("source.py").read()
parts = re.split(r"^# %%(?: \[markdown\])?\n", src, flags=re.M)[1:]
kinds = re.findall(r"^# %%( \[markdown\])?\n", src, flags=re.M)
nb = new_notebook()
for k, p in zip(kinds, parts):
    if k: nb.cells.append(new_markdown_cell("\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in p.strip().splitlines())))
    else: nb.cells.append(new_code_cell(p.strip()))
extra = "insights.md"
import os
if os.path.exists(extra): nb.cells.append(new_markdown_cell(open(extra).read()))
ExecutePreprocessor(timeout=300, kernel_name="python3").preprocess(nb, {"metadata": {"path": "."}})
nbformat.write(nb, "hr_salary_analysis.ipynb")
# dump text outputs for review
for i, c in enumerate(nb.cells):
    if c.cell_type == "code":
        for o in c.outputs:
            if o.output_type == "stream": print(o.text, end="")
            elif o.output_type == "error": print("ERROR", o.ename, o.evalue)
