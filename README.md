# docsorter-ocr

The **OCR / searchable-PDF** responsibility of
[DocumentSorter](https://github.com/leotsv/DocumentSorter), split out on
2026-09-10 when that repo passed the house 10,000-line cap. Everything
here is about turning image-only PDFs into searchable ones and editing
the invisible text layer — nothing about rules, filing, the Flask app or
the audit trail.

Vendored as a git submodule, never copied (house rule, see
[house-lib](https://github.com/leotsv/house-lib)): a copied helper drifts.

```bash
git submodule add https://github.com/leotsv/docsorter-ocr.git ocr
```

Then, in `app.py`:

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "ocr"))
import make_pdfs_searchable
```

The directory is put on `sys.path` rather than imported as a package so
the module keeps the name it has always had — `make_pdfs_searchable` —
and every existing caller, docstring and `--inject-text` command line
still reads the same.

## What is in it

| module | what it does |
|---|---|
| `make_pdfs_searchable.py` | recursive OCR pass over a folder of PDFs (`make_searchable`, `process_single_file`, resumable through `.ocr_progress.json`), plus the invisible-text-layer editors `inject_text_into_pdf` / `remove_injection_from_pdf` / `remove_text_from_pdf`, and the tesseract/poppler discovery used by both (`setup_tools`, `find_tesseract`, `find_poppler`) |

It is the program's ONE copy of the PDF/OCR toolchain (house duplication
tollgate, 2026-09-25): DocumentSorter's `sort_scans` imports
`pdfplumber`/`PdfReader`/`convert_from_path`/`pytesseract`,
`get_page_count` and `long_path` from here, and its OCR embed calls
`make_searchable(path, dpi, poppler_path=...)` — the `poppler_path`
argument lets a caller that owns its own tool setup pass its poppler
directory instead of relying on this module's `POPPLER_PATH`.

It is also a CLI in its own right:

```bash
python make_pdfs_searchable.py "C:\Users\micro\Documents\Scans_Organized"
python make_pdfs_searchable.py --inject-text "Hannah Tsvayberg" file.pdf
python make_pdfs_searchable.py --remove-text  "Hannah Tsvayberg" file.pdf
```

## Requirements

`pip install -r requirements.txt`, plus the two system tools the OCR pass
shells out to: **tesseract** and **poppler** (`pdftoppm`). `setup_tools()`
finds both on Windows without configuration; DocumentSorter passes the
paths from its Settings page when the owner has set them.

## Tests

```bash
python -m pytest -q
```

`test_make_pdfs.py` builds small real PDFs with reportlab — no OCR and no
network are needed to run it. It is collected both here and by
DocumentSorter's own suite, which runs it from the submodule checkout.
