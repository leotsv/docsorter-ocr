#!/usr/bin/env python3
"""
=== Make Scanned PDFs Searchable ===

Recursively scans a directory for image-only PDFs and adds invisible
searchable text layers using OCR (tesseract). Already-searchable PDFs
are automatically skipped.

REQUIREMENTS:
  pip install pdfplumber pypdf pdf2image pytesseract

  Also install these system tools:
  - tesseract:  https://github.com/tesseract-ocr/tesseract
  - poppler (pdftoppm): https://poppler.freedesktop.org/

  On Windows (easiest):
    1. Install Tesseract: https://github.com/UB-Mannheim/tesseract/wiki
       (During install, note the install path, e.g. C:\\Program Files\\Tesseract-OCR)
    2. Install poppler: https://github.com/oschwartz10612/poppler-windows/releases
       (Extract and add the 'bin' folder to your PATH)
    3. pip install pdfplumber pypdf pdf2image pytesseract

USAGE:
  python make_pdfs_searchable.py "C:\\Users\\micro\\Documents\\Scans_Organized"

  Or for the original scans folder:
  python make_pdfs_searchable.py "\\\\TS1400DE70\\Share\\Backup\\Documents\\Scans"

OPTIONS:
  --dpi 200            OCR resolution (default: 200, lower=faster, higher=better quality)
  --workers 2          Parallel workers (default: 2, increase for faster processing)
  --dry-run            Just report which files need OCR without processing
  --force              Re-process even if file already has text
  --tesseract PATH     Path to tesseract executable
  --poppler-path PATH  Path to poppler bin directory (containing pdftoppm)

TEXT INJECTION (for manual classification):
  python make_pdfs_searchable.py --inject-text "Hannah Tsvayberg" "C:\\path\\to\\file.pdf"
  python make_pdfs_searchable.py --inject-text "Hannah Tsvayberg" file1.pdf file2.pdf

  Adds invisible text to the first page of the PDF so sort_scans.py can classify it.
  Requires: pip install reportlab  (or fpdf2)

TEXT REMOVAL:
  python make_pdfs_searchable.py --remove-text "Hannah Tsvayberg" "C:\\path\\to\\file.pdf"

  Removes all text blocks containing the given string from every page.
"""

import os
import sys
import io
import re
import argparse
import shutil
import subprocess
import tempfile
import time
import json
import traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# Try imports
try:
    import pdfplumber
except ImportError:
    print("ERROR: pdfplumber not installed. Run: pip install pdfplumber")
    sys.exit(1)

try:
    from pypdf import PdfReader, PdfWriter
except ImportError:
    print("ERROR: pypdf not installed. Run: pip install pypdf")
    sys.exit(1)

try:
    from pdf2image import convert_from_path
except ImportError:
    print("ERROR: pdf2image not installed. Run: pip install pdf2image")
    sys.exit(1)

try:
    import pytesseract
except ImportError:
    print("ERROR: pytesseract not installed. Run: pip install pytesseract")
    sys.exit(1)


# ─── Configuration ───────────────────────────────────────────────────────

STATE_FILENAME = ".ocr_progress.json"
MIN_TEXT_LENGTH = 50  # Characters below this = image-only PDF

# Global poppler path (set from command line args)
POPPLER_PATH = None


def find_tesseract():
    """Auto-detect tesseract location on Windows."""
    # Common Windows install locations
    common_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        r"C:\Users\{}\AppData\Local\Tesseract-OCR\tesseract.exe".format(os.getenv("USERNAME", "")),
        r"C:\Tools\Tesseract-OCR\tesseract.exe",
    ]
    for p in common_paths:
        if os.path.isfile(p):
            return p
    return None


def find_poppler():
    """Auto-detect poppler bin directory on Windows."""
    # Check common locations
    common_dirs = [
        r"C:\Program Files\poppler\Library\bin",
        r"C:\Program Files\poppler\bin",
        r"C:\Program Files (x86)\poppler\bin",
        r"C:\Tools\poppler\bin",
        r"C:\poppler\bin",
        r"C:\poppler\Library\bin",
    ]
    # Also search user's home directory
    home = os.path.expanduser("~")
    for subdir in ["poppler", "Downloads\\poppler", "Tools\\poppler"]:
        for bindir in ["bin", "Library\\bin"]:
            common_dirs.append(os.path.join(home, subdir, bindir))

    # Search PATH for pdftoppm
    for d in os.environ.get("PATH", "").split(os.pathsep):
        if os.path.isfile(os.path.join(d, "pdftoppm.exe")) or os.path.isfile(os.path.join(d, "pdftoppm")):
            return d

    for d in common_dirs:
        if os.path.isdir(d):
            if os.path.isfile(os.path.join(d, "pdftoppm.exe")) or os.path.isfile(os.path.join(d, "pdftoppm")):
                return d
    return None


def setup_tools(tesseract_path=None, poppler_path=None):
    """Configure tesseract and poppler paths. Returns True if both found."""
    global POPPLER_PATH

    # --- Tesseract ---
    if tesseract_path:
        if os.path.isfile(tesseract_path):
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
            print(f"  Tesseract: {tesseract_path}")
        else:
            print(f"ERROR: Tesseract not found at: {tesseract_path}")
            return False
    else:
        # Check if tesseract is in PATH
        try:
            result = subprocess.run(["tesseract", "--version"], capture_output=True, timeout=10)
            # Resolve full path so worker processes can find it too
            full_path = shutil.which("tesseract")
            if full_path:
                pytesseract.pytesseract.tesseract_cmd = full_path
                print(f"  Tesseract: {full_path}")
            else:
                print(f"  Tesseract: found in PATH")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            # Try auto-detect
            detected = find_tesseract()
            if detected:
                pytesseract.pytesseract.tesseract_cmd = detected
                print(f"  Tesseract: auto-detected at {detected}")
            else:
                print("ERROR: Tesseract not found!")
                print("  Install from: https://github.com/UB-Mannheim/tesseract/wiki")
                print("  Or specify: --tesseract \"C:\\Program Files\\Tesseract-OCR\\tesseract.exe\"")
                return False

    # --- Poppler ---
    if poppler_path:
        if os.path.isdir(poppler_path):
            POPPLER_PATH = poppler_path
            print(f"  Poppler:   {poppler_path}")
        else:
            print(f"ERROR: Poppler directory not found: {poppler_path}")
            return False
    else:
        # Check if pdftoppm is in PATH
        try:
            result = subprocess.run(["pdftoppm", "-v"], capture_output=True, timeout=10)
            # Resolve full path so worker processes can find it too
            full_path = shutil.which("pdftoppm")
            if full_path:
                POPPLER_PATH = os.path.dirname(full_path)
                print(f"  Poppler:   {POPPLER_PATH}")
            else:
                print(f"  Poppler:   found in PATH")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            detected = find_poppler()
            if detected:
                POPPLER_PATH = detected
                print(f"  Poppler:   auto-detected at {detected}")
            else:
                print("ERROR: Poppler (pdftoppm) not found!")
                print("  Download from: https://github.com/oschwartz10612/poppler-windows/releases")
                print("  Extract and specify: --poppler-path \"C:\\path\\to\\poppler\\Library\\bin\"")
                return False

    # Quick validation
    try:
        test_result = subprocess.run(
            [pytesseract.pytesseract.tesseract_cmd or "tesseract", "--version"],
            capture_output=True, timeout=10
        )
        tess_version = test_result.stdout.decode().split("\n")[0] if test_result.stdout else "unknown"
        print(f"  Tesseract version: {tess_version}")
    except Exception:
        pass

    return True


def is_searchable(pdf_path):
    """Check if a PDF already has extractable text."""
    try:
        with pdfplumber.open(long_path(pdf_path)) as pdf:
            text = ""
            for page in pdf.pages[:3]:  # Check first 3 pages
                t = page.extract_text()
                if t:
                    text += t
                if len(text) >= MIN_TEXT_LENGTH:
                    return True
        return len(text.strip()) >= MIN_TEXT_LENGTH
    except Exception:
        return False


def long_path(p):
    """Add Windows long path prefix to avoid 260-char limit."""
    if sys.platform == "win32" and not p.startswith("\\\\?\\"):
        p = os.path.abspath(p)
        if p.startswith("\\\\"):
            return "\\\\?\\UNC\\" + p[2:]
        return "\\\\?\\" + p
    return p


def get_page_count(pdf_path):
    """Get number of pages in a PDF."""
    try:
        reader = PdfReader(long_path(pdf_path))
        return len(reader.pages)
    except Exception:
        return 1


def make_searchable(pdf_path, dpi=200):
    """Add searchable text layer to an image-only PDF."""
    lp = long_path(pdf_path)
    npages = get_page_count(pdf_path)

    # Use a short temp directory to avoid path length issues
    with tempfile.TemporaryDirectory(prefix="ocr") as tmpdir:
        page_pdfs = []

        for pg in range(1, npages + 1):
            # Convert single page to image using pdf2image
            try:
                kwargs = dict(
                    first_page=pg,
                    last_page=pg,
                    dpi=dpi,
                    thread_count=1,
                )
                if POPPLER_PATH:
                    kwargs["poppler_path"] = POPPLER_PATH
                images = convert_from_path(lp, **kwargs)
                if not images:
                    raise RuntimeError(f"No image for page {pg}")

                # Save as TIFF
                tiff_path = os.path.join(tmpdir, f"page_{pg:04d}.tiff")
                images[0].save(tiff_path, "TIFF")
                del images  # Free memory

            except Exception as e:
                raise RuntimeError(f"pdf2image failed on page {pg} (poppler_path={POPPLER_PATH}): {e}")

            # Run tesseract to create PDF with text layer
            ocr_base = os.path.join(tmpdir, f"ocr_{pg:04d}")
            tess_cmd = pytesseract.pytesseract.tesseract_cmd or "tesseract"
            try:
                cmd = [tess_cmd, tiff_path, ocr_base, "-l", "eng", "pdf"]
                result = subprocess.run(
                    cmd,
                    capture_output=True, text=True, timeout=180
                )
                if result.returncode != 0:
                    raise RuntimeError(f"tesseract error (cmd={cmd}): {result.stderr}")
            except FileNotFoundError as e:
                raise RuntimeError(f"tesseract not found at '{tess_cmd}': {e}")
            except subprocess.TimeoutExpired:
                raise RuntimeError(f"tesseract timeout on page {pg}")

            ocr_pdf = ocr_base + ".pdf"
            if not os.path.exists(ocr_pdf) or os.path.getsize(ocr_pdf) == 0:
                raise RuntimeError(f"tesseract produced empty output for page {pg}")

            page_pdfs.append(ocr_pdf)

            # Remove TIFF to free disk space
            try:
                os.remove(tiff_path)
            except OSError:
                pass

        # Merge pages into final PDF
        output_path = os.path.join(tmpdir, "output.pdf")
        if len(page_pdfs) == 1:
            shutil.copy2(page_pdfs[0], output_path)
        else:
            writer = PdfWriter()
            for pp in page_pdfs:
                reader = PdfReader(pp)
                for page in reader.pages:
                    writer.add_page(page)
            with open(output_path, "wb") as f:
                writer.write(f)

        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise RuntimeError("Final merged PDF is empty")

        # Copy to a persistent temp file before the TemporaryDirectory is deleted
        persistent = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False, prefix="ocr_out_")
        persistent.close()
        shutil.copy2(output_path, persistent.name)
        return persistent.name


def inject_text_into_pdf(pdf_path, text):
    """Add invisible text to the first page of a PDF for classification purposes.

    Prepends text directly to the page content stream using standard Helvetica,
    so both pypdf and pdfplumber can extract it for classification.
    Text is rendered white (invisible) at 10pt in the bottom-left corner.
    Timestamps are preserved.
    """
    lp = long_path(pdf_path)

    # Preserve original timestamps
    orig_stat = os.stat(lp)
    orig_atime = orig_stat.st_atime
    orig_mtime = orig_stat.st_mtime

    # Read into memory so the file handle is closed before we overwrite
    with open(lp, "rb") as f:
        pdf_bytes = io.BytesIO(f.read())

    reader = PdfReader(pdf_bytes)
    writer = PdfWriter()

    # Add all pages to writer first, then inject into the writer's copy.
    # This ensures the content stream ends up as a proper indirect object
    # in the writer's object pool (direct DecodedStreamObject on the reader's
    # page may not serialize correctly through add_page cloning).
    for page in reader.pages:
        writer.add_page(page)

    _inject_text_into_writer_page(writer, 0, text)

    # Write to temp file, then replace original
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False, prefix="inject_")
    tmp.close()
    with open(tmp.name, "wb") as f:
        writer.write(f)

    shutil.copy2(tmp.name, lp)
    os.utime(lp, (orig_atime, orig_mtime))
    os.remove(tmp.name)


def remove_injection_from_pdf(pdf_path):
    """Remove any previously injected invisible text from the first page."""
    from pypdf.generic import NameObject, DecodedStreamObject
    lp = long_path(pdf_path)
    orig_stat = os.stat(lp)
    with open(lp, "rb") as f:
        pdf_bytes = io.BytesIO(f.read())
    reader = PdfReader(pdf_bytes)
    writer = PdfWriter()
    for page in reader.pages:
        writer.add_page(page)
    page = writer.pages[0]
    existing = _get_page_stream_bytes(page)
    stripped = re.sub(rb"q 1 1 1 rg BT [^\n]*/F_inj[^\n]*Tj ET Q\n?", b"", existing)
    if stripped != existing:
        new_stream = DecodedStreamObject()
        new_stream.set_data(stripped)
        new_ref = writer._add_object(new_stream)
        page[NameObject("/Contents")] = new_ref
        tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False, prefix="inject_")
        tmp.close()
        with open(tmp.name, "wb") as f:
            writer.write(f)
        shutil.copy2(tmp.name, lp)
        os.utime(lp, (orig_stat.st_atime, orig_stat.st_mtime))
        os.remove(tmp.name)


def _inject_text_into_writer_page(writer, page_index, text):
    """Prepend invisible white text to a writer's page content stream.

    Works on the writer's already-cloned page so the new stream is registered
    as a proper indirect object via writer._add_object().  Text is PREPENDED
    before the existing content so it runs at the default page CTM (identity
    for the content stream), preventing any cm transforms in the existing
    content from making the characters appear non-upright to pdfplumber.
    """
    from pypdf.generic import NameObject, DecodedStreamObject, DictionaryObject

    page = writer.pages[page_index]

    # Escape special PDF string characters
    escaped = (text
               .replace("\\", "\\\\")
               .replace("(", "\\(")
               .replace(")", "\\)")
               .replace("\r", "")
               .replace("\n", " "))

    # Choose Tm matrix so that pdfplumber considers the text 'upright' after
    # applying the page's /Rotate angle (viewer-applied CCW rotation).
    # /Rotate=0:   text direction (1,0) in raw → identity Tm
    # /Rotate=90:  text direction (0,1) in raw → Tm [0 1 -1 0]
    # /Rotate=180: text direction (-1,0) in raw → Tm [-1 0 0 -1]
    # /Rotate=270: text direction (0,-1) in raw → Tm [0 -1 1 0]
    rotate = int(page.get("/Rotate", 0) or 0) % 360
    tm_map = {
        0:   "1 0 0 1",
        90:  "0 1 -1 0",
        180: "-1 0 0 -1",
        270: "0 -1 1 0",
    }
    tm = tm_map.get(rotate, "1 0 0 1")

    # PDF content stream: save state, white fill, Tm at (10,10), restore
    cmd = (
        b"q 1 1 1 rg "
        + f"BT {tm} 10 10 Tm /F_inj 10 Tf ".encode("ascii")
        + ("(" + escaped + ")").encode("latin-1")
        + b" Tj ET Q\n"
    )

    # Collect existing content stream bytes from the writer's page,
    # stripping any previous /F_inj injections to prevent duplicate overlaps
    # (overlapping identical chars at the same position cause garbled extraction)
    existing = _get_page_stream_bytes(page)
    existing = re.sub(rb"q 1 1 1 rg BT [^\n]*/F_inj[^\n]*Tj ET Q\n?", b"", existing)

    # PREPEND our command so it runs at the default CTM (before any cm ops
    # in the existing content that could rotate/scale the coordinate system)
    new_stream = DecodedStreamObject()
    new_stream.set_data(cmd + b"\n" + existing)

    # Register as an indirect object so pypdf serializes it correctly
    new_ref = writer._add_object(new_stream)
    page[NameObject("/Contents")] = new_ref

    # Register /F_inj as standard Helvetica in the page font resources
    if "/Resources" not in page:
        page[NameObject("/Resources")] = DictionaryObject()
    resources = page["/Resources"]
    if hasattr(resources, "get_object"):
        resources = resources.get_object()
    if "/Font" not in resources:
        resources[NameObject("/Font")] = DictionaryObject()
    fonts = resources["/Font"]
    if hasattr(fonts, "get_object"):
        fonts = fonts.get_object()
    if "/F_inj" not in fonts:
        font_obj = DictionaryObject()
        font_obj[NameObject("/Type")] = NameObject("/Font")
        font_obj[NameObject("/Subtype")] = NameObject("/Type1")
        font_obj[NameObject("/BaseFont")] = NameObject("/Helvetica")
        fonts[NameObject("/F_inj")] = font_obj


def _decode_pdf_escapes(s):
    """Decode PDF string escape sequences: \\040 -> space, \\\\ -> \\, etc."""
    result = []
    i = 0
    while i < len(s):
        if s[i] == '\\' and i + 1 < len(s):
            nxt = s[i + 1]
            if nxt in '01234567':
                # Octal escape \NNN (1-3 digits)
                octal = ''
                j = i + 1
                while j < len(s) and j < i + 4 and s[j] in '01234567':
                    octal += s[j]
                    j += 1
                result.append(chr(int(octal, 8)))
                i = j
            elif nxt == 'n':
                result.append('\n'); i += 2
            elif nxt == 'r':
                result.append('\r'); i += 2
            elif nxt == 't':
                result.append('\t'); i += 2
            else:
                result.append(nxt); i += 2
        else:
            result.append(s[i])
            i += 1
    return ''.join(result)


def _get_page_stream_bytes(page):
    """Get decompressed content stream bytes from a PDF page."""
    contents = page.get("/Contents")
    if contents is None:
        return b""
    contents = contents.get_object()

    # Single stream
    if hasattr(contents, "get_data"):
        return contents.get_data()

    # Array of stream objects
    parts = []
    for item in contents:
        obj = item.get_object() if hasattr(item, "get_object") else item
        if hasattr(obj, "get_data"):
            parts.append(obj.get_data())
    return b"\n".join(parts)


def remove_text_from_pdf(pdf_path, text_to_remove):
    """Remove all BT/ET text blocks containing the given string from every page.

    This reverses --inject-text by stripping text blocks whose rendered
    characters include the target string (case-insensitive match).
    Timestamps are preserved.
    """
    from pypdf.generic import DecodedStreamObject, NameObject

    lp = long_path(pdf_path)

    # Preserve original timestamps
    orig_stat = os.stat(lp)
    orig_atime = orig_stat.st_atime
    orig_mtime = orig_stat.st_mtime

    reader = PdfReader(lp)
    writer = PdfWriter()
    target = text_to_remove.lower()
    removed_count = 0

    for page in reader.pages:
        raw = _get_page_stream_bytes(page)
        # Decode PDF escapes (\040 → space, etc.) before checking
        raw_text = raw.decode("latin-1", errors="replace")
        decoded_text = _decode_pdf_escapes(raw_text).lower()

        if target not in decoded_text:
            # Page doesn't contain the target — keep as-is
            writer.add_page(page)
            continue

        # Remove BT...ET blocks whose text operators contain the target
        def _check_block(m):
            nonlocal removed_count
            block_text = m.group(0).decode("latin-1", errors="replace")
            decoded_block = _decode_pdf_escapes(block_text).lower()
            if target in decoded_block:
                removed_count += 1
                return b""
            return m.group(0)

        modified = re.sub(rb"BT\b.*?ET\b", _check_block, raw, flags=re.DOTALL)

        # Replace the page's content stream with the modified data
        new_stream = DecodedStreamObject()
        new_stream.set_data(modified)
        page[NameObject("/Contents")] = new_stream
        writer.add_page(page)

    if removed_count == 0:
        return 0

    # Write to temp file, then replace original
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False, prefix="rmtext_")
    tmp.close()
    with open(tmp.name, "wb") as f:
        writer.write(f)

    shutil.copy2(tmp.name, lp)
    os.utime(lp, (orig_atime, orig_mtime))
    os.remove(tmp.name)
    return removed_count


def process_single_file(args):
    """Process a single PDF file. Returns (path, success, message, elapsed)."""
    pdf_path, dpi, force, tesseract_cmd, poppler_path = args
    t0 = time.time()

    # Restore tool paths in this worker process
    global POPPLER_PATH
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    if poppler_path:
        POPPLER_PATH = poppler_path

    try:
        # Skip if already searchable (unless --force)
        if not force and is_searchable(pdf_path):
            return (pdf_path, "skip", "already searchable", time.time() - t0)

        # Save original timestamps before any modification
        lp = long_path(pdf_path)
        orig_stat = os.stat(lp)
        orig_atime = orig_stat.st_atime
        orig_mtime = orig_stat.st_mtime

        # Create searchable version
        output_path = make_searchable(pdf_path, dpi=dpi)

        # Replace original with searchable version
        backup = lp + ".bak"
        shutil.copy2(lp, backup)  # Safety backup
        shutil.copy2(output_path, lp)
        # Restore original timestamps
        os.utime(lp, (orig_atime, orig_mtime))
        os.remove(backup)  # Remove backup on success
        # Clean up the persistent temp file
        try:
            os.remove(output_path)
        except OSError:
            pass

        return (pdf_path, "ok", f"{get_page_count(pdf_path)} pages", time.time() - t0)

    except Exception as e:
        # Restore from backup if it exists
        lp = long_path(pdf_path)
        backup = lp + ".bak"
        if os.path.exists(backup):
            shutil.copy2(backup, lp)
            os.remove(backup)
        # Include full traceback for first few errors to aid debugging
        tb = traceback.format_exc()
        return (pdf_path, "error", f"{e}\n    TRACEBACK: {tb.splitlines()[-3] if len(tb.splitlines()) >= 3 else tb}", time.time() - t0)


def find_pdfs(directory):
    """Recursively find all PDF files."""
    pdfs = []
    for root, dirs, files in os.walk(directory):
        for f in sorted(files):
            if f.lower().endswith(".pdf") and not f.startswith("."):
                pdfs.append(os.path.join(root, f))
    return pdfs


def load_state(directory):
    """Load progress state."""
    state_path = os.path.join(directory, STATE_FILENAME)
    if os.path.exists(state_path):
        with open(state_path) as f:
            return json.load(f)
    return {"completed": [], "skipped": [], "errors": []}


def save_state(directory, state):
    """Save progress state."""
    state_path = os.path.join(directory, STATE_FILENAME)
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Add searchable text layers to scanned PDFs"
    )
    parser.add_argument("source", nargs="+", help="Directory to scan recursively, or file(s) when using --inject-text")
    parser.add_argument("--dpi", type=int, default=200, help="OCR resolution (default: 200)")
    parser.add_argument("--workers", type=int, default=2, help="Parallel workers (default: 2)")
    parser.add_argument("--dry-run", action="store_true", help="Just report, don't process")
    parser.add_argument("--force", action="store_true", help="Re-process all files")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh, ignore previous progress")
    parser.add_argument("--tesseract", default=None, help="Path to tesseract executable")
    parser.add_argument("--poppler-path", default=None, help="Path to poppler bin directory")
    parser.add_argument("--inject-text", default=None, metavar="TEXT",
                        help="Inject invisible text into PDF(s) for classification (requires reportlab or fpdf2)")
    parser.add_argument("--remove-text", default=None, metavar="TEXT",
                        help="Remove all text blocks containing TEXT from PDF(s)")
    args = parser.parse_args()

    # ── Remove-text mode ─────────────────────────────────────────────────
    if args.remove_text:
        files = [os.path.abspath(s) for s in args.source]
        for f in files:
            if not os.path.isfile(f):
                print(f"ERROR: File not found: {f}")
                sys.exit(1)
            if not f.lower().endswith(".pdf"):
                print(f"ERROR: Not a PDF: {f}")
                sys.exit(1)

        print(f"Removing text: \"{args.remove_text}\"")
        print(f"Files: {len(files)}")
        print()
        total_removed = 0
        for f in files:
            rel = os.path.basename(f)
            try:
                n = remove_text_from_pdf(f, args.remove_text)
                if n > 0:
                    print(f"  OK  {rel} ({n} text block(s) removed)")
                    total_removed += n
                else:
                    print(f"  SKIP {rel} (text not found)")
            except Exception as e:
                print(f"  ERR {rel}: {e}")
        print(f"\nDone. Removed {total_removed} text block(s) total.")
        return

    # ── Inject-text mode ──────────────────────────────────────────────────
    if args.inject_text:
        files = [os.path.abspath(s) for s in args.source]
        for f in files:
            if not os.path.isfile(f):
                print(f"ERROR: File not found: {f}")
                sys.exit(1)
            if not f.lower().endswith(".pdf"):
                print(f"ERROR: Not a PDF: {f}")
                sys.exit(1)

        print(f"Injecting text: \"{args.inject_text}\"")
        print(f"Files: {len(files)}")
        print()
        for f in files:
            rel = os.path.basename(f)
            try:
                inject_text_into_pdf(f, args.inject_text)
                print(f"  OK  {rel}")
            except Exception as e:
                print(f"  ERR {rel}: {e}")
        print("\nDone.")
        return

    # ── Normal OCR mode ───────────────────────────────────────────────────
    if len(args.source) != 1 or not os.path.isdir(os.path.abspath(args.source[0])):
        print("ERROR: In OCR mode, provide exactly one directory as input.")
        print("  For file input, use --inject-text \"some text\" file1.pdf file2.pdf")
        sys.exit(1)

    directory = os.path.abspath(args.source[0])

    print(f"Scanning: {directory}")
    print(f"DPI: {args.dpi} | Workers: {args.workers}")
    print()

    # Setup and verify tesseract + poppler
    print("Checking tools...")
    if not setup_tools(tesseract_path=args.tesseract, poppler_path=args.poppler_path):
        print("\nPlease install the missing tools and try again.")
        sys.exit(1)
    print()

    # Find all PDFs
    all_pdfs = find_pdfs(directory)
    print(f"Found {len(all_pdfs)} PDF files")

    # Load previous state
    if args.no_resume:
        state = {"completed": [], "skipped": [], "errors": []}
    else:
        state = load_state(directory)
        done_set = set(state["completed"] + state["skipped"])
        if done_set:
            print(f"Resuming: {len(state['completed'])} completed, {len(state['skipped'])} skipped")

    done_set = set(state["completed"] + state["skipped"])
    remaining = [p for p in all_pdfs if p not in done_set]
    print(f"Remaining to check: {len(remaining)}")
    print()

    if args.dry_run:
        print("=== DRY RUN: Checking which files need OCR ===")
        need_ocr = 0
        already_ok = 0
        for i, pdf_path in enumerate(remaining):
            rel = os.path.relpath(pdf_path, directory)
            searchable = is_searchable(pdf_path)
            if searchable:
                already_ok += 1
            else:
                need_ocr += 1
                npages = get_page_count(pdf_path)
                size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
                print(f"  NEEDS OCR: {rel} ({npages} pages, {size_mb:.1f} MB)")
            if (i + 1) % 100 == 0:
                print(f"  ... checked {i+1}/{len(remaining)}")
        print(f"\nSummary: {need_ocr} need OCR, {already_ok} already searchable")
        return

    # Process files
    start_time = time.time()
    processed = 0
    ok_count = 0
    skip_count = 0
    err_count = 0

    # Get current tool paths for passing to workers
    tess_cmd = pytesseract.pytesseract.tesseract_cmd
    pop_path = POPPLER_PATH

    # Precompute per-directory totals for all remaining files
    dir_totals = {}
    for p in remaining:
        d = os.path.dirname(p)
        dir_totals[d] = dir_totals.get(d, 0) + 1
    dir_processed = {}

    # Sequential processing (safer for disk I/O heavy work)
    if args.workers <= 1:
        for pdf_path in remaining:
            rel = os.path.relpath(pdf_path, directory)
            idx = len(done_set) + processed + 1
            d = os.path.dirname(pdf_path)
            dir_processed[d] = dir_processed.get(d, 0) + 1
            dir_idx = dir_processed[d]
            dir_total = dir_totals[d]
            dir_name = os.path.relpath(d, directory)

            result = process_single_file((pdf_path, args.dpi, args.force, tess_cmd, pop_path))
            path, status, msg, elapsed = result

            counter = f"[{dir_name} {dir_idx}/{dir_total} | {idx}/{len(all_pdfs)}]"
            if status == "ok":
                print(f"{counter} OK  {rel} ({msg}, {elapsed:.1f}s)")
                state["completed"].append(path)
                ok_count += 1
            elif status == "skip":
                print(f"{counter} SKIP {rel} ({msg})")
                state["skipped"].append(path)
                skip_count += 1
            else:
                print(f"{counter} ERR {rel}: {msg} ({elapsed:.1f}s)")
                state["errors"].append({"file": path, "error": msg})
                err_count += 1

            processed += 1
            if processed % 25 == 0:
                save_state(directory, state)
                total_elapsed = time.time() - start_time
                rate = processed / total_elapsed if total_elapsed > 0 else 0
                eta = (len(remaining) - processed) / rate / 60 if rate > 0 else 0
                print(f"  --- Saved progress. Rate: {rate:.2f}/sec, ETA: {eta:.0f} min ---")
    else:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(process_single_file, (p, args.dpi, args.force, tess_cmd, pop_path)): p
                for p in remaining
            }
            for future in as_completed(futures):
                path, status, msg, elapsed = future.result()
                rel = os.path.relpath(path, directory)
                idx = len(done_set) + processed + 1
                d = os.path.dirname(path)
                dir_processed[d] = dir_processed.get(d, 0) + 1
                dir_idx = dir_processed[d]
                dir_total = dir_totals.get(d, 0)
                dir_name = os.path.relpath(d, directory)

                counter = f"[{dir_name} {dir_idx}/{dir_total} | {idx}/{len(all_pdfs)}]"
                if status == "ok":
                    print(f"{counter} OK  {rel} ({msg}, {elapsed:.1f}s)")
                    state["completed"].append(path)
                    ok_count += 1
                elif status == "skip":
                    state["skipped"].append(path)
                    skip_count += 1
                else:
                    print(f"{counter} ERR {rel}: {msg}")
                    state["errors"].append({"file": path, "error": msg})
                    err_count += 1

                processed += 1
                if processed % 25 == 0:
                    save_state(directory, state)

    save_state(directory, state)
    total_elapsed = time.time() - start_time

    print(f"\n{'='*60}")
    print(f"DONE in {total_elapsed/60:.1f} minutes")
    print(f"  Converted: {ok_count}")
    print(f"  Skipped (already searchable): {skip_count}")
    print(f"  Errors: {err_count}")
    print(f"  Progress saved to: {os.path.join(directory, STATE_FILENAME)}")
    if err_count > 0:
        print(f"\nFiles with errors:")
        for e in state["errors"][-10:]:
            print(f"  {os.path.relpath(e['file'], directory)}: {e['error']}")


if __name__ == "__main__":
    main()
