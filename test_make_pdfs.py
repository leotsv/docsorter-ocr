"""Unit tests for make_pdfs_searchable.py.

Pure helpers are tested directly. inject/remove text round-trips use small
real PDFs built with reportlab (no OCR, no network). setup_tools is exercised
with monkeypatched subprocess/which so no real tesseract/poppler is required.
"""

import json
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(__file__))

import make_pdfs_searchable as mps


# ── tiny PDF factory ──────────────────────────────────────────────────────────

def _make_real_pdf(path, body_text="Hello world"):
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    c = canvas.Canvas(str(path), pagesize=letter)
    c.drawString(72, 720, body_text)
    c.showPage()
    c.save()
    return str(path)


# ── pure helpers ──────────────────────────────────────────────────────────────

def test_long_path_idempotent():
    once = mps.long_path("a.pdf")
    assert mps.long_path(once) == once


def test_get_page_count_real_pdf(tmp_path):
    p = _make_real_pdf(tmp_path / "one.pdf")
    assert mps.get_page_count(p) == 1


def test_get_page_count_bad_returns_one(tmp_path):
    assert mps.get_page_count(str(tmp_path / "missing.pdf")) == 1


def test_is_searchable_true_for_text_pdf(tmp_path):
    p = _make_real_pdf(tmp_path / "txt.pdf",
                       body_text="The quick brown fox jumps over the lazy dog " * 3)
    assert mps.is_searchable(p) is True


def test_is_searchable_false_for_missing(tmp_path):
    assert mps.is_searchable(str(tmp_path / "nope.pdf")) is False


def test_find_pdfs_recursive_and_filters(tmp_path):
    (tmp_path / "a.pdf").write_bytes(b"%PDF")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "b.pdf").write_bytes(b"%PDF")
    (tmp_path / ".hidden.pdf").write_bytes(b"%PDF")   # skipped
    (tmp_path / "c.txt").write_text("x")              # skipped
    found = mps.find_pdfs(str(tmp_path))
    names = sorted(os.path.basename(f) for f in found)
    assert names == ["a.pdf", "b.pdf"]


def test_load_state_default_when_absent(tmp_path):
    state = mps.load_state(str(tmp_path))
    assert state == {"completed": [], "skipped": [], "errors": []}


def test_save_and_load_state_roundtrip(tmp_path):
    state = {"completed": ["x.pdf"], "skipped": [], "errors": ["y.pdf"]}
    mps.save_state(str(tmp_path), state)
    assert os.path.isfile(os.path.join(str(tmp_path), mps.STATE_FILENAME))
    assert mps.load_state(str(tmp_path)) == state


# ── _decode_pdf_escapes ───────────────────────────────────────────────────────

def test_decode_pdf_escapes_octal():
    assert mps._decode_pdf_escapes(r"a\040b") == "a b"          # \040 == space


def test_decode_pdf_escapes_named():
    assert mps._decode_pdf_escapes(r"a\nb\tc") == "a\nb\tc"


def test_decode_pdf_escapes_backslash_and_parens():
    assert mps._decode_pdf_escapes(r"a\\b\(c\)") == r"a\b(c)"


def test_decode_pdf_escapes_plain():
    assert mps._decode_pdf_escapes("plain text") == "plain text"


# ── find_tesseract / find_poppler (no real installs assumed) ──────────────────

def test_find_tesseract_returns_none_when_absent(monkeypatch):
    monkeypatch.setattr(mps.os.path, "isfile", lambda p: False)
    assert mps.find_tesseract() is None


def test_find_poppler_returns_none_when_absent(monkeypatch):
    monkeypatch.setattr(mps.os.path, "isfile", lambda p: False)
    monkeypatch.setattr(mps.os.path, "isdir", lambda p: False)
    monkeypatch.setenv("PATH", "")
    assert mps.find_poppler() is None


# ── setup_tools ───────────────────────────────────────────────────────────────

def test_setup_tools_explicit_paths_ok(tmp_path, monkeypatch):
    tess = tmp_path / "tesseract.exe"
    tess.write_bytes(b"x")
    popp = tmp_path / "poppler_bin"
    popp.mkdir()
    # Avoid touching the real pytesseract module global
    monkeypatch.setattr(mps, "pytesseract",
                        types.SimpleNamespace(pytesseract=types.SimpleNamespace(tesseract_cmd=str(tess))))
    # Validation subprocess.run -> harmless fake
    monkeypatch.setattr(mps.subprocess, "run",
                        lambda *a, **k: types.SimpleNamespace(stdout=b"tesseract 5.0", returncode=0))
    ok = mps.setup_tools(tesseract_path=str(tess), poppler_path=str(popp))
    assert ok is True
    assert mps.POPPLER_PATH == str(popp)


def test_setup_tools_bad_tesseract_path(tmp_path):
    assert mps.setup_tools(tesseract_path=str(tmp_path / "missing.exe")) is False


def test_setup_tools_bad_poppler_path(tmp_path, monkeypatch):
    tess = tmp_path / "tesseract.exe"
    tess.write_bytes(b"x")
    monkeypatch.setattr(mps, "pytesseract",
                        types.SimpleNamespace(pytesseract=types.SimpleNamespace(tesseract_cmd=str(tess))))
    assert mps.setup_tools(tesseract_path=str(tess),
                           poppler_path=str(tmp_path / "no_such_dir")) is False


# ── inject / remove text round-trip on a real PDF ─────────────────────────────

def test_inject_text_roundtrip_preserves_timestamps(tmp_path):
    p = _make_real_pdf(tmp_path / "doc.pdf", "Original body text here for layout.")
    before = os.stat(p)

    mps.inject_text_into_pdf(p, "INJECTED_CLASSIFY_TOKEN")

    # Timestamp preserved (inject explicitly restores mtime)
    after = os.stat(p)
    assert abs(after.st_mtime - before.st_mtime) < 2

    # The injected token is recoverable from the content stream
    from pypdf import PdfReader
    page = PdfReader(p).pages[0]
    raw = mps._get_page_stream_bytes(page)
    assert b"/F_inj" in raw
    assert b"INJECTED_CLASSIFY_TOKEN" in raw


def test_remove_injection_strips_token(tmp_path):
    p = _make_real_pdf(tmp_path / "doc2.pdf", "Body for removal test.")
    mps.inject_text_into_pdf(p, "TOKEN_TO_REMOVE")

    from pypdf import PdfReader
    raw_before = mps._get_page_stream_bytes(PdfReader(p).pages[0])
    assert b"TOKEN_TO_REMOVE" in raw_before

    mps.remove_injection_from_pdf(p)

    raw_after = mps._get_page_stream_bytes(PdfReader(p).pages[0])
    assert b"/F_inj" not in raw_after
    assert b"TOKEN_TO_REMOVE" not in raw_after


def test_inject_escapes_special_chars(tmp_path):
    p = _make_real_pdf(tmp_path / "doc3.pdf", "Body.")
    # Parentheses and backslashes must be escaped in the PDF string
    mps.inject_text_into_pdf(p, r"a(b)c\d")
    from pypdf import PdfReader
    raw = mps._get_page_stream_bytes(PdfReader(p).pages[0])
    assert b"/F_inj" in raw


# ── remove_text_from_pdf ──────────────────────────────────────────────────────

def test_remove_text_from_pdf_strips_injected_block(tmp_path):
    p = _make_real_pdf(tmp_path / "rm.pdf", "Body content here.")
    mps.inject_text_into_pdf(p, "REMOVE_ME_TOKEN")

    count = mps.remove_text_from_pdf(p, "REMOVE_ME_TOKEN")
    assert count >= 1
    from pypdf import PdfReader
    raw = mps._get_page_stream_bytes(PdfReader(p).pages[0])
    assert b"REMOVE_ME_TOKEN" not in raw


def test_remove_text_from_pdf_no_match_returns_zero(tmp_path):
    p = _make_real_pdf(tmp_path / "rm2.pdf", "Just some body text.")
    assert mps.remove_text_from_pdf(p, "NONEXISTENT_STRING_XYZ") == 0


def test_get_page_stream_bytes_empty_when_no_contents():
    # A bare page object with no /Contents yields empty bytes
    fake_page = {"x": 1}
    assert mps._get_page_stream_bytes(fake_page) == b""


# ── process_single_file ───────────────────────────────────────────────────────

def test_process_single_file_skips_already_searchable(tmp_path):
    p = _make_real_pdf(tmp_path / "txt.pdf",
                       body_text="The quick brown fox jumps over the lazy dog. " * 3)
    result = mps.process_single_file((p, 200, False, None, None))
    path, status, msg, elapsed = result
    assert status == "skip"
    assert "already searchable" in msg


def test_process_single_file_ocr_path_mocked(tmp_path, monkeypatch):
    # Force the non-searchable branch and stub make_searchable so no OCR runs.
    p = _make_real_pdf(tmp_path / "img.pdf", "x")
    monkeypatch.setattr(mps, "is_searchable", lambda path: False)

    searchable = tmp_path / "searchable.pdf"
    _make_real_pdf(searchable, "now searchable content here for the document")
    monkeypatch.setattr(mps, "make_searchable", lambda path, dpi=200: str(searchable))

    result = mps.process_single_file((p, 200, True, None, None))
    path, status, msg, elapsed = result
    assert status == "ok"
    # Backup file must be cleaned up
    assert not os.path.exists(mps.long_path(p) + ".bak")


# ── main() inject / remove modes ──────────────────────────────────────────────

def test_main_inject_mode(tmp_path, monkeypatch, capsys):
    p = _make_real_pdf(tmp_path / "doc.pdf", "Body text here.")
    monkeypatch.setattr(sys, "argv",
                        ["make_pdfs_searchable.py", p, "--inject-text", "MAIN_INJECT"])
    mps.main()
    assert "Done." in capsys.readouterr().out
    from pypdf import PdfReader
    raw = mps._get_page_stream_bytes(PdfReader(p).pages[0])
    assert b"MAIN_INJECT" in raw


def test_main_remove_mode(tmp_path, monkeypatch, capsys):
    p = _make_real_pdf(tmp_path / "doc.pdf", "Body.")
    mps.inject_text_into_pdf(p, "REMOVE_VIA_MAIN")
    monkeypatch.setattr(sys, "argv",
                        ["make_pdfs_searchable.py", p, "--remove-text", "REMOVE_VIA_MAIN"])
    mps.main()
    out = capsys.readouterr().out
    assert "removed" in out.lower()
    from pypdf import PdfReader
    raw = mps._get_page_stream_bytes(PdfReader(p).pages[0])
    assert b"REMOVE_VIA_MAIN" not in raw


def test_main_inject_missing_file_exits(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv",
                        ["make_pdfs_searchable.py", str(tmp_path / "ghost.pdf"),
                         "--inject-text", "x"])
    with pytest.raises(SystemExit):
        mps.main()


def test_main_ocr_mode_requires_directory(tmp_path, monkeypatch):
    f = _make_real_pdf(tmp_path / "f.pdf", "x")
    monkeypatch.setattr(sys, "argv", ["make_pdfs_searchable.py", f])
    with pytest.raises(SystemExit):
        mps.main()


def test_process_single_file_error_restores_backup(tmp_path, monkeypatch):
    p = _make_real_pdf(tmp_path / "err.pdf", "original content")
    original = open(p, "rb").read()
    monkeypatch.setattr(mps, "is_searchable", lambda path: False)
    monkeypatch.setattr(mps, "make_searchable",
                        lambda path, dpi=200: (_ for _ in ()).throw(RuntimeError("ocr boom")))
    result = mps.process_single_file((p, 200, True, None, None))
    path, status, msg, elapsed = result
    assert status == "error"
    # File restored to original (or left intact); backup removed
    assert not os.path.exists(mps.long_path(p) + ".bak")
    assert open(p, "rb").read() == original


# ── extracted helpers (dup tollgate 2026-09-25) ───────────────────────────────

@pytest.mark.parametrize("caller, expected", [
    (r"C:\caller\poppler", r"C:\caller\poppler"),   # caller's path wins
    (None, r"C:\global\poppler"),                    # else the module's
])
def test_make_searchable_poppler_path(tmp_path, monkeypatch, caller, expected):
    """poppler_path= overrides the module global, so DocumentSorter's
    sort_scans (which owns its own POPPLER_PATH) reuses this pipeline."""
    p = _make_real_pdf(tmp_path / "scan.pdf")
    seen = {}

    def fake_convert(path, **kw):
        seen.update(kw)
        raise RuntimeError("stop")
    monkeypatch.setattr(mps, "convert_from_path", fake_convert)
    monkeypatch.setattr(mps, "POPPLER_PATH", r"C:\global\poppler")
    with pytest.raises(RuntimeError, match="pdf2image failed"):
        mps.make_searchable(p, dpi=150, poppler_path=caller)
    assert seen["poppler_path"] == expected
    assert seen["dpi"] == 150


def test_load_into_writer_and_replace_with_keep_pages_and_times(tmp_path):
    p = _make_real_pdf(tmp_path / "one.pdf")
    os.utime(p, (1_000_000_000, 1_000_000_000))
    st = os.stat(p)
    writer = mps._load_into_writer(p)
    assert len(writer.pages) == 1
    mps._replace_with(writer, p, st, "t_")
    assert os.stat(p).st_mtime == pytest.approx(1_000_000_000)
    assert mps.get_page_count(p) == 1


def test_pdf_files_or_exit(tmp_path):
    p = _make_real_pdf(tmp_path / "a.pdf")
    assert mps._pdf_files_or_exit([p]) == [os.path.abspath(p)]
    txt = tmp_path / "a.txt"
    txt.write_text("x")
    with pytest.raises(SystemExit):
        mps._pdf_files_or_exit([str(txt)])
    with pytest.raises(SystemExit):
        mps._pdf_files_or_exit([str(tmp_path / "missing.pdf")])


def test_record_result_files_each_status(capsys):
    state = {"completed": [], "skipped": [], "errors": []}
    counts = {"ok": 0, "skip": 0, "err": 0}
    mps._record_result(state, counts, "[c]", "a.pdf", ("A", "ok", "done", 1.0), True)
    mps._record_result(state, counts, "[c]", "b.pdf", ("B", "skip", "has text", 0.1), False)
    mps._record_result(state, counts, "[c]", "c.pdf", ("C", "error", "boom", 2.0), True)
    out = capsys.readouterr().out
    assert counts == {"ok": 1, "skip": 1, "err": 1}
    assert state["completed"] == ["A"] and state["skipped"] == ["B"]
    assert state["errors"] == [{"file": "C", "error": "boom"}]
    assert "SKIP" not in out            # parallel mode is quiet on skips
    assert "ERR c.pdf: boom (2.0s)" in out
