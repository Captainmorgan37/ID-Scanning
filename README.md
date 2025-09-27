# Passport MRZ Scanner – Local Setup Notes

This Streamlit application reads passport Machine Readable Zone (MRZ) text using
RapidOCR (ONNXRuntime) and matches it against a PDF crew/passenger manifest.  If
RapidOCR cannot be installed on your platform the app now falls back to
[EasyOCR](https://github.com/JaidedAI/EasyOCR), so installing the packages
listed in `requirements.txt` is sufficient to run the app locally:

```bash
pip install -r requirements.txt
```

The PDF manifest parsing is implemented with [`PyPDF2`](https://pypdf2.readthedocs.io/)
which ships pure Python wheels.  You **do not** need `PyMuPDF`/`fitz` for this
project.  Earlier versions of this app depended on PaddleOCR, whose packaging
pulled in `pymupdf==1.20.2` and required compiling from source.  Switching to
RapidOCR removes that fragile dependency.  If you previously tried to install
the old package you can remove it with:

```bash
pip uninstall pymupdf
```

If you have another workflow that genuinely requires PyMuPDF, prefer a recent
release (`pip install "pymupdf>=1.21"`) so pip can download a prebuilt wheel
instead of compiling from source.
