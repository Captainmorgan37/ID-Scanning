# Passport MRZ Scanner – Local Setup Notes

This Streamlit application reads passport Machine Readable Zone (MRZ) text using
PaddleOCR and matches it against a PDF crew/passenger manifest.  Only the
packages listed in `requirements.txt` are required to run the app locally:

```bash
pip install -r requirements.txt
```

The PDF manifest parsing is implemented with [`PyPDF2`](https://pypdf2.readthedocs.io/)
which ships pure Python wheels.  You **do not** need `PyMuPDF`/`fitz` for this
project.  Installing an old `pymupdf==1.20.2` release forces pip to build from
source, which can take several minutes and often fails on constrained
environments.  If you previously tried to install that package you can remove it
with:

```bash
pip uninstall pymupdf
```

If you have another workflow that genuinely requires PyMuPDF, prefer a recent
release (`pip install "pymupdf>=1.21"`) so pip can download a prebuilt wheel
instead of compiling from source.
