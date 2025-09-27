# Passport MRZ Scanner — Local OCR + PDF Manifest Matching
# --------------------------------------------------------
# What this does
# - No paid AI: OpenCV + RapidOCR (ONNXRuntime) for OCR, deterministic MRZ parsing with check‑digit validation
# - Accepts camera or multi-upload of passport photos
# - Accepts a PDF manifest (e.g., eAPIS / crew & pax list) and extracts Name, Passport Number, Expiry
# - Matches each scanned passport to the manifest by Passport # (primary) and Name (secondary), then verifies expiry
# - Shows a results table: Matched / Mismatch / Not Found with reasons

import streamlit as st
from PIL import Image
import numpy as np
try:
    import cv2
except ImportError:  # pragma: no cover - optional dependency
    cv2 = None

from datetime import datetime

try:
    from rapidocr_onnxruntime import RapidOCR  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    RapidOCR = None  # type: ignore[assignment]

try:
    import easyocr  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    easyocr = None
import re
import PyPDF2
st.set_page_config(page_title="Passport MRZ Scanner + Manifest Match", page_icon="🛂", layout="wide")

# =============================
# MRZ helpers (parse & validate)
# =============================

def compute_check_digit(data: str) -> str:
    weights = [7, 3, 1]
    total = 0
    for i, ch in enumerate(data):
        if ch.isdigit():
            val = int(ch)
        elif ch == '<':
            val = 0
        else:
            val = ord(ch) - 55  # A=10 ... Z=35
        total += val * weights[i % 3]
    return str(total % 10)


def validate(field: str, check_digit: str) -> bool:
    return bool(field and check_digit and compute_check_digit(field) == check_digit)


def expand_date_yyMMdd(raw6: str, kind: str) -> str:
    """Expand YYMMDD to YYYY-MM-DD.
    kind: 'dob' -> assume 19YY; 'exp' -> assume 20YY (simple heuristic)."""
    if len(raw6) != 6 or not raw6.isdigit():
        return ""
    yy, mm, dd = raw6[:2], raw6[2:4], raw6[4:6]
    if not ("01" <= mm <= "12") or not ("01" <= dd <= "31"):
        return ""
    century = "19" if kind == "dob" else "20"
    return f"{century}{yy}-{mm}-{dd}"


def smart_digit_swap_to_pass(field_raw: str, check_digit: str) -> str:
    """Attempt small digit substitutions to satisfy MRZ check digit (camera misreads like 3↔8, 5↔6, 0↔8, 1↔7)."""
    if not field_raw or not check_digit:
        return ""
    swaps = [("3","8"),("5","6"),("0","8"),("1","7"),("2","7"),("9","8")]
    seen = {field_raw}
    def try_variants(s):
        if compute_check_digit(s) == check_digit:
            return s
        for a,b in swaps:
            for repl in [(a,b),(b,a)]:
                t = s.replace(repl[0], repl[1], 1)
                if t not in seen:
                    seen.add(t)
                    if compute_check_digit(t) == check_digit:
                        return t
        return ""
    return try_variants(field_raw)


def parse_mrz_lines(line1: str, line2: str):
    # Normalize to 44 chars
    line1 = (line1 + '<'*44)[:44]
    line2 = (line2 + '<'*44)[:44]

    # Line 1
    doc_type = line1[0:2]
    issuing_country = line1[2:5]
    names_raw = line1[5:44]
    parts = names_raw.split('<<')
    surname = parts[0].replace('<', ' ').strip()
    given_names = ' '.join(p.replace('<',' ').strip() for p in parts[1:] if p)

    # Line 2 — fixed slices
    passport_number_field = line2[0:9]
    passport_number = passport_number_field.replace('<','')
    passport_check = line2[9]
    nationality = line2[10:13]
    dob_raw = line2[13:19]
    dob_check = line2[19]
    sex = line2[20]
    expiry_raw = line2[21:27]
    expiry_check = line2[27]

    # Validate + smart corrections
    pn_valid = validate(passport_number_field, passport_check)
    if not pn_valid:
        corrected = smart_digit_swap_to_pass(passport_number_field, passport_check)
        if corrected:
            passport_number_field = corrected
            passport_number = corrected.replace('<','')
            pn_valid = True

    dob_valid = validate(dob_raw, dob_check)
    expiry_valid = validate(expiry_raw, expiry_check)
    if not expiry_valid:
        corrected_exp = smart_digit_swap_to_pass(expiry_raw, expiry_check)
        if corrected_exp:
            expiry_raw = corrected_exp
            expiry_valid = True

    dob = expand_date_yyMMdd(dob_raw, 'dob')
    expiry = expand_date_yyMMdd(expiry_raw, 'exp')

    return {
        'doc_type': doc_type,
        'issuing_country': issuing_country,
        'surname': surname,
        'given_names': given_names,
        'full_name': f"{given_names} {surname}".strip(),
        'passport_number': passport_number,
        'passport_number_check': passport_check,
        'passport_valid': pn_valid,
        'nationality': nationality,
        'date_of_birth': dob,
        'dob_valid': dob_valid,
        'sex': sex,
        'expiry_date': expiry,
        'expiry_valid': expiry_valid,
        'mrz_line1': line1,
        'mrz_line2': line2,
    }

# ==================================
# OpenCV MRZ detection & OCR reader
# ==================================

class OCRWrapper:
    """Small adapter that hides the difference between RapidOCR and EasyOCR."""

    def __init__(self):
        self.backend = None
        self.engine = None

        if RapidOCR is not None:
            self.engine = RapidOCR(det_use_cuda=False, rec_use_cuda=False, cls_use_cuda=False)
            self.backend = "rapidocr"
        elif easyocr is not None:
            # EasyOCR can run fully on CPU; restrict to Latin characters.
            self.engine = easyocr.Reader(["en"], gpu=False)
            self.backend = "easyocr"

        if self.engine is None:
            raise RuntimeError(
                "No OCR backend available. Please install 'rapidocr-onnxruntime' (preferred) or 'easyocr'."
            )

    def __call__(self, image: np.ndarray):
        if self.backend == "rapidocr":
            return self.engine(image)

        # EasyOCR returns a list of (bbox, text, confidence).
        results = self.engine.readtext(image, detail=1, paragraph=False)
        formatted = []
        for item in results:
            if len(item) < 2:
                continue
            if len(item) == 2:
                bbox, text = item
                conf = 0.0
            else:
                bbox, text, conf = item
            formatted.append((bbox, text, conf))
        return formatted, None


@st.cache_resource(show_spinner=False)
def get_ocr():
    return OCRWrapper()


def find_mrz_crop(bgr: np.ndarray) -> np.ndarray:
    """Return an MRZ-focused crop.

    If OpenCV is available we use contour based detection. Otherwise we fall back to
    returning the bottom third of the image which still contains the MRZ in most
    passport photos.
    """

    h, w = bgr.shape[:2]

    if cv2 is None:
        y0 = int(h * 0.6)
        return bgr[y0:h, 0:w]

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
    grad = cv2.convertScaleAbs(cv2.subtract(gradX, gradY))
    grad = cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX)
    _, thresh = cv2.threshold(grad, 0, 255, cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    closed = cv2.dilate(closed, None, iterations=1)
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        y0 = int(h * 0.65)
        return bgr[y0:h, 0:w]
    candidates = []
    for c in cnts:
        x, y, wc, hc = cv2.boundingRect(c)
        aspect = wc / float(hc + 1e-6)
        area = wc * hc
        if aspect > 6 and area > (w * h) * 0.02 and y > h * 0.4:
            candidates.append((y, x, wc, hc))
    if not candidates:
        y0 = int(h * 0.65)
        return bgr[y0:h, 0:w]
    candidates.sort(key=lambda t: t[0], reverse=True)
    y, x, wc, hc = candidates[0]
    pad = int(0.02 * w)
    x0 = max(0, x - pad)
    x1 = min(w, x + wc + pad)
    y0 = max(0, y - int(0.02 * h))
    y1 = min(h, y + hc + int(0.02 * h))
    return bgr[y0:y1, x0:x1]


def ocr_mrz_lines(pil_img: Image.Image):
    with st.spinner("Loading OCR model (first run can take up to a minute)..."):
        ocr = get_ocr()
    rgb = np.array(pil_img)

    if cv2 is not None:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        mrz_bgr = find_mrz_crop(bgr)
        gray = cv2.cvtColor(mrz_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        bin_img = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 10
        )
        mrz_preview = Image.fromarray(cv2.cvtColor(mrz_bgr, cv2.COLOR_BGR2RGB))
    else:
        # Fallback: simple bottom crop & threshold using numpy only
        mrz_rgb = find_mrz_crop(rgb)
        gray = (0.299 * mrz_rgb[:, :, 0] + 0.587 * mrz_rgb[:, :, 1] + 0.114 * mrz_rgb[:, :, 2]).astype(
            np.uint8
        )
        thresh = gray.mean()
        bin_img = (gray > thresh).astype(np.uint8) * 255
        mrz_preview = Image.fromarray(mrz_rgb)

    result, _ = ocr(bin_img)
    candidates = []
    if result:
        for item in result:
            if len(item) < 2:
                continue
            text = item[1]
            if isinstance(text, (list, tuple)):
                text = text[0]
            text = str(text).upper()
            text = re.sub(r'[^A-Z0-9<]', '', text)
            if len(text) >= 30:
                candidates.append(text)
    candidates = sorted(set(candidates), key=lambda s: len(s), reverse=True)
    if len(candidates) >= 2:
        l1, l2 = candidates[0], candidates[1]
    elif len(candidates) == 1:
        l1, l2 = candidates[0], ''
    else:
        l1, l2 = '', ''
    return l1, l2, mrz_preview

# ==============================
# PDF manifest parsing (PyPDF2)
# ==============================

PASSPORT_RE = re.compile(
    r"P\s*Number\s*([A-Z0-9]+)\s*\(Exp\.\s*([0-9]{4}-[0-9]{2}-[0-9]{2})\)",
    re.IGNORECASE,
)
NAME_RE = re.compile(r"Name\s+([A-Z'\- ]+)", re.IGNORECASE)


def normalize_name(n: str) -> str:
    n = n.upper().strip()
    n = re.sub(r"[^A-Z ]", " ", n)
    n = re.sub(r"\s+", " ", n)
    return n


def parse_manifest_pdf(uploaded_pdf) -> list:
    """Return list of dicts: {role: 'CREW'/'PAX', name, passport_number, expiry_date} """
    reader = PyPDF2.PdfReader(uploaded_pdf)
    text = "\n".join(page.extract_text() or '' for page in reader.pages)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    entries = []
    current_role = None
    current_name = None

    for ln in lines:
        u = ln.upper()
        if u.startswith('CREW'):
            current_role = 'CREW'
            current_name = None
            continue
        if u.startswith('PAX'):
            current_role = 'PAX'
            current_name = None
            continue
        m_name = NAME_RE.search(ln)
        if m_name:
            current_name = normalize_name(m_name.group(1))
            continue
        m_pass = PASSPORT_RE.search(ln)
        if m_pass and current_name:
            pnum = m_pass.group(1).strip()
            exp = m_pass.group(2).strip()
            entries.append({
                'role': current_role or '',
                'name': current_name,
                'passport_number': pnum,
                'expiry_date': exp
            })
            current_name = None

    return entries

# ==========================
# Matching logic & UI output
# ==========================

def compare_dates(exp_manifest: str, exp_mrz: str) -> bool:
    try:
        d1 = datetime.strptime(exp_manifest, "%Y-%m-%d").date()
        d2 = datetime.strptime(exp_mrz, "%Y-%m-%d").date()
        return d1 == d2
    except Exception:
        return False


def name_similarity(n1: str, n2: str) -> float:
    s1 = set(normalize_name(n1).split())
    s2 = set(normalize_name(n2).split())
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)


st.title("🛂 Passport Scan → PDF Manifest Match (No API)")

st.info(
    "First MRZ scan downloads the RapidOCR ONNX weights (~90 MB). The initial fetch can take up to a minute, "
    "so please keep the app tab open until it finishes."
)

col_left, col_right = st.columns([1,1])

with col_left:
    st.subheader("1) Upload manifest PDF")
    pdf_file = st.file_uploader("PDF (e.g., eAPIS)", type=["pdf"])
    manifest = []
    if pdf_file:
        try:
            manifest = parse_manifest_pdf(pdf_file)
            st.success(f"Parsed {len(manifest)} entries from manifest.")
            st.dataframe(manifest, use_container_width=True)
        except Exception as e:
            st.error(f"PDF parse error: {e}")

with col_right:
    st.subheader("2) Scan passports (camera or upload)")
    mode = st.radio("Input", ["📷 Camera", "📂 Upload"], horizontal=True)
    scans = []
    if mode == "📷 Camera":
        cam = st.camera_input("Take a passport photo")
        if cam:
            img = Image.open(cam).convert('RGB')
            l1,l2,prev = ocr_mrz_lines(img)
            st.image(prev, caption="MRZ crop", use_container_width=True)
            st.code(f"{l1}\n{l2}")
            if l1 and l2:
                scans.append(parse_mrz_lines(l1,l2))
    else:
        files = st.file_uploader("Upload passport photos", type=["jpg","jpeg","png"], accept_multiple_files=True)
        if files:
            for f in files:
                img = Image.open(f).convert('RGB')
                l1,l2,prev = ocr_mrz_lines(img)
                st.image(prev, caption=f"MRZ crop: {f.name}", use_container_width=True)
                st.code(f"{l1}\n{l2}")
                if l1 and l2:
                    scans.append(parse_mrz_lines(l1,l2))

st.markdown("---")

if manifest and scans:
    st.subheader("3) Matching results")
    rows = []
    unmatched_scans = []

    idx_by_pn = {m['passport_number'].upper(): m for m in manifest}

    for s in scans:
        s_pn = (s.get('passport_number') or '').upper()
        s_name = s.get('full_name','')
        s_exp = s.get('expiry_date','')
        match = idx_by_pn.get(s_pn)
        if match:
            name_ok = name_similarity(match['name'], s_name) >= 0.5
            exp_ok = compare_dates(match['expiry_date'], s_exp)
            status = "✅ Match" if (name_ok and exp_ok) else ("⚠️ Check name" if not name_ok else "⚠️ Check expiry")
            rows.append({
                'status': status,
                'manifest_name': match['name'],
                'manifest_passport': match['passport_number'],
                'manifest_expiry': match['expiry_date'],
                'scan_name': s_name,
                'scan_passport': s_pn,
                'scan_expiry': s_exp
            })
        else:
            unmatched_scans.append(s)

    if rows:
        st.dataframe(rows, use_container_width=True)

    if unmatched_scans:
        st.warning("Some scanned passports were not found in the manifest by passport number:")
        for s in unmatched_scans:
            st.write({
                'scan_name': s.get('full_name',''),
                'scan_passport': s.get('passport_number',''),
                'scan_expiry': s.get('expiry_date','')
            })

st.caption("Note: For best accuracy, fill the frame with the MRZ band, reduce glare, and keep the passport flat.")

# ==================
# Requirements to add
# ==================
# streamlit
# pillow
# numpy
# opencv-python-headless
# rapidocr-onnxruntime
# PyPDF2
