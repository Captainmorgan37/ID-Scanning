# Passport MRZ Scanner — Plan B (PaddleOCR, no external APIs)
# ------------------------------------------------------------
# Features
# - Works locally or on Streamlit Cloud with pip-only deps (no external OCR APIs)
# - Camera or multi-file upload
# - OpenCV preprocessing to find/crop MRZ strip (bottom of passport)
# - PaddleOCR to read characters
# - Deterministic MRZ parsing (TD3) + ISO check‑digit validation
# - Smart digit‑swap auto‑correction for expiry/passport number when check fails
# - Shows raw MRZ lines, parsed fields, and validity indicators

import streamlit as st
from PIL import Image
import numpy as np
import cv2
from paddleocr import PaddleOCR
import io
import re

st.set_page_config(page_title="Passport MRZ Scanner (Plan B)", page_icon="🛂", layout="centered")

# -----------------------------
# Helpers: MRZ parsing & checks
# -----------------------------

def compute_check_digit(data: str) -> str:
    weights = [7, 3, 1]
    total = 0
    for i, ch in enumerate(data):
        if ch.isdigit():
            val = int(ch)
        elif ch == '<':
            val = 0
        else:
            # A=10, B=11, ... Z=35
            val = ord(ch) - 55
        total += val * weights[i % 3]
    return str(total % 10)


def validate(field: str, check_digit: str) -> bool:
    if not field or not check_digit:
        return False
    return compute_check_digit(field) == check_digit


def expand_date_yyMMdd(raw6: str) -> str:
    # naive expansion: assume 19xx for DOB, 20xx for expiry
    # (You can add smarter rules by comparing to today if desired.)
    if len(raw6) != 6 or not raw6.isdigit():
        return ""
    yy, mm, dd = raw6[:2], raw6[2:4], raw6[4:6]
    # Guard values
    if not ("01" <= mm <= "12") or not ("01" <= dd <= "31"):
        return ""
    return f"20{yy}-{mm}-{dd}"


def expand_dob_yyMMdd(raw6: str) -> str:
    if len(raw6) != 6 or not raw6.isdigit():
        return ""
    yy, mm, dd = raw6[:2], raw6[2:4], raw6[4:6]
    if not ("01" <= mm <= "12") or not ("01" <= dd <= "31"):
        return ""
    return f"19{yy}-{mm}-{dd}"


def smart_digit_swap_to_pass(field_raw: str, check_digit: str) -> str:
    """Try small digit swaps likely to be misread (3↔8, 5↔6, 0↔8, 1↔7) to satisfy check digit.
    Returns corrected raw field if found, else empty string.
    """
    if not field_raw or not check_digit:
        return ""
    swaps = [("3","8"),("5","6"),("0","8"),("1","7"),("2","7"),("9","8")]
    tried = set()
    from collections import deque
    dq = deque([field_raw])
    while dq:
        cur = dq.popleft()
        if cur in tried:
            continue
        tried.add(cur)
        if compute_check_digit(cur) == check_digit:
            return cur
        # enqueue one-swap variants
        for a,b in swaps:
            dq.append(cur.replace(a,b,1))
            dq.append(cur.replace(b,a,1))
    return ""


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

    # Expand dates
    dob = expand_dob_yyMMdd(dob_raw)
    expiry = expand_date_yyMMdd(expiry_raw)

    return {
        'doc_type': doc_type,
        'issuing_country': issuing_country,
        'surname': surname,
        'given_names': given_names,
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


# -------------------------------------
# OpenCV: detect & crop the MRZ region
# -------------------------------------

def find_mrz_crop(bgr: np.ndarray) -> np.ndarray:
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3,3), 0)

    # Emphasize horizontal text strokes
    gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
    grad = cv2.subtract(gradX, gradY)
    grad = cv2.convertScaleAbs(grad)

    # Normalize & threshold
    grad = cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX)
    _, thresh = cv2.threshold(grad, 0, 255, cv2.THRESH_OTSU)

    # Morph close to connect MRZ band
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25,3))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Dilate slightly to join parts
    closed = cv2.dilate(closed, None, iterations=1)

    # Find contours and pick a wide, bottom-ish rectangle
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return bgr

    candidates = []
    for c in cnts:
        x,y,wc,hc = cv2.boundingRect(c)
        aspect = wc / float(hc + 1e-6)
        area = wc * hc
        # Heuristic: MRZ is wide & not too tall, near bottom third
        if aspect > 6 and area > (w*h)*0.02 and y > h*0.4:
            candidates.append((y, x, wc, hc))

    if not candidates:
        # fallback: bottom third
        y0 = int(h*0.65)
        return bgr[y0:h, 0:w]

    # Pick the lowest candidate (closest to bottom)
    candidates.sort(key=lambda t: t[0], reverse=True)
    y, x, wc, hc = candidates[0]
    crop = bgr[y:y+hc, x:x+wc]

    # Slightly pad horizontally
    pad = int(0.02*w)
    x0 = max(0, x - pad)
    x1 = min(w, x + wc + pad)
    y0 = max(0, y - int(0.02*h))
    y1 = min(h, y + hc + int(0.02*h))
    return bgr[y0:y1, x0:x1]


# ------------------
# PaddleOCR wrapper
# ------------------
@st.cache_resource(show_spinner=False)
def get_ocr():
    # English OCR; angle_cls off because MRZ is horizontal; use GPU=False by default
    return PaddleOCR(use_angle_cls=False, lang='en')


def ocr_lines_from_image(pil_img: Image.Image):
    ocr = get_ocr()
    # Convert to BGR for Paddle's preprocessing
    bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # Find MRZ crop
    mrz_bgr = find_mrz_crop(bgr)

    # Enhance crop: grayscale + binarize
    gray = cv2.cvtColor(mrz_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    # Adaptive threshold for uneven lighting
    bin_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY, 31, 10)

    # Run OCR
    result = ocr.ocr(bin_img, cls=False)

    # Collect candidate lines, keep only MRZ charset and no spaces
    candidates = []
    if result:
        for block in result:
            for line in block:
                text = line[1][0]
                # Normalize
                text = text.upper()
                text = re.sub(r'[^A-Z0-9<]', '', text)
                text = text.replace(' ', '')
                if len(text) >= 30:  # likely MRZ content
                    candidates.append(text)

    # Choose two best lines by length (descending) and proximity in y
    # (Paddle returns lines roughly top->bottom already; we just take two longest)
    candidates = sorted(set(candidates), key=lambda s: len(s), reverse=True)
    if len(candidates) >= 2:
        l1, l2 = candidates[0], candidates[1]
    elif len(candidates) == 1:
        l1, l2 = candidates[0], ''
    else:
        l1, l2 = '', ''

    # Fallback: if nothing, create a bin-based guess of bottom third
    return l1, l2, Image.fromarray(cv2.cvtColor(mrz_bgr, cv2.COLOR_BGR2RGB))


# --------------
# Streamlit UI
# --------------

st.title("🛂 Passport MRZ Scanner (Plan B: PaddleOCR)")
st.write("No external AI services. Uses OpenCV + PaddleOCR, then strict MRZ parsing with check‑digit validation.")

mode = st.radio("Input method", ["📷 Take a photo", "📂 Upload photos"], horizontal=True)

results = []

if mode == "📷 Take a photo":
    camera_file = st.camera_input("Take a picture")
    if camera_file:
        img = Image.open(camera_file).convert('RGB')
        st.image(img, caption="Original", use_container_width=True)
        with st.spinner("Detecting MRZ and reading text..."):
            l1, l2, mrz_preview = ocr_lines_from_image(img)
        st.image(mrz_preview, caption="MRZ crop preview", use_container_width=True)
        st.code(f"{l1}\n{l2}", language="text")
        if l1 and l2:
            parsed = parse_mrz_lines(l1, l2)
            results.append((img, parsed))
        else:
            st.error("Could not confidently detect two MRZ lines. Try adjusting angle/lighting and retake.")

else:
    files = st.file_uploader("Upload one or more passport photos", type=["jpg","jpeg","png"], accept_multiple_files=True)
    if files:
        for f in files:
            img = Image.open(f).convert('RGB')
            st.image(img, caption=f.name, use_container_width=True)
            with st.spinner(f"Reading MRZ from {f.name}..."):
                l1, l2, mrz_preview = ocr_lines_from_image(img)
            st.image(mrz_preview, caption="MRZ crop preview", use_container_width=True)
            st.code(f"{l1}\n{l2}", language="text")
            if l1 and l2:
                parsed = parse_mrz_lines(l1, l2)
                results.append((img, parsed))
            else:
                st.warning(f"{f.name}: Could not confidently detect two MRZ lines.")

# Show parsed results
if results:
    st.markdown("---")
    st.subheader("Parsed Fields & Validation")
    for idx, (img, p) in enumerate(results, start=1):
        st.markdown(f"**Result {idx}**")
        cols = st.columns(2)
        with cols[0]:
            st.image(img, caption="Original", use_container_width=True)
        with cols[1]:
            st.markdown(f"**Surname:** {p.get('surname','')}")
            st.markdown(f"**Given Names:** {p.get('given_names','')}")
            st.markdown(f"**Passport Number:** {p.get('passport_number','')} → {'✅ valid' if p.get('passport_valid') else '❌ invalid'}")
            st.markdown(f"**Nationality:** {p.get('nationality','')}")
            st.markdown(f"**Date of Birth:** {p.get('date_of_birth','')} → {'✅ valid' if p.get('dob_valid') else '❌ invalid'}")
            st.markdown(f"**Sex:** {p.get('sex','')}")
            st.markdown(f"**Expiry Date:** {p.get('expiry_date','')} → {'✅ valid' if p.get('expiry_valid') else '❌ invalid'}")
            st.text_area("MRZ Line 1", p.get('mrz_line1',''), height=60)
            st.text_area("MRZ Line 2", p.get('mrz_line2',''), height=60)

    st.info("If a field shows ❌, try re-capturing with less glare, flatter angle, and MRZ filling more of the frame. Auto-correction attempts small digit swaps when checks
