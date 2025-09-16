import streamlit as st
from PIL import Image, ImageOps, ImageFilter
import io
import re
import pytesseract

# --- Helper functions ---
def preprocess_for_mrz(img: Image.Image) -> Image.Image:
    """Crop to bottom portion and enhance contrast for MRZ OCR"""
    img = img.convert("L")
    img = ImageOps.autocontrast(img)
    img = img.filter(ImageFilter.SHARPEN)
    w, h = img.size
    bottom_portion = img.crop((0, int(h*0.65), w, h))
    bottom_portion = bottom_portion.resize((int(w*1.5), int((h*0.35)*1.5)))
    return bottom_portion

def find_mrz_text(img: Image.Image) -> str:
    """Run OCR and return MRZ-like text (only A-Z, 0-9, <)"""
    mrz_img = preprocess_for_mrz(img)
    text = pytesseract.image_to_string(mrz_img, config='--psm 6')
    lines = [re.sub(r'[^A-Z0-9<\n]', '', l.upper()) for l in text.splitlines() if l.strip()]
    candidates = [l for l in lines if len(l) >= 30]
    if len(candidates) >= 2:
        return "\n".join(candidates[:2])
    return "\n".join(lines)

def parse_mrz_td3(mrz_text: str):
    """Parse TD3-format MRZ (2 lines, 44 chars) into structured fields"""
    lines = [l.replace(" ", "") for l in mrz_text.splitlines() if l.strip()]
    if len(lines) < 2:
        return {}
    l1, l2 = lines[0], lines[1]
    l1 = (l1 + "<"*44)[:44]
    l2 = (l2 + "<"*44)[:44]
    try:
        doc_type = l1[0:2]
        issuing_country = l1[2:5]
        names_raw = l1[5:44]
        names = names_raw.split("<<")
        surname = names[0].replace("<", " ").strip()
        given_names = " ".join([n.replace("<", " ").strip() for n in names[1:] if n])
        passport_number = l2[0:9].replace("<", "").strip()
        nationality = l2[10:13]
        dob_raw = l2[13:19]  # YYMMDD
        dob = f"19{dob_raw[0:2]}-{dob_raw[2:4]}-{dob_raw[4:6]}"
        sex = l2[20]
        expiry_raw = l2[21:27]
        expiry = f"20{expiry_raw[0:2]}-{expiry_raw[2:4]}-{expiry_raw[4:6]}"
        return {
            "doc_type": doc_type,
            "issuing_country": issuing_country,
            "surname": surname,
            "given_names": given_names,
            "passport_number": passport_number,
            "nationality": nationality,
            "date_of_birth": dob,
            "sex": sex,
            "expiry_date": expiry,
            "mrz_raw": mrz_text
        }
    except Exception:
        return {"mrz_raw": mrz_text}

# --- Streamlit UI ---
st.set_page_config(page_title="Passport Scanner", page_icon="🛂", layout="centered")

st.title("🛂 Passport Scanner (Test Mode)")
st.write("Upload or capture a passport photo. This prototype extracts the MRZ and parses key fields.")

uploaded = st.file_uploader("Upload passport photo", type=["jpg", "jpeg", "png"])
cam = st.camera_input("Or take a picture")

image_file = cam or uploaded

if image_file:
    img = Image.open(image_file)
    st.image(img, caption="Captured image")

    with st.spinner("Extracting MRZ..."):
        mrz_text = find_mrz_text(img)
        parsed = parse_mrz_td3(mrz_text)

    st.subheader("Extracted Data (editable)")
    fname = st.text_input("Given names", value=parsed.get("given_names", ""))
    sname = st.text_input("Surname", value=parsed.get("surname", ""))
    pnum = st.text_input("Passport number", value=parsed.get("passport_number", ""))
    nat = st.text_input("Nationality", value=parsed.get("nationality", ""))
    dob = st.text_input("Date of birth (YYYY-MM-DD)", value=parsed.get("date_of_birth", ""))
    sex = st.text_input("Sex (M/F)", value=parsed.get("sex", ""))
    expiry = st.text_input("Expiry (YYYY-MM-DD)", value=parsed.get("expiry_date", ""))
    st.text_area("Raw MRZ / OCR", value=parsed.get("mrz_raw", mrz_text), height=120)
