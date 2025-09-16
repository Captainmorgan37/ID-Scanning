import streamlit as st
from openai import OpenAI
from PIL import Image, ImageOps, ImageFilter
import io, base64
import json

# --- Convert PIL image to base64 for GPT ---
def image_to_base64(img: Image.Image) -> str:
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# --- Preprocess: crop + enhance MRZ area ---
def crop_mrz(img: Image.Image) -> Image.Image:
    w, h = img.size
    mrz_zone = img.crop((0, int(h*0.65), w, h))  # bottom ~35%
    mrz_zone = mrz_zone.convert("L")             # grayscale
    mrz_zone = ImageOps.autocontrast(mrz_zone)   # boost contrast
    mrz_zone = mrz_zone.filter(ImageFilter.SHARPEN)
    return mrz_zone

# --- Check digit validation ---
def compute_check_digit(data: str) -> str:
    weights = [7, 3, 1]
    total = 0
    for i, char in enumerate(data):
        if char.isdigit():
            val = int(char)
        elif char == "<":
            val = 0
        else:
            val = ord(char) - 55  # A=10, B=11...
        total += val * weights[i % 3]
    return str(total % 10)

def validate_mrz_field(field: str, check_digit: str) -> bool:
    return compute_check_digit(field) == check_digit

# --- Parse MRZ lines deterministically ---
def parse_mrz_lines(line1: str, line2: str):
    line1 = (line1 + "<"*44)[:44]
    line2 = (line2 + "<"*44)[:44]

    surname, given_names = "", ""
    try:
        # Line 1
        doc_type = line1[0:2]
        issuing_country = line1[2:5]
        names_raw = line1[5:44]
        parts = names_raw.split("<<")
        surname = parts[0].replace("<", " ").strip()
        given_names = " ".join([p.replace("<", " ").strip() for p in parts[1:] if p])

        # Line 2
        passport_number = line2[0:9].replace("<", "")
        passport_number_check = line2[9]
        nationality = line2[10:13]
        dob_raw = line2[13:19]
        dob_check = line2[19]
        sex = line2[20]
        expiry_raw = line2[21:27]
        expiry_check = line2[27]

        # Expand dates
        dob = f"19{dob_raw[0:2]}-{dob_raw[2:4]}-{dob_raw[4:6]}"
        expiry = f"20{expiry_raw[0:2]}-{expiry_raw[2:4]}-{expiry_raw[4:6]}"

        return {
            "doc_type": doc_type,
            "issuing_country": issuing_country,
            "surname": surname,
            "given_names": given_names,
            "passport_number": passport_number,
            "passport_number_check": passport_number_check,
            "passport_valid": validate_mrz_field(line2[0:9], passport_number_check),
            "nationality": nationality,
            "date_of_birth": dob,
            "dob_valid": validate_mrz_field(dob_raw, dob_check),
            "sex": sex,
            "expiry_date": expiry,
            "expiry_valid": validate_mrz_field(expiry_raw, expiry_check),
            "mrz_line1": line1,
            "mrz_line2": line2
        }
    except Exception:
        return {"mrz_line1": line1, "mrz_line2": line2}

# --- Streamlit App ---
st.set_page_config(page_title="Passport Scanner", page_icon="🛂", layout="centered")
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.title("🛂 Passport Scanner (Two-Step MRZ Parsing)")
st.write("Step 1: GPT extracts raw MRZ lines. Step 2: Parsing + validation done locally.")

uploaded = st.file_uploader("Upload passport photo", type=["jpg", "jpeg", "png"])
cam = st.camera_input("Or take a picture")

image_file = cam or uploaded

if image_file:
    full_img = Image.open(image_file)
    st.image(full_img, caption="Original Photo")

    # Crop MRZ zone
    mrz_img = crop_mrz(full_img)
    st.image(mrz_img, caption="Processed MRZ Zone")

    with st.spinner("Extracting raw MRZ lines with GPT..."):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a document OCR assistant. Extract only the two MRZ lines "
                            "(44 characters each, uppercase, < for filler, no spaces, no JSON, no explanations)."
                        )
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Return only the two MRZ lines."},
                            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + image_to_base64(mrz_img)}}
                        ]
                    }
                ],
                max_tokens=200
            )
            mrz_text = response.choices[0].message.content.strip()
        except Exception as e:
            mrz_text = f"Error: {e}"

    st.subheader("Raw MRZ Extracted")
    st.code(mrz_text)

    # Split into lines and parse locally
    lines = [l.strip() for l in mrz_text.splitlines() if l.strip()]
    if len(lines) >= 2:
        parsed = parse_mrz_lines(lines[0], lines[1])

        st.subheader("Parsed Fields with Validation")
        st.markdown(f"**Surname:** {parsed.get('surname','')}")
        st.markdown(f"**Given Names:** {parsed.get('given_names','')}")
        st.markdown(f"**Passport Number:** {parsed.get('passport_number','')} → {'✅ valid' if parsed.get('passport_valid') else '❌ invalid'}")
        st.markdown(f"**Nationality:** {parsed.get('nationality','')}")
        st.markdown(f"**Date of Birth:** {parsed.get('date_of_birth','')} → {'✅ valid' if parsed.get('dob_valid') else '❌ invalid'}")
        st.markdown(f"**Sex:** {parsed.get('sex','')}")
        st.markdown(f"**Expiry Date:** {parsed.get('expiry_date','')} → {'✅ valid' if parsed.get('expiry_valid') else '❌ invalid'}")

        st.text_area("MRZ Line 1", parsed.get("mrz_line1",""))
        st.text_area("MRZ Line 2", parsed.get("mrz_line2",""))
    else:
        st.error("Could not extract two valid MRZ lines.")
