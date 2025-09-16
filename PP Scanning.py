import streamlit as st
from openai import OpenAI
from PIL import Image, ImageOps, ImageFilter
import io, base64
import re
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

# --- Check digit validation (ISO 18013 MRZ algorithm) ---
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

# --- Streamlit App ---
st.set_page_config(page_title="Passport Scanner", page_icon="🛂", layout="centered")
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.title("🛂 Passport Scanner (With Validation)")
st.write("This version crops + enhances the MRZ, extracts fields with GPT, and validates check digits.")

uploaded = st.file_uploader("Upload passport photo", type=["jpg", "jpeg", "png"])
cam = st.camera_input("Or take a picture")

image_file = cam or uploaded

if image_file:
    full_img = Image.open(image_file)
    st.image(full_img, caption="Original Photo")

    # Crop MRZ zone
    mrz_img = crop_mrz(full_img)
    st.image(mrz_img, caption="Processed MRZ Zone")

    with st.spinner("Analyzing MRZ with AI..."):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a document parsing assistant. Extract the MRZ from passports "
                            "and parse it into structured fields. "
                            "Return JSON with keys: surname, given_names, passport_number, passport_number_check, "
                            "nationality, date_of_birth (YYYY-MM-DD), sex, expiry_date (YYYY-MM-DD), expiry_check."
                        )
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Extract MRZ fields and include the MRZ check digits."},
                            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + image_to_base64(mrz_img)}}
                        ]
                    }
                ],
                max_tokens=500
            )
            parsed_json = response.choices[0].message.content
        except Exception as e:
            parsed_json = f"Error: {e}"

    st.subheader("Extracted Data (Raw JSON)")
    st.code(parsed_json, language="json")

    # --- Post-processing: validate check digits ---
    try:
        data = json.loads(parsed_json)
    except Exception:
        data = {}

    if data:
        st.subheader("Validated Fields")
        # Passport number
        pn = data.get("passport_number", "")
        pn_check = data.get("passport_number_check", "")
        pn_valid = validate_mrz_field(pn, pn_check) if pn and pn_check else False

        # Expiry
        exp = re.sub(r"[^0-9]", "", data.get("expiry_date", ""))
        exp_check = data.get("expiry_check", "")
        exp_valid = validate_mrz_field(exp, exp_check) if exp and exp_check else False

        # Show results
        st.markdown(f"**Surname:** {data.get('surname','')}")
        st.markdown(f"**Given Names:** {data.get('given_names','')}")
        st.markdown(f"**Passport Number:** {pn} → {'✅ valid' if pn_valid else '❌ invalid'}")
        st.markdown(f"**Nationality:** {data.get('nationality','')}")
        st.markdown(f"**Date of Birth:** {data.get('date_of_birth','')}")
        st.markdown(f"**Sex:** {data.get('sex','')}")
        st.markdown(f"**Expiry Date:** {data.get('expiry_date','')} → {'✅ valid' if exp_valid else '❌ invalid'}")
