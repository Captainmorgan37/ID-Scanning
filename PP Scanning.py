import streamlit as st
from openai import OpenAI
from PIL import Image
import io, base64

def image_to_base64(img: Image.Image) -> str:
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

st.set_page_config(page_title="Passport Scanner", page_icon="🛂", layout="centered")
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.title("🛂 Passport Scanner (AI Vision)")
st.write("Upload or capture a passport photo. The app uses GPT-4o-mini vision to read the MRZ and show a sample email draft.")

uploaded = st.file_uploader("Upload passport photo", type=["jpg", "jpeg", "png"])
cam = st.camera_input("Or take a picture")

image_file = cam or uploaded

if image_file:
    image = Image.open(image_file)
    st.image(image, caption="Captured Image")

    with st.spinner("Analyzing passport..."):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a document parsing assistant. Extract the MRZ from passports "
                            "and parse it into structured fields. Return JSON with: surname, given_names, "
                            "passport_number, nationality, date_of_birth (YYYY-MM-DD), sex, expiry_date (YYYY-MM-DD)."
                        )
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Extract MRZ and structured fields from this passport."},
                            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + image_to_base64(image)}}
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

    # --- Show sample email draft ---
    st.subheader("📧 Sample Email Draft")
    sample_email = f"""
To: example@domain.com
Subject: Passport Information Submission

Hello,

Please find below the extracted passport information:

{parsed_json}

The scanned passport image is attached for reference.

Regards,
Automated Passport Scanner
"""
    st.text_area("Email Preview", sample_email, height=300)

    # Show "attachment preview"
    st.image(image, caption="This image would be attached to the email", use_container_width=True)
