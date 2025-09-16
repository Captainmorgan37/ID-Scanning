import streamlit as st
from openai import OpenAI
from PIL import Image
import io, base64

# --- Helper to convert PIL image to base64 string ---
def image_to_base64(img: Image.Image) -> str:
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# --- Streamlit App ---
st.set_page_config(page_title="Passport Scanner", page_icon="🛂", layout="centered")
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.title("🛂 Passport Scanner (AI Vision)")
st.write("Upload or capture a passport photo. The app uses GPT-4o-mini vision to read the MRZ and extract fields.")

# Upload or Camera input
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
                            "You are a document parsing assistant. Extract the machine-readable zone (MRZ) "
                            "from passports and parse it into structured fields. "
                            "Return JSON with: surname, given_names, passport_number, nationality, "
                            "date_of_birth (YYYY-MM-DD), sex, expiry_date (YYYY-MM-DD)."
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

            result_text = response.choices[0].message.content

        except Exception as e:
            result_text = f"Error: {e}"

    st.subheader("Extracted Data")
    st.write(result_text)
