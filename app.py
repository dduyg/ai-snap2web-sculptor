import streamlit as st
from PIL import Image
import numpy as np
import io
import base64
from sklearn.cluster import KMeans
from openai import OpenAI

st.set_page_config(page_title="Screenshot → Webpage HTML Generator", layout="wide")

st.title("Screenshot → Full Webpage HTML (AI Reconstruction)")
st.write("""
Upload a screenshot of any webpage and the AI will reconstruct it as a **complete single-file HTML**
page with inline CSS, inline JS, and embedded base64 images.
""")

# Ensure API key exists
if "OPENAI_API_KEY" not in st.secrets:
    st.error("Missing API key in secrets.toml")
    st.stop()

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ------------------------------
# Model Selection
# ------------------------------
model_choice = st.sidebar.selectbox(
    "Choose AI Model",
    ["gpt-4o-mini", "gpt-4o", "gpt-4.1"],
    index=0
)

# ------------------------------
# Color Palette Extraction
# ------------------------------
def extract_palette(img: Image.Image, n_colors=5):
    arr = np.array(img)
    h, w, _ = arr.shape
    sample = arr.reshape((h * w, 3))

    km = KMeans(n_clusters=n_colors, n_init="auto")
    km.fit(sample)

    centers = km.cluster_centers_.astype(int)
    hex_colors = [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in centers]
    return hex_colors

# ------------------------------
# Simple Font-Style Heuristic
# ------------------------------
def detect_font_style(img: Image.Image):
    arr = np.array(img.convert("L"))
    edges = np.mean(np.abs(np.diff(arr, axis=1)))
    variance = np.var(arr)

    # Simple heuristics
    if variance < 500:
        style = "lightweight sans-serif such as 'Inter' or 'Roboto'"
    elif edges > 20:
        style = "serif such as 'Georgia' or 'Times New Roman'"
    else:
        style = "modern sans-serif such as 'SF Pro', 'Inter', or 'Roboto'"

    return style

# ------------------------------
# Convert image to base64
# ------------------------------
def img_to_base64(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

# ------------------------------
# Main app
# ------------------------------
uploaded = st.file_uploader("Upload screenshot", type=["png", "jpg", "jpeg"])

BASE_PROMPT = """
You are a professional webpage reconstruction engine.

Recreate the webpage shown in the screenshot as a **single HTML file**.

REQUIREMENTS:
- Output a FULL HTML document: <html>, <head>, <style>, <body>, optional <script>.
- Use ONLY inline CSS (inside <style>).
- No external CSS or JS.
- Reproduce structure, layout, colors, spacing, fonts, and proportions accurately.
- Include base64 images where appropriate.
- Maintain semantic hierarchy (headers, sections, nav, buttons, etc.)
- DO NOT output explanations. ONLY output HTML.
"""

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded Screenshot", use_column_width=True)

    # ---- Extract palette and font ----
    with st.spinner("Extracting color palette…"):
        palette = extract_palette(image)

    with st.spinner("Analyzing font style…"):
        font_style = detect_font_style(image)

    st.sidebar.markdown("### 🎨 Detected Color Palette")
    for c in palette:
        st.sidebar.markdown(f"<div style='background:{c};width:100%;height:20px;border-radius:4px;'></div> {c}", unsafe_allow_html=True)

    st.sidebar.markdown("### 🔤 Likely Font Style")
    st.sidebar.write(font_style)

    # ---- Final Prompt ----
    enriched_prompt = BASE_PROMPT + f"""

Color Palette (most dominant first):
{', '.join(palette)}

Recommended Font Family:
Use {font_style}.

Use these colors and font recommendations to match the screenshot as closely as possible.
"""

    if st.button("Generate HTML"):
        with st.spinner(f"Generating HTML using {model_choice}…"):

            img_b64 = img_to_base64(image)

            response = client.chat.completions.create(
                model=model_choice,
                max_tokens=8000,
                temperature=0,
                messages=[
                    {"role": "system", "content": enriched_prompt},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": f"data:image/png;base64,{img_b64}"
                            }
                        ]
                    }
                ]
            )

            html_code = response.choices[0].message.content.strip()

        st.success("HTML successfully generated!")

        # ---- Preview ----
        st.subheader("Live Preview")
        st.components.v1.html(html_code, height=1000, scrolling=True)

        # ---- Code view ----
        st.subheader("Generated HTML Code")
        st.code(html_code, language="html")

        # ---- Download ----
        st.download_button(
            "Download index.html",
            data=html_code,
            file_name="index.html",
            mime="text/html"
        )

        # ---- Copy to clipboard ----
        copy_button = f"""
            <textarea id="htmlcode" style="display:none;">{html_code}</textarea>
            <button onclick="
                navigator.clipboard.writeText(document.getElementById('htmlcode').value)
                .then(() => alert('Copied to clipboard!'));
            ">Copy to Clipboard</button>
        """
        st.components.v1.html(copy_button, height=60)

else:
    st.info("Upload a webpage screenshot to begin.")
