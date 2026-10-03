import streamlit as st
from google import genai
from google.genai import types

### Load your API Key
try:
    gemini_api_key = st.secrets['MyGeminiKey']  # Info: https://docs.streamlit.io/develop/api-reference/connections/st.secrets
except (KeyError, FileNotFoundError):
    st.error("No Gemini key found. Add `MyGeminiKey` under **Manage app → ⋮ → Settings → Secrets**, then refresh this page.")
    st.stop()

client = genai.Client(api_key=gemini_api_key)
MODEL = "gemini-3.1-flash-lite"

### PDF upload (optional)
uploaded_pdfs = st.file_uploader(
    "Upload PDF documents (optional)",
    type="pdf",
    accept_multiple_files=True,
)

### User input
query = st.text_area("Enter your question", placeholder="e.g., Summarize the key points of this document.")

if st.button("Ask Gemini"):
    if not query.strip():
        st.warning("Type a question first, then press Ask Gemini.")
    else:
        # Build the request: each PDF as a document part, followed by the question
        contents = [
            types.Part.from_bytes(data=pdf.getvalue(), mime_type="application/pdf")
            for pdf in uploaded_pdfs
        ]
        contents.append(query)

        with st.spinner("Reading documents and thinking..." if uploaded_pdfs else "Thinking..."):
            try:
                response = client.models.generate_content(model=MODEL, contents=contents)
                st.write(response.text)
            except Exception as e:
                st.error(f"Gemini request failed: {e}")
