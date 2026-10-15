import streamlit as st
from sentence_transformers import SentenceTransformer


@st.cache_resource
def load_model():
    return SentenceTransformer(
        "zoraizbinsamee/resume-job-matching-sbert"
    )


model = load_model()


st.title("SEAblings Resume–Job Matcher")

resume_text = st.text_area(
    "Paste your resume",
    height=300
)

job_requirement = st.text_input(
    "Enter one job requirement",
    placeholder="Example: Experience managing cross-functional stakeholders"
)


if st.button("Find Resume Evidence"):

    if not resume_text or not job_requirement:
        st.warning("Please provide both inputs.")

    else:

        # Split resume into lines / bullets
        resume_lines = [
            line.strip()
            for line in resume_text.split("\n")
            if len(line.strip()) > 20
        ]

        # Encode requirement
        requirement_embedding = model.encode(
            [job_requirement],
            convert_to_tensor=True
        )

        # Encode all resume bullets
        resume_embeddings = model.encode(
            resume_lines,
            convert_to_tensor=True
        )

        # Compare requirement against every resume bullet
        similarities = model.similarity(
            requirement_embedding,
            resume_embeddings
        )[0]

        # Find best match
        best_index = similarities.argmax().item()
        best_score = similarities[best_index].item()

        best_evidence = resume_lines[best_index]

        st.subheader("Best Resume Evidence")

        st.write(best_evidence)

        st.metric(
            "Semantic Similarity",
            f"{best_score:.3f}"
        )
