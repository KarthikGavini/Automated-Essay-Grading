# app.py

import streamlit as st
from PyPDF2 import PdfReader
import docx
from transformers import AutoTokenizer, AutoModelForSequenceClassification, T5ForConditionalGeneration, T5Tokenizer
import torch
from feedback import generate_feedback  # Import feedback generation logic
from plagiarism_model import check_plagiarism  # Import plagiarism detection logic

# Load the trained model and tokenizer for grading
MODEL_PATH = "./final_model"  # Update this path if needed
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

# Load the T5 model for prompt-based tasks
PROMPT_MODEL_PATH = "./T3"  # Path to the T5 model
prompt_tokenizer = T5Tokenizer.from_pretrained(PROMPT_MODEL_PATH)
prompt_model = T5ForConditionalGeneration.from_pretrained(PROMPT_MODEL_PATH)

# Move models to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
prompt_model.to(device)

# Mapping numerical grades to letter grades
GRADE_MAPPING = {
    6: "A",
    5: "B",
    4: "C",
    3: "D",
    2: "E",
    1: "F"
}

# Function to extract text from uploaded files
def extract_text_from_file(uploaded_file):
    if uploaded_file.type == "application/pdf":
        pdf_reader = PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(uploaded_file)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text
    else:
        return None

# Function to predict essay score using the classification model
def predict_score(essay, model, tokenizer):
    inputs = tokenizer(
        essay,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )
    inputs = {key: val.to(device) for key, val in inputs.items()}  # Move inputs to the same device as the model
    with torch.no_grad():  # Disable gradient computation for inference
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = logits.argmax(-1).item()
    numerical_grade = predicted_class + 1  # Convert back to 1–6 scale
    letter_grade = GRADE_MAPPING[numerical_grade]  # Map to letter grade
    return letter_grade

# Function to generate output using the T5 model
def generate_prompt_based_output(prompt, essay, model, tokenizer, device):
    """
    Generates output (e.g., scores or feedback) for the given prompt and essay.
    
    Args:
        prompt (str): The task prompt (e.g., "Score this essay:" or "Generate feedback for this essay:").
        essay (str): The essay text to evaluate.
        model: The T5 model.
        tokenizer: The T5 tokenizer.
        device: The device (CPU or GPU).
    
    Returns:
        str: The generated output (e.g., scores or feedback).
    """
    # Combine the prompt and essay into a single input string
    input_text = f"{prompt} {essay}"
    
    # Tokenize the input
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True, padding=True).to(device)
    
    # Generate output
    outputs = model.generate(**inputs, max_length=128)
    
    # Decode the output into a human-readable string
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return output_text

# Set page configuration
st.set_page_config(
    page_title="Essay Grading Tool",
    page_icon="📝",
    layout="wide"
)

# Title and description
st.title("📝 Essay Grading Tool")
st.markdown("""
Welcome to the Essay Grading Tool! This platform helps teachers save time by instantly grading essays, 
providing constructive feedback, and detecting plagiarism. It ensures fair and personalized learning for students.
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
option = st.sidebar.radio("Select an Option", ["All-in-One", "Grade Only", "Plagiarism Only", "Feedback Only", "Prompt-Based"])

# Input options: Prompt and Essay
if option == "Prompt-Based":
    prompt = st.text_input("Enter Prompt (e.g., 'Score this essay:' or 'Generate feedback for this essay:')", value="Score this essay:")
else:
    prompt = None

essay_input_method = st.radio("Choose Input Method", ["Upload File", "Paste Text"])
essay_text = ""

if essay_input_method == "Upload File":
    uploaded_file = st.file_uploader("Upload Essay (PDF or DOCX)", type=["pdf", "docx"])
    if uploaded_file:
        st.success(f"File '{uploaded_file.name}' uploaded successfully!")
        # Extract text from the uploaded file
        essay_text = extract_text_from_file(uploaded_file)
        if essay_text:
            st.subheader("Extracted Text")
            st.text_area("Preview of the Extracted Text", essay_text, height=300)
        else:
            st.error("Unsupported file format or unable to extract text.")
else:
    essay_text = st.text_area("Paste Essay Text Here", height=300)

# Submit button for processing
if st.button("Process"):
    if not essay_text:
        st.error("Please provide an essay to process.")
    else:
        if option in ["All-in-One", "Grade Only"]:
            # Predict grade using the classification model
            grade = predict_score(essay_text, model, tokenizer)

            if option == "All-in-One":
                # Check plagiarism using the plagiarism detection logic
                similarity_score, risk_level, closest_text = check_plagiarism(essay_text)

                # Generate dynamic feedback using the Enhanced Rule-Based Feedback System
                feedback = generate_feedback(essay_text)

                # Display results
                st.header("🌟 All-in-One Results")
                st.subheader("🎓 Grade")
                st.write(f"**Grade:** {grade}")
                st.subheader("🔍 Plagiarism Check")
                st.write(f"**Plagiarism Detected:** {similarity_score:.2f}%")
                st.write(f"**Risk Level:** {risk_level}")
                st.write(f"**Most Similar Text:** {closest_text}")
                st.subheader("📝 Feedback")
                st.text(feedback)
            elif option == "Grade Only":
                st.header("🎓 Grade Results")
                st.subheader("Grade")
                st.write(f"**Grade:** {grade}")

        elif option == "Plagiarism Only":
            # Check plagiarism using the plagiarism detection logic
            similarity_score, risk_level, closest_text = check_plagiarism(essay_text)

            # Display results
            st.header("🔍 Plagiarism Check Results")
            st.subheader("Plagiarism Detection")
            st.write(f"**Plagiarism Detected:** {similarity_score:.2f}%")
            st.write(f"**Risk Level:** {risk_level}")
            st.write(f"**Most Similar Text:** {closest_text}")

        elif option == "Feedback Only":
            # Generate dynamic feedback using the Enhanced Rule-Based Feedback System
            feedback = generate_feedback(essay_text)

            # Display results
            st.header("📝 Feedback Results")
            st.subheader("Constructive Feedback")
            st.text(feedback)

        elif option == "Prompt-Based":
            if not prompt.strip():
                st.error("Please provide a valid prompt.")
            else:
                # Generate output using the T5 model
                output = generate_prompt_based_output(prompt, essay_text, prompt_model, prompt_tokenizer, device)

                # Parse the output into structured components
                try:
                    total_score = output.split("Total Score:")[1].split()[0]
                    total_score = f"{float(total_score)}/15"  # Format as "5/15"
                    content_feedback = output.split("Content Feedback:")[1].split("Organization Feedback:")[0].strip()
                    organization_feedback = output.split("Organization Feedback:")[1].split("Language Feedback:")[0].strip()
                    language_feedback = output.split("Language Feedback:")[1].strip()

                    # Display results
                    st.header("📝 Prompt-Based Results")
                    st.subheader("🌟 Score")
                    st.write(f"**Total Score:** {total_score}")
                    st.subheader("📝 Feedback")
                    st.write(f"**Content Feedback:** {content_feedback}")
                    st.write(f"**Organization Feedback:** {organization_feedback}")
                    st.write(f"**Language Feedback:** {language_feedback}")
                except Exception as e:
                    st.error("Error parsing the model output. Please ensure the model generates output in the expected format.")
                    st.write(f"Raw Output: {output}")
