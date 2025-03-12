import streamlit as st
from PyPDF2 import PdfReader
import docx

# Hide the default Streamlit footer
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.title("📝 Essay Grading Tool")

# # Function to extract text from uploaded files
# def extract_text_from_file(uploaded_file):
#     if uploaded_file.type == "application/pdf":
#         pdf_reader = PdfReader(uploaded_file)
#         text = ""
#         for page in pdf_reader.pages:
#             text += page.extract_text()
#         return text
#     elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
#         doc = docx.Document(uploaded_file)
#         text = "\n".join([para.text for para in doc.paragraphs])
#         return text
#     else:
#         return None

# # Set page configuration
# st.set_page_config(
#     page_title="Essay Grading Tool",
#     page_icon="📝",
#     layout="wide"
# )

# # Title and description
# st.title("📝 Essay Grading Tool")
# st.markdown("""
# Welcome to the Essay Grading Tool! This platform helps teachers save time by instantly grading essays, 
# providing constructive feedback, and detecting plagiarism. It ensures fair and personalized learning for students.
# """)

# # Sidebar for navigation
# st.sidebar.title("Navigation")
# option = st.sidebar.radio("Select an Option", ["All-in-One", "Grade Only", "Plagiarism Only", "Feedback Only"])

# # Input options: File upload or text input
# essay_input_method = st.radio("Choose Input Method", ["Upload File", "Paste Text"])
# essay_text = ""

# if essay_input_method == "Upload File":
#     uploaded_file = st.file_uploader("Upload Essay (PDF or DOCX)", type=["pdf", "docx"])
#     if uploaded_file:
#         st.success(f"File '{uploaded_file.name}' uploaded successfully!")
#         # Extract text from the uploaded file
#         essay_text = extract_text_from_file(uploaded_file)
#         if essay_text:
#             st.subheader("Extracted Text")
#             st.text_area("Preview of the Extracted Text", essay_text, height=300)
#         else:
#             st.error("Unsupported file format or unable to extract text.")
# else:
#     essay_text = st.text_area("Paste Essay Text Here", height=300)

# # Submit button for processing
# if st.button("Process"):
#     if not essay_text:
#         st.error("Please provide an essay to process.")
#     else:
#         # Placeholder for model logic
#         grade = "A"  # Replace with actual grade logic
#         plagiarism_percentage = 15  # Replace with actual plagiarism logic
#         sources = ["[Link 1](https://example.com)", "[Link 2](https://example.com)"]  # Replace with actual sources
#         feedback = [
#             "- **Strengths:** Clear thesis statement and well-organized structure.",
#             "- **Areas for Improvement:** Expand on key arguments with more evidence."
#         ]

#         # Display results based on the selected mode
#         if option == "All-in-One":
#             st.header("All-in-One Results")
#             st.subheader("Grade")
#             st.write(f"**Grade:** {grade}")
#             st.subheader("Plagiarism Check")
#             st.write(f"**Plagiarism Detected:** {plagiarism_percentage}%")
#             st.write("**Sources:** " + ", ".join(sources))
#             st.subheader("Feedback")
#             for point in feedback:
#                 st.write(point)
        
#         elif option == "Grade Only":
#             st.header("Grade Results")
#             st.subheader("Grade")
#             st.write(f"**Grade:** {grade}")
        
#         elif option == "Plagiarism Only":
#             st.header("Plagiarism Check Results")
#             st.subheader("Plagiarism Detection")
#             st.write(f"**Plagiarism Detected:** {plagiarism_percentage}%")
#             st.write("**Sources:** " + ", ".join(sources))
        
#         elif option == "Feedback Only":
#             st.header("Feedback Results")
#             st.subheader("Constructive Feedback")
#             for point in feedback:
#                 st.write(point)

# # # Footer
# # st.markdown("---")
# # st.markdown("Developed with ❤️ by Karthik Gavini")

