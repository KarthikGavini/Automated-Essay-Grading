# feedback_system.py

import language_tool_python
from textstat import textstat
import spacy

# Initialize LanguageTool for grammar and spelling checks
tool = language_tool_python.LanguageTool('en-US')

# Load the spaCy English model for sentence tokenization
nlp = spacy.load("en_core_web_sm")

def check_grammar_and_spelling(text):
    matches = tool.check(text)
    feedback = []
    if matches:
        feedback.append(f"- Grammar/Spelling Issues Found: {len(matches)}")
        for match in matches[:5]:  # Show up to 5 issues
            feedback.append(f"  - Issue: '{match.context}' | Suggestion: {match.message}")
    else:
        feedback.append("- No grammar or spelling errors detected.")
    return feedback


def evaluate_readability(text):
    feedback = []
    flesch_reading_score = textstat.flesch_reading_ease(text)
    gunning_fog_score = textstat.gunning_fog(text)
    smog_score = textstat.smog_index(text)

    # Interpret Flesch Reading Ease score
    if flesch_reading_score >= 90:
        feedback.append("- Readability: Very easy to read. Great for a general audience.")
    elif flesch_reading_score >= 60:
        feedback.append("- Readability: Fairly easy to read. Suitable for most readers.")
    else:
        feedback.append("- Readability: Difficult to read. Consider simplifying your language.")

    # Interpret Gunning Fog score
    if gunning_fog_score <= 12:
        feedback.append("- Complexity: Easy to understand for high school students.")
    else:
        feedback.append("- Complexity: Difficult to understand. Aim for a lower Gunning Fog score.")

    # Interpret SMOG Index
    if smog_score <= 12:
        feedback.append("- SMOG Index: Suitable for high school-level readers.")
    else:
        feedback.append("- SMOG Index: Too complex for high school-level readers. Simplify your language.")

    return feedback


def analyze_sentence_length(text):
    feedback = []
    doc = nlp(text)  # Process the text with spaCy
    sentences = [sent.text for sent in doc.sents]  # Extract sentences using spaCy

    for i, sentence in enumerate(sentences, 1):
        word_count = len(sentence.split())
        if word_count > 30:
            feedback.append(f"- Sentence {i}: Too long ({word_count} words). Break it into shorter sentences.")
        elif word_count < 5:
            feedback.append(f"- Sentence {i}: Too short ({word_count} words). Add more detail.")
    return feedback


def analyze_paragraph_structure(text):
    feedback = []
    paragraphs = text.split('\n\n')  # Split by double newlines to identify paragraphs
    if len(paragraphs) < 3:
        feedback.append("- Structure: Your essay has fewer than 3 paragraphs. Consider breaking it into more sections.")
    for i, paragraph in enumerate(paragraphs, 1):
        sentences = paragraph.split('.')  # Split by periods to count sentences
        if len(sentences) < 3:
            feedback.append(f"- Paragraph {i}: This paragraph has fewer than 3 sentences. Add more detail.")
    return feedback


def check_word_count(text):
    feedback = []
    word_count = len(text.split())
    if word_count < 250:
        feedback.append(f"- Word Count: The essay is too short ({word_count} words). Aim for at least 250 words.")
    elif word_count > 1000:
        feedback.append(f"- Word Count: The essay is quite long ({word_count} words). Ensure all content is relevant.")
    else:
        feedback.append(f"- Word Count: The essay is within the recommended range ({word_count} words).")
    return feedback


def generate_feedback(text):
    feedback = []

    # Grammar and spelling feedback
    feedback.append("=== Grammar and Spelling ===")
    feedback.extend(check_grammar_and_spelling(text))

    # Readability feedback
    feedback.append("\n=== Readability ===")
    feedback.extend(evaluate_readability(text))

    # Sentence length feedback
    feedback.append("\n=== Sentence Length ===")
    feedback.extend(analyze_sentence_length(text))

    # Paragraph structure feedback
    feedback.append("\n=== Paragraph Structure ===")
    feedback.extend(analyze_paragraph_structure(text))

    # Word count feedback
    feedback.append("\n=== Word Count ===")
    feedback.extend(check_word_count(text))

    return "\n".join(feedback)
