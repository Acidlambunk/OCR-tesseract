import cv2
import pytesseract
import pandas as pd
import re

# Path to tesseract executable (modify for your system)
pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'

# Answer key: question number mapped to correct answer
answer_key = {
    1: 'A',
    2: 'B',
    3: 'C',
    4: 'D',
    5: 'A',
    6: 'A',
    7: 'B',
    8: 'C',
    9: 'D',
    10: 'A'
}

def preprocess_image(image_path):
    """Loads and preprocesses the image for OCR with noise reduction and contrast enhancement."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Resize image to make small text more readable
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    
    # Apply a median blur to reduce noise
    img = cv2.medianBlur(img, 5)
    
    # Apply adaptive thresholding for better contrast
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                cv2.THRESH_BINARY_INV, 11, 2)
    
    # Sharpen the image for better OCR detection
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    
    return img

def fix_ocr_errors(ocr_text):
    """Fixes common OCR misreads (like 'Qi' -> 'Q1')."""
    # Replace 'Qi' with 'Q1' and any other known common OCR errors
    fixed_text = re.sub(r'Qi', 'Q1', ocr_text)
    return fixed_text

def extract_answers(image_path):
    """Extracts answers from the image using OCR."""
    # Preprocess the image
    processed_img = preprocess_image(image_path)
    
    # OCR configuration to improve accuracy
    custom_config = r'--oem 3 --psm 6'
    
    # Run OCR on the image with the custom config
    ocr_result = pytesseract.image_to_string(processed_img, config=custom_config)
    
    # Fix common OCR errors
    ocr_result = fix_ocr_errors(ocr_result)
    
    # Print OCR result for debugging
    print("OCR Result (after fixing):\n", ocr_result)
    
    # Assuming answers are written like 'Q1: A', extract them:
    answers = {}
    for line in ocr_result.splitlines():
        if line.strip():  # Ignore empty lines
            parts = line.split(':')
            if len(parts) == 2:
                try:
                    # Try converting to int
                    question_number = int(parts[0].strip().replace('Q', ''))
                    answer = parts[1].strip().upper()
                    answers[question_number] = answer
                except ValueError:
                    # Skip lines that can't be processed
                    print(f"Skipping invalid line: {line}")
    return answers

def grade(answers, answer_key):
    """Compares the student's answers to the answer key and calculates score."""
    total_questions = len(answer_key)
    correct_answers = sum(1 for q in answer_key if answer_key[q] == answers.get(q, ''))
    
    # Result DataFrame
    results = pd.DataFrame({
        'Question': list(answer_key.keys()),
        'Correct Answer': [answer_key[q] for q in answer_key],
        'Student Answer': [answers.get(q, 'Not answered') for q in answer_key],
        'Result': ['Correct' if answer_key[q] == answers.get(q, '') else 'Incorrect' for q in answer_key]
    })
    
    score = (correct_answers / total_questions) * 100
    return results, score

# Usage example:
image_path = 'test_answer_sheet.jpg'  # Path to the scanned image of the answer sheet
student_answers = extract_answers(image_path)
results_df, final_score = grade(student_answers, answer_key)

# Display results
print("Grading Results:\n", results_df)
print(f"Final Score: {final_score:.2f}%")
