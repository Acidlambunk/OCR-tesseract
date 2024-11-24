import cv2
import numpy as np

def create_test_image(output_path):
    """Creates a high-quality sample answer sheet image for OCR testing."""
    # Create a higher resolution white background image
    img = np.ones((2000, 1600, 3), dtype=np.uint8) * 255  # Increased size for better quality

    # Define the font and start point for writing answers
    font = cv2.FONT_HERSHEY_SIMPLEX
    start_y = 100  # Start further down for more space
    font_scale = 2  # Increased font size for better clarity
    color = (0, 0, 0)  # Black text
    thickness = 3  # Thicker text for better OCR readability

    # Sample answers
    answers = {
        1: 'A',
        2: 'B',
        3: 'C',
        4: 'D',
        5: 'A',
        6: 'B',
        7: 'B',
        8: 'C',
        9: 'D',
        10: 'A',
    }

    # Write the answers onto the image
    for question, answer in answers.items():
        text = f"Q{question}: {answer}"
        img = cv2.putText(img, text, (100, start_y), font, font_scale, color, thickness, cv2.LINE_AA)  # Added anti-aliasing
        start_y += 100  # Increased space between lines for clarity

    # Save the image
    cv2.imwrite(output_path, img)

# Usage
output_image_path = 'test_answer_sheet.jpg'
create_test_image(output_image_path)
print(f"High-quality test image created at: {output_image_path}")
