import cv2
import pytesseract
import numpy as np
from matplotlib import pyplot as plt

pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

def ocr_opencv(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 5, 75, 75)  # keep edges, remove noise
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Morphological Closing to fill gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    clean = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)

    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1); plt.title("Thresholded"); plt.imshow(th, cmap='gray'); plt.axis('off')
    plt.subplot(1,2,2); plt.title("Cleaned"); plt.imshow(clean, cmap='gray'); plt.axis('off')
    plt.show()

    config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(clean, config=config)
    return text.strip()

# Run it
output = ocr_opencv(r'C:\Users\Rihan\Desktop\richu\copy\WhatsApp Image 2025-06-18 at 23.13.34_616388d4.jpg')
print("📝 Extracted Text:\n", output)
