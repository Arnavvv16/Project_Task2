from PIL import Image, ImageFilter, ImageEnhance
import pytesseract
im_file = "ocr-demo2.png"
im = Image.open(im_file)
# print(im)
# print(im.size)
# im.show()

def preprocess_image(img_path):
    """Apply image preprocessing for better OCR results"""
    image = Image.open(img_path)
    image = image.convert("L")                        # Grayscale
    image = image.filter(ImageFilter.SHARPEN)         # Sharpen image
    image = ImageEnhance.Contrast(image).enhance(2)   # Increase contrast
    return image

processed_image = preprocess_image(im_file)
text = pytesseract.image_to_string(processed_image)
print(text)

#Save the extracted text to a file
with open("ocr_output_demo2.txt", "w", encoding="utf-8") as f:
    f.write(text)
